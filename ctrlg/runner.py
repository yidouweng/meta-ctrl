"""Command-line plumbing shared by the four generation drivers."""
import json
import os
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "models/Meta-Llama-3-8B-Instruct"

# How a task was decoded, written into every generation record:
#   mode           "two_level", or "syntax_mask" (γ alone)
#   fallback       why a two-level arm used the syntax mask: "beta size limit" or
#                  "no accepting plan within the maximum plan length"
#   goals_dropped  goals the semantic automaton dropped to fit its limits
#   dp_budget      actions (atoms) the lookahead covered
DECODING_FIELDS = ("mode", "fallback", "goals_dropped", "dp_budget")
FALLBACK_SIZE = "beta size limit"
FALLBACK_NO_PLAN = "no accepting plan within the maximum plan length"


def decoding_fields(stats: Dict) -> Dict:
    """The DECODING_FIELDS present in a driver's per-task stats."""
    return {k: stats[k] for k in DECODING_FIELDS if stats.get(k) is not None}


def add_common_args(ap, *, prompts: str, meta_hmm: str, max_budget: int, max_new_tokens: int = 2048) -> None:
    ap.add_argument("--prompt_file", default=prompts)
    ap.add_argument("--prompt_indices", default="", help="comma-separated prompt indices (default: all)")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n_prompts", type=int, default=0, help="0 = all prompts from --start_idx")
    ap.add_argument("--base_model", default=BASE_MODEL, help="HF id or local path of the base LM")
    ap.add_argument("--cuda_device", default="0")
    ap.add_argument("--semantic_hmm_path", default=meta_hmm, help="meta-token HMM directory (trained or uniform)")
    ap.add_argument("--max_budget", type=int, default=max_budget, help="maximum plan length in actions")
    ap.add_argument("--max_new_tokens", type=int, default=max_new_tokens)
    ap.add_argument("--ctrlg_scale", type=float, default=1.0, help="lookahead calibration exponent c")
    ap.add_argument("--beta_hard_mask", action="store_true",
                    help="ablation 'goal reachability': allow an action iff the goal stays reachable")
    ap.add_argument("--syntax_only", action="store_true", help="ablation: syntax mask only, no lookahead")
    ap.add_argument("--no_resume", action="store_true", help="regenerate prompts already in the output")
    ap.add_argument("--debug", action="store_true")


def load_lm(base_model: str, cuda_device: str):
    """Tokenizer, model, logit width and device."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_device)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True).to(device)
    model.eval()
    return tokenizer, model, int(model.config.vocab_size), device


def eot_id(tokenizer) -> int:
    """End-of-turn token that closes a plan: `<|eot_id|>` for Llama 3, else the tokenizer's EOS."""
    vocab = tokenizer.get_vocab()
    return int(vocab["<|eot_id|>"]) if "<|eot_id|>" in vocab else int(tokenizer.eos_token_id)


def _indices(args, n: int) -> List[int]:
    if args.prompt_indices:
        return [int(x) for x in args.prompt_indices.split(",") if x.strip()]
    end = n if args.n_prompts <= 0 else min(n, args.start_idx + args.n_prompts)
    return list(range(args.start_idx, end))


def run_prompts(args, generate: Callable[[Dict], Dict], *, output_json: str = None, output_dir: str = None,
                include_failures: bool = False) -> None:
    """Run `generate` over the selected prompts.

    output_json (VirtualHome): one JSON list; finished prompts are also
    appended to <output>.partial.jsonl so an interrupted run resumes. With
    `include_failures`, failed prompts enter the list with their (empty) plan.
    output_dir (BEHAVIOR): prompt_<idx>.json per prompt plus batch_summary.json;
    prompts whose file exists are skipped.
    """
    prompts = json.load(open(args.prompt_file))
    idxs = _indices(args, len(prompts))
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(out) + ".partial.jsonl")
        done = {}
        if partial.exists() and not args.no_resume:
            for line in partial.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if not rec.get("error"):  # failed prompts are retried
                        done[rec.get("identifier")] = rec
        results, failures = [], []
        for i in idxs:
            p = prompts[i]
            rec = done.get(p.get("identifier"))
            if rec is None:
                print(f">>> {i} {p.get('identifier')}", flush=True)
                rec = generate(p)
                rec.update(decoding_fields(rec.get("stats") or {}))
                with open(partial, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            (failures if rec.get("error") else results).append(rec)
        if include_failures:
            results = [{"identifier": r["identifier"], "llm_output": r["llm_output"],
                        **decoding_fields(r.get("stats") or {})} for r in results + failures]
        json.dump(results, open(out, "w"), indent=2)
        failures_path = out.with_suffix(".failures.json")
        if failures:
            json.dump(failures, open(failures_path, "w"), indent=2)
        elif failures_path.exists():
            failures_path.unlink()
        print(f"{len(results)} plans written to {out}, {len(failures)} failures")
        if failures:  # a failed prompt is not a plan; fail loudly instead of evaluating fewer tasks
            raise SystemExit(f"{len(failures)} prompts failed (see {out.with_suffix('.failures.json')}); rerun to retry them")
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_path = f"{output_dir}/batch_summary.json"
    earlier = {e["prompt_idx"]: e for e in json.load(open(summary_path))} if os.path.exists(summary_path) else {}
    summary = []
    for i in idxs:
        p = prompts[i]
        path = f"{output_dir}/prompt_{i}.json"
        if os.path.exists(path) and not args.no_resume:
            if i in earlier:  # keep the stats of prompts finished in an earlier run
                summary.append(earlier[i])
            continue
        print(f">>> {i} {p['identifier']}", flush=True)
        t0 = time.time()
        try:
            rec = generate(p)
            json.dump([{"identifier": p["identifier"], "llm_output": rec["llm_output"],
                        **decoding_fields(rec.get("stats", {}))}], open(path, "w"), indent=2)
            entry = dict(rec.get("stats", {}), prompt_idx=i, identifier=p["identifier"])
        except Exception as e:
            traceback.print_exc()
            entry = {"prompt_idx": i, "identifier": p["identifier"], "mode": "error", "error": str(e)}
        entry["elapsed"] = round(time.time() - t0, 1)
        summary.append(entry)
        json.dump(summary, open(summary_path, "w"), indent=2)
    json.dump(summary, open(summary_path, "w"), indent=2)
    failed =[e["prompt_idx"] for e in summary if e.get("mode") == "error"]
    if failed:  # a failed prompt has no plan file; fail loudly instead of evaluating fewer tasks
        raise SystemExit(f"prompts {failed} failed (see {output_dir}/batch_summary.json); rerun to retry them")

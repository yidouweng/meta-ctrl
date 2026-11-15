#!/usr/bin/env python3
"""Syntax-only baseline with XGrammar via vLLM, BEH-SD, Llama-3-8B-Instruct, greedy.

Per task, the grammar admits the output format of our BEH-SD gamma automaton
(ctrlg/gamma/topdown/behavior_sd_twolevel.py):

    {"output": ["pred(inst)", "not pred(inst1, inst2)", ...]}     (>= 1 element)

one (optionally negated, 'not ') atom per list element (the gamma offers no
' and ' joins), the task's usable predicates (usable_predicates) at their arity
(SD_PREDICATES), and the scene's object instances, written with ANY
tokenization. JSON whitespace between structural tokens is free (bounded, as in
the AS XGrammar scripts); atom strings are spelled exactly as the gamma spells
them ("pred(a.1, b.2)").

By default the grammar follows the prompt's own output format, as in the
paper's run (Appendix N): atoms joined by ' and ' / ' or ' inside a list
element. --gamma_format restricts it to the gamma's format above. Any scene object may fill any argument slot;
--affordances gamma additionally applies the gamma's per-argument affordance
sets, so that with --gamma_format the grammar's language equals the gamma's.

Prompt: the BEH-SD system prompt (data/prompts/system_prompts.json) and the
benchmark prompt merged into one user turn; max_tokens 2048.

    python baselines/xgrammar_beh_sd.py     # -> outputs/xgrammar_beh_sd
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from ctrlg.gamma.topdown.behavior_sd_twolevel import build_beh_sd_twolevel_state_machine, SD_PREDICATES

M = "models/Meta-Llama-3-8B-Instruct"
PROMPTS = "data/prompts/behavior/subgoal_decomposition/subgoal_decomposition_prompts.json"
SYSTEM = "data/prompts/system_prompts.json"
WS = 'ws ::= ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n])?)?)?)?)?)?)?)?'


def q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def system_prompt():
    return "\n".join(json.load(open(SYSTEM))["modules"]["behavior"]["SD"]["system_prompt_lines"]).strip()


def grammar_for(sm, affordances="none", allow_and=False, allow_or=False):
    inst_all = sorted(sm.all_instances)
    conj = ('((" and " | " or ") ("not ")? atom)*' if allow_or
            else '(" and " ("not ")? atom)*' if allow_and else "")
    lines = ['root ::= "{" ws "\\"output\\"" ws ":" ws "[" ws item (ws "," ws item)* ws "]" ws "}"',
             WS, 'item ::= "\\"" ("not ")? atom ' + conj + ' "\\""',
             "obj ::= " + " | ".join(q(o) for o in inst_all)]
    alts, extra = [], []

    def rule(pred, role):
        if affordances == "none":
            return "obj"
        bases = sm.affordances.get((pred, role))
        if bases is None:
            return "obj"
        insts = sorted(i for b in bases for i in sm.objects.get(b, []))
        if not insts:
            return None
        name = f"a_{pred}_{role}"
        extra.append(f"{name} ::= " + " | ".join(q(i) for i in insts))
        return name

    for p in sorted(sm.usable_predicates):
        ar = SD_PREDICATES[p]
        args = [rule(p, k) for k in range(ar)]
        if any(a is None for a in args):
            continue
        alts.append(q(p + "(") + " " + ' ", " '.join(args) + ' ")"')
    lines.append("atom ::= " + " | ".join("(" + x + ")" for x in alts))
    return "\n".join(lines + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="every stride-th task (1 = all)")
    ap.add_argument("--out_dir", default="outputs/xgrammar_beh_sd")
    ap.add_argument("--affordances", choices=["none", "gamma"], default="none")
    ap.add_argument("--gamma_format", action="store_true",
                    help="one atom per list element, as the gamma writes plans (instead of the prompt's format)")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--dump_grammars", default=None, help="write the per-task grammars to this JSON and exit")
    ap.add_argument("--no_grammar", action="store_true", help="unconstrained Llama, same prompts")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                    help="fraction of GPU memory vLLM may use (does not change greedy outputs)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(M)
    eos = tok.convert_tokens_to_ids("<|eot_id|>")
    sysp = system_prompt()
    allP = json.load(open(PROMPTS))
    idx = list(range(0, len(allP), args.stride))
    prompts, grammars = [], []
    for i in idx:
        r = allP[i]
        sm = build_beh_sd_twolevel_state_machine(tok, r["llm_prompt"], eos)
        grammars.append(grammar_for(sm, args.affordances, not args.gamma_format, not args.gamma_format))
        prompts.append(tok.apply_chat_template([{"role": "user", "content": sysp + "\n\n" + r["llm_prompt"]}],
                                               tokenize=False, add_generation_prompt=True))
    if args.dump_grammars:
        json.dump([{"identifier": allP[i]["identifier"], "grammar": g} for i, g in zip(idx, grammars)],
                  open(args.dump_grammars, "w"), indent=1)
        print("dumped", len(grammars), "grammars to", args.dump_grammars)
        return
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    params = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                             guided_decoding=None if args.no_grammar else GuidedDecodingParams(grammar=g, backend="xgrammar"))
              for g in grammars]
    llm = LLM(model=M, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=8192, enforce_eager=True)
    outs = llm.generate(prompts, params)
    os.makedirs(args.out_dir, exist_ok=True)
    merged, meta = [], []
    for i, o in zip(idx, outs):
        rec = {"identifier": allP[i]["identifier"], "llm_output": o.outputs[0].text}
        json.dump([rec], open(os.path.join(args.out_dir, f"prompt_{i}.json"), "w"))
        merged.append(rec)
        meta.append(dict(rec, finish_reason=o.outputs[0].finish_reason, n_tokens=len(o.outputs[0].token_ids)))
    json.dump(merged, open(os.path.join(args.out_dir, "merged.json"), "w"))
    json.dump(meta, open(os.path.join(args.out_dir, "gen_meta.json"), "w"), indent=1)
    print("wrote", args.out_dir, len(merged))
    for x in merged[:2]:
        print(repr(x["llm_output"][:300]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Syntax-only baseline with XGrammar via vLLM, BEH-AS, Llama-3-8B-Instruct, greedy.

Per task, the grammar admits the BEH-AS JSON list format with the task's viable
action names and any scene object instance, written with ANY tokenization.
Vocabulary comes from the same prompt parser as our gamma.

Prompt: the benchmark prompt alone as the user turn (no system prompt);
max_tokens 1024. These defaults reproduce the paper's run (Appendix N):

    python baselines/xgrammar_beh_as.py     # -> outputs/xgrammar_beh_as/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from ctrlg.gamma.topdown.behavior_twolevel import build_beh_twolevel_state_machine

M = "models/Meta-Llama-3-8B-Instruct"
PROMPTS = "data/prompts/behavior/action_sequencing/action_sequence_prompts.json"


def q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def grammar_for(sm):
    objs = " | ".join(q('"' + o + '"') for o in sorted(sm.all_instances))
    acts = " | ".join(q('"' + a + '"') for a in sorted(sm.viable_actions))
    return "\n".join([
        'root ::= "[" ws item (ws "," ws item)* ws "]"',
        'ws ::= ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n])?)?)?)?)?)?)?)?',
        'item ::= "{" ws "\\"action\\"" ws ":" ws act ws "," ws "\\"object\\"" ws ":" ws obj ws "}"',
        "act ::= " + acts,
        "obj ::= " + objs,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="every stride-th task (1 = all)")
    ap.add_argument("--out_dir", default="outputs/xgrammar_beh_as")
    ap.add_argument("--system_prompt", action="store_true",
                    help="prepend the BEH-AS system prompt to the user turn "
                         "(the paper's run uses the bare benchmark prompt)")
    ap.add_argument("--no_grammar", action="store_true", help="unconstrained Llama, same prompts")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                    help="fraction of GPU memory vLLM may use (does not change greedy outputs)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(M)
    eos = tok.convert_tokens_to_ids("<|eot_id|>")
    allP = json.load(open(PROMPTS))
    idx = list(range(0, len(allP), args.stride))
    prompts, params = [], []
    for i in idx:
        r = allP[i]
        sm = build_beh_twolevel_state_machine(tok, r["llm_prompt"], eos)
        user = r["llm_prompt"]
        if args.system_prompt:
            sp = json.load(open("data/prompts/system_prompts.json"))
            user = "\n".join(sp["modules"]["behavior"]["AS"]["system_prompt_lines"]).strip() + "\n\n" + user
        prompts.append(tok.apply_chat_template([{"role": "user", "content": user}],
                                               tokenize=False, add_generation_prompt=True))
        params.append(SamplingParams(temperature=0.0, max_tokens=1024,
                                     guided_decoding=None if args.no_grammar else GuidedDecodingParams(grammar=grammar_for(sm), backend="xgrammar")))
    llm = LLM(model=M, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=8192, enforce_eager=True)
    outs = llm.generate(prompts, params)
    os.makedirs(args.out_dir, exist_ok=True)
    merged = []
    for i, o in zip(idx, outs):
        rec = {"identifier": allP[i]["identifier"], "llm_output": o.outputs[0].text}
        json.dump(rec, open(os.path.join(args.out_dir, f"prompt_{i}.json"), "w"))
        merged.append(rec)
    json.dump(merged, open(os.path.join(args.out_dir, "merged.json"), "w"))
    print("wrote", args.out_dir, len(merged))
    for x in merged[:2]:
        print(repr(x["llm_output"][:200]))


if __name__ == "__main__":
    main()

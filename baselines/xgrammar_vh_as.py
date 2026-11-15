#!/usr/bin/env python3
"""Syntax-only baseline with an off-the-shelf grammar-constrained decoder
(XGrammar via vLLM), VH-AS, Llama-3-8B-Instruct, greedy.

Per task, the grammar admits the VH-AS JSON plan format with the task's viable
action names (correct arity) and the scene's (name, id) objects, written with
ANY tokenization (unlike our canonical gamma mask). Vocabulary comes from the
same prompt parser as our gamma (build_vh_state_machine), so the only
difference from our gamma mask is canonical vs any tokenization.

Prompt: the benchmark prompt alone as the user turn (no system prompt);
max_tokens 512. These defaults reproduce the paper's run (Appendix N):

    python baselines/xgrammar_vh_as.py      # -> outputs/xgrammar_vh_as.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from ctrlg.gamma.topdown.virtualhome import build_vh_state_machine

M = "models/Meta-Llama-3-8B-Instruct"
PROMPTS = "data/prompts/virtualHome/action_sequencing/helm_prompt.json"


def q(s):  # GBNF string literal
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def grammar_for(sm):
    objs = []
    for name, ids in sorted(sm.scene_inventory.items()):
        for i in ids:
            objs.append(q('"' + str(name) + '"') + " ws \",\" ws " + q('"' + str(i) + '"'))
    lines = ['root ::= "{" ws pair (ws "," ws pair)* ws "}"',
             'ws ::= ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n])?)?)?)?)?)?)?)?',
             "obj ::= " + " | ".join(objs)]
    alts = []
    for a in sorted(sm.viable_actions):
        ar = sm.action_arity.get(a, 1)
        key = q('"' + a + '"') + ' ws ":" ws "[" ws '
        if ar == 0:
            alts.append(key + '"]"')
        elif ar == 1:
            alts.append(key + 'obj ws "]"')
        else:
            alts.append(key + 'obj ws "," ws obj ws "]"')
    lines.append("pair ::= " + " | ".join("(" + x + ")" for x in alts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="every stride-th task (1 = all)")
    ap.add_argument("--out", default="outputs/xgrammar_vh_as.json")
    ap.add_argument("--no_grammar", action="store_true", help="unconstrained Llama, same prompts")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                    help="fraction of GPU memory vLLM may use (does not change greedy outputs)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(M)
    eos = tok.convert_tokens_to_ids("<|eot_id|>")
    P = json.load(open(PROMPTS))[:: args.stride]
    prompts, params = [], []
    for r in P:
        sm = build_vh_state_machine(tokenizer=tok, prompt=r["llm_prompt"], eos_token_id=eos, max_ws_tokens=5)
        g = grammar_for(sm)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": r["llm_prompt"]}],
                                               tokenize=False, add_generation_prompt=True))
        params.append(SamplingParams(temperature=0.0, max_tokens=512,
                                     guided_decoding=None if args.no_grammar else GuidedDecodingParams(grammar=g, backend="xgrammar")))
    llm = LLM(model=M, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=8192, enforce_eager=True)
    outs = llm.generate(prompts, params)
    res = [{"identifier": r["identifier"], "llm_output": o.outputs[0].text} for r, o in zip(P, outs)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out, len(res))
    for x in res[:3]:
        print(repr(x["llm_output"][:200]))


if __name__ == "__main__":
    main()

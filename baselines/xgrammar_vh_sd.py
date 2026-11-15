#!/usr/bin/env python3
"""Syntax-only baseline with an off-the-shelf grammar-constrained decoder
(XGrammar via vLLM), VH-SD, Llama-3-8B-Instruct, greedy.

Per task, the grammar admits the output format of our VH-SD gamma automaton
(ctrlg/gamma/topdown/subgoal_decomp.py):

    {"output": ["PRED(name.id, name.id)", ...]}      (>= 1 element)

with one predicate per list element, the task's viable predicate/action names
(SDStateMachine.viable_predicates) at their arity (SD_PREDICATE_ARITY; arity 0
is written "PRED()"), and the scene's (name, id) objects, written with ANY
tokenization. JSON whitespace between structural tokens is free (bounded, as in
the AS XGrammar scripts); the predicate strings themselves are spelled exactly
as the gamma spells them ("PRED(a.1, b.2)").

By default the grammar follows the prompt's own output format, as in the
paper's run (Appendix N): the full JSON object its in-context examples show,
with atoms joined by ' and ' inside a list element. --gamma_format restricts it
to the gamma's format above. Any scene object may fill any argument slot;
--affordances gamma additionally applies the gamma's per-argument affordance
sets, so that with --gamma_format the grammar's language equals the gamma's.

Prompt: the VH-SD system prompt (data/prompts/system_prompts.json) and the
benchmark prompt merged into one user turn; max_tokens 2048.

    python baselines/xgrammar_vh_sd.py     # -> outputs/xgrammar_vh_sd.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from ctrlg.gamma.topdown.subgoal_decomp import build_sd_state_machine
from ctrlg.gamma.topdown.sd_predicate_defs import (
    SD_PREDICATE_ARITY, CROSS_ARG_PREDICATES, compute_sd_arg1_allowed_objects)

M = "models/Meta-Llama-3-8B-Instruct"
PROMPTS = "data/prompts/virtualHome/subgoal_decomposition/helm_prompt.json"
SYSTEM = "data/prompts/system_prompts.json"
WS = 'ws ::= ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n] ([ \\t\\n])?)?)?)?)?)?)?)?'


def q(s):  # GBNF string literal
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def system_prompt():
    d = json.load(open(SYSTEM))
    return "\n".join(d["modules"]["virtualhome"]["SD"]["system_prompt_lines"]).strip()


def _objs(sm, names):
    """GBNF alternation of 'name.id' literals for the given object names."""
    lits = [q(f"{n}.{i}") for n in sorted(names) for i in sm.objects.get(n, [])]
    return " | ".join(lits)


def grammar_for(sm, affordances="none", allow_and=False, task_format=False):
    all_names = set(sm.objects)
    out = '"\\"output\\"" ws ":" ws "[" ws item (ws "," ws item)* ws "]"'
    if task_format:  # the full object the VH-SD prompt asks for (its in-context examples)
        head = ('"\\"necessity_to_use_action\\"" ws ":" ws ("\\"yes\\"" | "\\"no\\"") ws "," ws '
                '"\\"actions_to_include\\"" ws ":" ws "[" ws (act (ws "," ws act)*)? ws "]" ws "," ws ')
        lines = ['root ::= "{" ws ' + head + out + ' ws "}"', 'act ::= "\\"" [A-Z_]+ (" or " [A-Z_]+)* "\\""']
    else:
        lines = ['root ::= "{" ws ' + out + ' ws "}"']
    lines += [WS, "obj ::= " + _objs(sm, all_names)]
    alts, extra = [], []

    def allowed(pred, k):
        a = sm.allowed_objects.get((pred, k))
        return all_names if a is None else set(a) & all_names

    for p in sorted(sm.viable_predicates):
        ar = SD_PREDICATE_ARITY[p]
        head = '"\\"" ' + q(p + "(")
        if ar == 0:
            alts.append(head + ' ")\\""')
            continue
        if affordances == "none":
            body = " obj" + (' ", " obj' if ar == 2 else "")
            alts.append(head + body + ' ")\\""')
            continue
        r0 = f"a_{p.lower()}_0"
        names0 = allowed(p, 0)
        if ar == 1:
            if not names0:
                continue
            extra.append(f"{r0} ::= " + _objs(sm, names0))
            alts.append(f'{head} {r0} ")\\""')
            continue
        if p in CROSS_ARG_PREDICATES:
            sub = []
            for j, n0 in enumerate(sorted(names0)):
                a1 = compute_sd_arg1_allowed_objects(p, n0, sm.objects, sm.properties, sm.categories)
                names1 = allowed(p, 1) if a1 is None else set(a1) & all_names
                if not names1 or not sm.objects.get(n0):
                    continue
                r1 = f"a_{p.lower()}_1_{j}"
                extra.append(f"{r1} ::= " + _objs(sm, names1))
                sub.append("(" + " | ".join(q(f"{n0}.{i}") for i in sm.objects[n0]) + f') ", " {r1}')
            if not sub:
                continue
            extra.append(f"{r0} ::= " + " | ".join("(" + s + ")" for s in sub))
            alts.append(f'{head} {r0} ")\\""')
        else:
            names1 = allowed(p, 1)
            if not names0 or not names1:
                continue
            r1 = f"a_{p.lower()}_1"
            extra.append(f"{r0} ::= " + _objs(sm, names0))
            extra.append(f"{r1} ::= " + _objs(sm, names1))
            alts.append(f'{head} {r0} ", " {r1} ")\\""')
    if allow_and:  # atoms without the surrounding quotes, joined by ' and ' inside one string
        bare = [x.replace('"\\"" ', "", 1).rsplit(' ")\\""', 1)[0] + ' ")"' for x in alts]
        lines.append("atom ::= " + " | ".join("(" + x + ")" for x in bare))
        lines.append('item ::= "\\"" atom (" and " atom)* "\\""')
    else:
        lines.append("item ::= " + " | ".join("(" + x + ")" for x in alts))
    return "\n".join(lines + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="every stride-th task (1 = all)")
    ap.add_argument("--out", default="outputs/xgrammar_vh_sd.json")
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
    P = json.load(open(PROMPTS))[:: args.stride]
    prompts, grammars = [], []
    for r in P:
        sm = build_sd_state_machine(tok, r["llm_prompt"], eos)
        grammars.append(grammar_for(sm, args.affordances, not args.gamma_format, not args.gamma_format))
        prompts.append(tok.apply_chat_template([{"role": "user", "content": sysp + "\n\n" + r["llm_prompt"]}],
                                               tokenize=False, add_generation_prompt=True))
    if args.dump_grammars:
        json.dump([{"identifier": r["identifier"], "grammar": g} for r, g in zip(P, grammars)],
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
    res = [{"identifier": r["identifier"], "llm_output": o.outputs[0].text,
            "finish_reason": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids)}
           for r, o in zip(P, outs)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump([{"identifier": x["identifier"], "llm_output": x["llm_output"]} for x in res], open(args.out, "w"), indent=1)
    json.dump(res, open(args.out.replace(".json", "_meta.json"), "w"), indent=1)
    print("wrote", args.out, len(res))
    for x in res[:3]:
        print(repr(x["llm_output"][:300]))


if __name__ == "__main__":
    main()

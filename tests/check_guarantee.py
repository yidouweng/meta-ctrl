#!/usr/bin/env python3
"""Check the semantic guarantee on a generation output, without the LM.

Every plan of an output (outputs/llama_<module>_<arm>[.json]) is rebuilt as its
generation driver built it -- syntax automaton gamma, tau, candidate tracker,
semantic automaton beta with goal dropping, backward DP, TwoLevelProcessor --
and its tokens are fed through the processor one step at a time (teacher
forcing). Reported per plan and in total:

  invalid    committed actions beta rejects (the guarantee says: none)
  nonaccept  two-level plans whose final beta state does not accept
  forbidden  stored tokens this processor would not allow (plan not producible
             by this code; the first one is printed)
  dead_end_blocks, precondition_only_steps, forced_closes, violations
             the processor's guarantee counters (get_stats)

Exit status 1 if any plan commits a beta-invalid action, else 0.

    python tests/check_guarantee.py --module beh_as --outputs outputs/llama_beh_as_trained
    python tests/check_guarantee.py --module vh_as --outputs outputs/llama_vh_as_mask.json \\
        --hmm models/uniform-as --hard_mask

Speed: every allow/forbid decision of the processor (in-block mask, dead ends,
block closes, EOS gating) depends only on whether p(action, accept within the
budget) > 0, i.e. on accept-reachability, never on the HMM weights. By default
the check therefore uses the DP's boolean reachability table (the
goal-reachability arm's own path) and skips the weighted table; --full_dp
builds the weighted table as the decoder does (slow on CPU for large automata).
The initial belief is uniform (the prompt-conditioned belief has full support,
so it changes no decision). Token ids are not stored: each plan is re-spelled
as a token path through gamma.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the drivers read data/ and models/ relative to the repo

import torch
from transformers import AutoTokenizer

import ctrlg
from ctrlg.beta import backward_dp
from ctrlg.common.constants import NEG_INF
from ctrlg.models import load_semantic_hmm
from ctrlg.runner import eot_id
from ctrlg.two_level.processor import GUARANTEE_STATS, TwoLevelProcessor, _Step

VOCAB_SIZE = 128256  # Llama 3
DEV = "cpu"
TRAINED = {"as": "models/meta-hmm-as",
           "sd": "models/meta-hmm-sd"}
PROMPTS = {"vh_as": "data/prompts/virtualHome/action_sequencing/helm_prompt.json",
           "vh_sd": "data/prompts/virtualHome/subgoal_decomposition/helm_prompt.json",
           "beh_as": "data/prompts/behavior/action_sequencing/action_sequence_prompts.json",
           "beh_sd": "data/prompts/behavior/subgoal_decomposition/subgoal_decomposition_prompts.json"}


# --------------------------------------------------------------- automata
# Each builder mirrors its driver's generate function up to the processor and
# returns (processor, gamma, describe) or (None, gamma, None) when the driver
# decodes the task with the syntax mask alone. The prompt token ids only tell
# the processor where generation starts, so a one-token stand-in is used.
PROMPT_IDS = [128000]


def _dp(beta_sm, hmm, vocab, budget, n_valid, hard):
    from ctrlg.beta.uniform_meta import resolve_uniform_logw
    return backward_dp.BetaBackwardDP(state_machine=beta_sm, semantic_hmm=hmm, semantic_vocab=vocab,
                                      max_budget=budget, device=DEV, hard_mask=hard,
                                      uniform_action_logw=resolve_uniform_logw(hmm, n_valid))


def build_beh_as(text, tok, eos, hmm, vocab, hard, scale):
    import generate_beh_as as G
    from ctrlg.beta.beh_state_machine import parse_beh_states
    from ctrlg.gamma.topdown.behavior_twolevel import build_beh_twolevel_state_machine
    from ctrlg.two_level.beh_candidates import BehaviorTwoLevelCandidateTracker
    from ctrlg.two_level.beh_tau import BehaviorTwoLevelTauParser
    sm = build_beh_twolevel_state_machine(tok, text, eos)
    inst = sorted(sm.all_instances)
    cid = {x: i for i, x in enumerate(inst)}
    valid = G.enumerate_beh_gamma_valid(sm, cid, inst)
    init, target = parse_beh_states(text)
    beta, _, _ = G.build_beta_with_goal_dropping(init, target, inst, cid, valid, G.MAX_BUDGET, G.parse_objects(text))
    if beta.num_accept == 0:
        return None, sm, None
    dp_budget = G.lookahead_budget(beta.num_states, G.MAX_BUDGET)
    tau = BehaviorTwoLevelTauParser(tokenizer=tok, gamma_dfa=sm, instance_to_canonical=cid)
    tracker = BehaviorTwoLevelCandidateTracker(tokenizer=tok, gamma_dfa=sm, tau_parser=tau,
                                               canonical_id_to_instance=inst, gamma_valid_actions=valid)
    prefix_len = len(sm.prefix_seq)
    proc = TwoLevelProcessor(
        ctrlg.DFAModel(sm.build_graph_bfs(VOCAB_SIZE), VOCAB_SIZE), vocab_size=VOCAB_SIZE, eos_token_id=eos,
        prompt_ids=PROMPT_IDS, tokenizer=tok, gamma_dfa=sm, candidate_tracker=tracker, beta_lookahead=beta,
        beta_dp=_dp(beta, hmm, vocab, dp_budget, len(valid), hard), max_meta_budget=G.MAX_BUDGET,
        ctrlg_scale=scale, block_start_fn=lambda gs: gs == ("PREFIX", prefix_len), eos_gate_phase="AFTER_DICT",
        close_tokens={t for t, d in sm._after_dict_alts.items() if d == "close"}, device=DEV)
    return proc, sm, lambda a, ids: f"{a}({','.join(inst[i] for i in ids)})"


def build_beh_sd(text, tok, eos, hmm, vocab, hard, scale):
    import generate_beh_sd as G
    from ctrlg.beta.beh_sd_state_machine import BehSDBetaStateMachine, parse_beh_sd_states
    from ctrlg.gamma.topdown.behavior_sd_twolevel import build_beh_sd_twolevel_state_machine
    from ctrlg.two_level.beh_sd_candidates import BehaviorSDTwoLevelCandidateTracker
    from ctrlg.two_level.beh_sd_tau import BehaviorSDTwoLevelTauParser
    sm = build_beh_sd_twolevel_state_machine(tok, text, eos)
    inst = list(sm.all_instances)
    cid = {x: i for i, x in enumerate(inst)}
    valid = G.enumerate_beh_sd_gamma_valid(sm, cid, inst)
    init, target = parse_beh_sd_states(text)
    beta = BehSDBetaStateMachine(initial_states=init, target_states=target, all_instances=inst, inst_to_cid=cid,
                                 gamma_valid_steps=valid, max_states=G.MAX_STATES, max_budget=G.MAX_BUDGET,
                                 objects=G._parse_sd_objects(text), edge_budget=G.DP_EDGE_BUDGET)
    dp_budget = G.lookahead_budget(beta.num_states, G.MAX_BUDGET)
    tau = BehaviorSDTwoLevelTauParser(tokenizer=tok, gamma_dfa=sm, instance_to_canonical=cid)
    tracker = BehaviorSDTwoLevelCandidateTracker(tokenizer=tok, gamma_dfa=sm, tau_parser=tau,
                                                 canonical_id_to_instance=list(inst), gamma_valid_steps=valid)
    close = {sm.close_list_seq[0]} if getattr(sm, "close_list_seq", None) else set()
    proc = TwoLevelProcessor(
        ctrlg.DFAModel(sm.build_graph_bfs(VOCAB_SIZE), VOCAB_SIZE), vocab_size=VOCAB_SIZE, eos_token_id=eos,
        prompt_ids=PROMPT_IDS, tokenizer=tok, gamma_dfa=sm, candidate_tracker=tracker, beta_lookahead=beta,
        beta_dp=_dp(beta, hmm, vocab, dp_budget, len(valid), hard), max_meta_budget=G.MAX_BUDGET,
        ctrlg_scale=scale, block_start_fn=G._is_atom_start, eos_gate_phase="AFTER_PRED", close_tokens=close,
        device=DEV)
    return proc, sm, lambda a, ids: f"{a}({','.join(inst[i] for i in ids)})"


def build_vh_as(text, tok, eos, hmm, vocab, hard, scale):
    import generate_vh_as as G
    from ctrlg.beta.vh_state_machine import VHBetaStateMachine
    from ctrlg.gamma.topdown.compile import compile_topdown
    from ctrlg.gamma.topdown.virtualhome import build_vh_state_machine
    from ctrlg.two_level.candidates import VirtualHomeCandidateTracker
    from ctrlg.two_level.tau import VirtualHomeTauParser
    sm = build_vh_state_machine(tokenizer=tok, prompt=text, eos_token_id=eos, max_ws_tokens=5)
    raw_to_cid, cid_to_obj, objs = G.build_canonical_mappings(sm)
    valid = G.enumerate_gamma_valid_actions(sm, cid_to_obj)
    try:
        beta = VHBetaStateMachine(prompt=text, gamma_valid_actions=valid, max_states=G.MAX_BETA_STATES)
    except ValueError as e:
        if "max_states" not in str(e):
            raise
        return None, sm, None
    dp = _dp(beta, hmm, vocab, G.MAX_BUDGET, len(valid), hard)
    rho = torch.full((hmm.hidden_size,), -math.log(hmm.hidden_size))
    if dp.compute_log_M(rho, dp.initial_state_id, G.MAX_BUDGET).item() <= NEG_INF / 2:
        return None, sm, None
    tau = VirtualHomeTauParser(tokenizer=tok, gamma_dfa=sm, orig_id_to_canonical=raw_to_cid)
    tracker = VirtualHomeCandidateTracker(tokenizer=tok, gamma_dfa=sm, tau_parser=tau, canonical_id_to_obj=objs)
    proc = TwoLevelProcessor(
        ctrlg.DFAModel(compile_topdown(sm, VOCAB_SIZE), VOCAB_SIZE), vocab_size=VOCAB_SIZE, eos_token_id=eos,
        prompt_ids=PROMPT_IDS, tokenizer=tok, gamma_dfa=sm, candidate_tracker=tracker, beta_lookahead=beta,
        beta_dp=dp, max_meta_budget=G.MAX_BUDGET, ctrlg_scale=scale, device=DEV)
    return proc, sm, lambda a, ids: f"{a}({','.join('%s.%s' % tuple(objs[i]) for i in ids)})"


def build_vh_sd(text, tok, eos, hmm, vocab, hard, scale):
    import generate_vh_sd as G
    from ctrlg.beta.sd_state_machine import SDBetaStateMachine
    from ctrlg.gamma.topdown.compile import compile_topdown
    from ctrlg.gamma.topdown.subgoal_decomp import SDStateMachine, enumerate_gamma_valid_predicates, parse_sd_prompt
    from ctrlg.two_level.candidates import SubgoalDecompCandidateTracker
    from ctrlg.two_level.tau import SubgoalDecompTauParser
    objects, props, cats, init, goals, action_goals = parse_sd_prompt(text)
    sm = SDStateMachine(tok, objects, props, cats, eos)
    raw_to_cid, cid_to_obj, ordered = G.build_id_mapping(objects)
    valid = enumerate_gamma_valid_predicates(sm, raw_to_cid)
    beta = SDBetaStateMachine(initial_states=init, goal_states=goals, action_goals=action_goals, objects=objects,
                              properties=props, categories=cats, gamma_valid_predicates=valid,
                              orig_id_to_canonical=raw_to_cid, canonical_to_info=dict(enumerate(ordered)))
    if beta.num_accept == 0:
        return None, sm, None
    tau = SubgoalDecompTauParser(tokenizer=tok, gamma_dfa=sm, orig_id_to_canonical=raw_to_cid)
    tracker = SubgoalDecompCandidateTracker(tokenizer=tok, gamma_dfa=sm, tau_parser=tau,
                                            canonical_id_to_obj=cid_to_obj, gamma_valid_predicates=valid)
    prefix_len = len(sm.prefix_seq)
    proc = TwoLevelProcessor(
        ctrlg.DFAModel(compile_topdown(sm, VOCAB_SIZE, use_bfs=True), VOCAB_SIZE), vocab_size=VOCAB_SIZE,
        eos_token_id=eos, prompt_ids=PROMPT_IDS, tokenizer=tok, gamma_dfa=sm, candidate_tracker=tracker,
        beta_lookahead=beta, beta_dp=_dp(beta, hmm, vocab, G.MAX_BUDGET, len(valid), hard),
        max_meta_budget=G.MAX_BUDGET, ctrlg_scale=scale,
        block_start_fn=lambda gs: isinstance(gs, tuple) and gs[0] == "PREFIX" and gs[1] == prefix_len,
        eos_gate_phase="AFTER_STEP", close_tokens=set(sm.close_seq), device=DEV)
    return proc, sm, lambda a, ids: f"{a}({','.join('%s.%s' % tuple(ordered[i]) for i in ids)})"


BUILDERS = {"beh_as": build_beh_as, "beh_sd": build_beh_sd, "vh_as": build_vh_as, "vh_sd": build_vh_sd}


# ----------------------------------------------------------- tokenization
def gamma_path(tok, strs, sm, text):
    """A token path through gamma that spells `text` (HF tokenization first,
    then longest match, with backtracking); None if there is none."""
    hf_at, pos = {}, 0
    for t in tok(text, add_special_tokens=False)["input_ids"]:
        hf_at.setdefault(pos, t)
        pos += len(strs[t])
    path, failed, n = [], set(), len(text)
    sys.setrecursionlimit(max(10000, 4 * n))

    def walk(state, p):
        if p == n:
            return True
        if (state, p) in failed:
            return False
        allowed = sm.allowed_tokens(state)
        first = [hf_at[p]] if hf_at.get(p) in allowed else []
        rest = sorted((t for t in allowed if t < len(strs) and strs[t] and t not in first
                       and text.startswith(strs[t], p)), key=lambda t: -len(strs[t]))
        for t in first + rest:
            if not text.startswith(strs[t], p):
                continue
            nxt = sm.next_state(state, t)
            if nxt is None or nxt == getattr(sm, "DEAD", ("DEAD",)):
                continue
            path.append(t)
            if walk(nxt, p + len(strs[t])):
                return True
            path.pop()
        failed.add((state, p))
        return False

    return list(path) if walk(sm.initial_state(), 0) else None


# ------------------------------------------------------------------ check
def check_plan(proc, sm, describe, tok, strs, eos, text):
    path = gamma_path(tok, strs, sm, text)
    if path is None:
        return {"error": "no gamma token path spells the plan"}
    invalid = []
    handle = proc._handle_completed_action

    def on_commit(step):  # beta-validity of every committed action, from the processor's state
        if proc._beta_lookahead.apply(proc._beta_state_id, _Step(step.action, step.ids)) is None:
            invalid.append({"index": len(proc._completed_actions), "action": describe(step.action, step.ids)})
        handle(step)
    proc._handle_completed_action = on_commit

    forbidden, ids, zeros = [], list(PROMPT_IDS), torch.zeros(1, VOCAB_SIZE)
    for i, t in enumerate(path + [eos]):
        logits = proc(torch.tensor([ids]), zeros.clone())
        if not bool(logits[0, t] > -1e20):
            allowed = (logits[0] > -1e20).nonzero(as_tuple=True)[0].tolist()
            forbidden.append({"token_index": i, "token": strs[t] if t != eos else "<eot>",
                              "action_index": len(proc._completed_actions),
                              "allowed": [strs[a] for a in allowed[:8]]})
        ids.append(t)
    st = proc.get_stats()
    return {"n_tokens": len(path), "actions": st["completed_actions"], "invalid": invalid,
            "accept": bool(proc._beta_lookahead.is_accept(st["beta_state"])), "forbidden": forbidden,
            **{k: st[k] for k in GUARANTEE_STATS}}


def load_outputs(path):
    """identifier -> plan text, from a VirtualHome JSON list or a BEHAVIOR prompt_<i>.json directory."""
    if os.path.isdir(path):
        recs = [r for f in glob.glob(os.path.join(path, "prompt_*.json")) for r in json.load(open(f))]
    else:
        recs = json.load(open(path))
    return {r["identifier"]: r["llm_output"] for r in recs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--outputs", required=True, help="outputs/llama_<module>_<arm>[.json]")
    ap.add_argument("--hmm", default=None, help="meta-token HMM the output was decoded with (default: trained)")
    ap.add_argument("--hard_mask", action="store_true", help="the output is from the goal-reachability arm")
    ap.add_argument("--ctrlg_scale", type=float, default=1.0, help="does not change any allow/forbid decision")
    ap.add_argument("--base_model", default="models/Meta-Llama-3-8B-Instruct", help="tokenizer only")
    ap.add_argument("--full_dp", action="store_true", help="build the weighted DP table (slow on CPU)")
    ap.add_argument("--indices", default="", help="comma-separated prompt indices (default: all)")
    ap.add_argument("--report", default=None, help="write per-plan results as JSON lines here")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if not args.full_dp:  # boolean reachability only (see module docstring)
        backward_dp.BetaBackwardDP._build_dp_table = lambda self: torch.zeros(1, 1, 1)

    family = args.module.split("_")[1]
    hmm_dir = args.hmm or TRAINED[family]
    hmm = load_semantic_hmm(hmm_dir, device=DEV)
    if family == "as":
        vocab = backward_dp.SemanticVocab.from_json("data/vocab/hmm_vocab_as_actions_only_full_schema.json")
    else:
        from ctrlg.beta.sd_vocab import load_sd_semantic_vocab
        vocab = load_sd_semantic_vocab(hmm_dir, "virtualhome" if args.module == "vh_sd" else "behavior")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    strs = [tok.decode([i]) for i in range(len(tok))]
    eos = eot_id(tok)

    plans = load_outputs(args.outputs)
    prompts = json.load(open(PROMPTS[args.module]))
    todo = [(i, p) for i, p in enumerate(prompts) if p["identifier"] in plans]
    if args.indices:
        keep = {int(x) for x in args.indices.split(",")}
        todo = [(i, p) for i, p in todo if i in keep]
    hard = args.hard_mask or not args.full_dp
    report = open(args.report, "w") if args.report else None
    tot = {"plans": 0, "syntax_mask": 0, "errors": 0, "plans_invalid": 0, "invalid": 0, "nonaccept": 0,
           "plans_forbidden": 0, **{k: 0 for k in GUARANTEE_STATS}}
    for i, p in todo:
        tot["plans"] += 1
        try:
            proc, sm, describe = BUILDERS[args.module](p["llm_prompt"], tok, eos, hmm, vocab, hard, args.ctrlg_scale)
            r = ({"mode": "syntax_mask"} if proc is None
                 else dict(mode="two_level", **check_plan(proc, sm, describe, tok, strs, eos, plans[p["identifier"]])))
        except Exception as e:  # report and go on
            r = {"error": repr(e)}
        r = {"prompt_idx": i, "identifier": p["identifier"], **r}
        if r.get("error"):
            tot["errors"] += 1
        elif r["mode"] == "syntax_mask":
            tot["syntax_mask"] += 1
        else:
            tot["plans_invalid"] += bool(r["invalid"])
            tot["invalid"] += len(r["invalid"])
            tot["nonaccept"] += not r["accept"]
            tot["plans_forbidden"] += bool(r["forbidden"])
            for k in GUARANTEE_STATS:
                tot[k] += r[k]
        line = f"{i:4d} {p['identifier'][:48]:48s} {r.get('mode', 'error'):11s}"
        if r.get("error"):
            line += f" ERROR {r['error']}"
        elif r["mode"] == "two_level":
            line += (f" actions={r['actions']:2d} invalid={len(r['invalid'])} accept={int(r['accept'])} "
                     + " ".join(f"{k}={r[k]}" for k in GUARANTEE_STATS))
            for x in r["invalid"]:
                line += f"\n       invalid action {x['index']}: {x['action']}"
            if r["forbidden"]:
                f0 = r["forbidden"][0]
                line += (f"\n       first forbidden token #{f0['token_index']} {f0['token']!r} in action "
                         f"{f0['action_index']} (allowed: {f0['allowed']}); {len(r['forbidden'])} in total")
        print(line, flush=True)
        if report:
            report.write(json.dumps(r) + "\n")
            report.flush()
    print("TOTAL " + " ".join(f"{k}={v}" for k, v in tot.items()))
    sys.exit(1 if tot["invalid"] else 0)


if __name__ == "__main__":
    main()

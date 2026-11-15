#!/usr/bin/env python3
"""Meta-Ctrl decoding for BEHAVIOR subgoal decomposition.

A plan is a JSON list of predicate atoms, one per element. gamma fixes the
format and the admissible atoms per predicate role; beta tracks goal atoms and
their preconditions, dropping goals until (states x candidate atoms) fits the
DP edge budget, so every task has an accepting state. Meta-Ctrl uses the
calibrated lookahead c = 2 here (--ctrlg_scale; paper, Appendix I).

    python generate_beh_sd.py --semantic_hmm_path models/<meta-hmm> --ctrlg_scale 2 --output_dir outputs/beh_sd
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctrlg
from ctrlg.beta.backward_dp import BetaBackwardDP
from ctrlg.beta.beh_sd_state_machine import BehSDBetaStateMachine, parse_beh_sd_states
from ctrlg.beta.sd_vocab import load_sd_semantic_vocab
from ctrlg.beta.uniform_meta import resolve_uniform_logw
from ctrlg.gamma.topdown.behavior_sd_twolevel import SD_PREDICATES, build_beh_sd_twolevel_state_machine
from ctrlg.prompts import prompt_ids as build_prompt_ids
from ctrlg.models import compute_prompt_end_embedding, load_semantic_hmm
from ctrlg.runner import FALLBACK_NO_PLAN, add_common_args, eot_id, load_lm, run_prompts
from ctrlg.two_level.beh_sd_candidates import BehaviorSDTwoLevelCandidateTracker
from ctrlg.two_level.beh_sd_tau import BehaviorSDTwoLevelTauParser
from ctrlg.two_level.processor import GUARANTEE_STATS, TwoLevelProcessor
from ctrlg.two_level.token_mask import TokenMask

PROMPTS = "data/prompts/behavior/subgoal_decomposition/subgoal_decomposition_prompts.json"
META_HMM = "models/meta-hmm-sd"
MAX_BUDGET = 35
MAX_NEW_TOKENS = 2048
# The DP build time grows with (automaton states x candidate atoms); goals are
# dropped until it is at most this. The state cap bounds the DP table's memory.
DP_EDGE_BUDGET = 40_000_000
MAX_STATES = 1_200_000
LARGE_BETA_STATES = 30_000  # above this many states the lookahead covers LARGE_BETA_LOOKAHEAD atoms
LARGE_BETA_LOOKAHEAD = 20

_OBJ_RE = re.compile(r"\{'name': '([^']+)',\s*'category':\s*'([^']+)'\}")


def build_prompt(tokenizer, user_prompt: str) -> List[int]:
    return build_prompt_ids(tokenizer, "behavior", "SD", user_prompt)


def _parse_sd_objects(prompt_text: str) -> List[Dict]:
    """{name, category} pairs from the "## Relevant objects" block."""
    start = prompt_text.find("## Relevant objects")
    end = prompt_text.find("\n## ", start + 1)
    text = prompt_text if start < 0 else prompt_text[start:end if end > 0 else len(prompt_text)]
    return [{"name": n, "category": c} for n, c in _OBJ_RE.findall(text)]


def enumerate_beh_sd_gamma_valid(sm, inst_to_cid: Dict[str, int],
                                 inst_list: List[str]) -> List[Tuple[str, bool, Tuple[int, ...]]]:
    """Every (predicate, negated, canonical ids) gamma can emit; each positive
    atom also appears negated."""
    def admissible(inst, viable):
        seq = sm.instance_token_seqs.get(inst)
        if not seq:
            return False
        if viable is None:
            return True
        node = 0
        for t in seq:
            node = sm.instance_trie.next_node(node, t)
            if node is None or node not in viable:
                return False
        return True

    out = []
    for pred in sorted(SD_PREDICATES):
        roles = [sm._viable_instance_nodes(pred, r, kind=sm.TK_CANON) for r in range(SD_PREDICATES[pred])]
        if len(roles) == 1:
            for inst in inst_list:
                if admissible(inst, roles[0]):
                    out += [(pred, False, (inst_to_cid[inst],)), (pred, True, (inst_to_cid[inst],))]
        elif len(roles) == 2:
            for a in inst_list:
                if not admissible(a, roles[0]):
                    continue
                for b in inst_list:
                    if admissible(b, roles[1]):
                        ids = (inst_to_cid[a], inst_to_cid[b])
                        out += [(pred, False, ids), (pred, True, ids)]
    return out


def _is_atom_start(gs) -> bool:
    base = gs[0] if isinstance(gs, tuple) else gs
    if isinstance(base, tuple):
        base = base[0]
    return base == "PRED_START"


def lookahead_budget(num_states: int, max_budget: int) -> int:
    """Atoms the lookahead covers; the plan itself may still use max_budget."""
    return min(max_budget, LARGE_BETA_LOOKAHEAD) if num_states > LARGE_BETA_STATES else max_budget


def generate_one(prompt: Dict, tokenizer, model, *, vocab_size, eos_id, semantic_hmm, semantic_vocab, max_budget,
                 ctrlg_scale, hard_mask, syntax_only, device, max_new_tokens=MAX_NEW_TOKENS, debug=False) -> Dict:
    text = prompt["llm_prompt"]
    prompt_ids = build_prompt(tokenizer, text)
    sm = build_beh_sd_twolevel_state_machine(tokenizer, text, eos_id)
    dfa_model = ctrlg.DFAModel(sm.build_graph_bfs(vocab_size), vocab_size).to(device)
    if syntax_only:
        mask = TokenMask(dfa_model, prompt_ids, vocab_size, eos_id, device)
        with torch.no_grad():
            out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or eos_id,
                                 logits_processor=LogitsProcessorList([mask]))
        return {"llm_output": tokenizer.decode(out[0, len(prompt_ids):].tolist(), skip_special_tokens=True),
                "stats": {"mode": "syntax_mask"}}

    inst_list = list(sm.all_instances)
    inst_to_cid = {inst: i for i, inst in enumerate(inst_list)}
    gamma_valid = enumerate_beh_sd_gamma_valid(sm, inst_to_cid, inst_list)
    init_states, target_states = parse_beh_sd_states(text)
    beta_sm = BehSDBetaStateMachine(initial_states=init_states, target_states=target_states,
                                    all_instances=inst_list, inst_to_cid=inst_to_cid, gamma_valid_steps=gamma_valid,
                                    max_states=MAX_STATES, max_budget=max_budget, objects=_parse_sd_objects(text),
                                    edge_budget=DP_EDGE_BUDGET, debug=debug)
    assert beta_sm.num_accept > 0, "goal dropping leaves an accepting state by construction"
    dp_budget = lookahead_budget(beta_sm.num_states, max_budget)
    beta_dp = BetaBackwardDP(state_machine=beta_sm, semantic_hmm=semantic_hmm, semantic_vocab=semantic_vocab,
                             max_budget=dp_budget, device=device, hard_mask=hard_mask,
                             uniform_action_logw=resolve_uniform_logw(semantic_hmm, len(gamma_valid)))
    tau = BehaviorSDTwoLevelTauParser(tokenizer=tokenizer, gamma_dfa=sm, instance_to_canonical=inst_to_cid)
    tracker = BehaviorSDTwoLevelCandidateTracker(tokenizer=tokenizer, gamma_dfa=sm, tau_parser=tau,
                                                 canonical_id_to_instance=list(inst_list),
                                                 gamma_valid_steps=gamma_valid)
    beta_init = None
    if semantic_hmm.nn_proj is not None:
        emb = compute_prompt_end_embedding(model, prompt_ids, device=device)
        if int(emb.shape[-1]) == int(semantic_hmm.nn_proj.shape[1]):
            beta_init = semantic_hmm.initial_belief(emb)
    # At AFTER_PRED the plan either continues with `", "` or closes with `"]}`;
    # only the closer is gated.
    close = {sm.close_list_seq[0]} if getattr(sm, "close_list_seq", None) else set()
    processor = TwoLevelProcessor(
        dfa_model, vocab_size=vocab_size, eos_token_id=eos_id, prompt_ids=prompt_ids, tokenizer=tokenizer,
        gamma_dfa=sm, candidate_tracker=tracker, beta_lookahead=beta_sm, beta_dp=beta_dp,
        beta_initial_belief=beta_init, max_meta_budget=max_budget, ctrlg_scale=ctrlg_scale,
        block_start_fn=_is_atom_start, eos_gate_phase="AFTER_PRED", close_tokens=close, device=device, debug=debug)
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids], device=device),
                             max_new_tokens=max_new_tokens, do_sample=False, logits_processor=[processor])
    ps = processor.get_stats()
    stats = {"mode": "two_level", "goals_dropped": len(beta_sm.dropped_goal_indices), "n_kept": beta_sm.n_goals,
             "beta_states": beta_sm.num_states, "n_atoms": ps["completed_actions"],
             "is_accept": beta_sm.is_accept(ps["beta_state"]), "meta_budget_remaining": ps["meta_budget"],
             "dp_budget": dp_budget, **{k: ps[k] for k in GUARANTEE_STATS}}
    return {"llm_output": tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True), "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, prompts=PROMPTS, meta_hmm=META_HMM, max_budget=MAX_BUDGET, max_new_tokens=MAX_NEW_TOKENS)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    tokenizer, model, vocab_size, device = load_lm(args.base_model, args.cuda_device)
    eos_id = eot_id(tokenizer)
    semantic_hmm = load_semantic_hmm(args.semantic_hmm_path, device=device)
    semantic_vocab = load_sd_semantic_vocab(args.semantic_hmm_path, "behavior")

    def gen(prompt):
        return generate_one(prompt, tokenizer, model, vocab_size=vocab_size, eos_id=eos_id,
                            semantic_hmm=semantic_hmm, semantic_vocab=semantic_vocab, max_budget=args.max_budget,
                            ctrlg_scale=args.ctrlg_scale, hard_mask=args.beta_hard_mask,
                            syntax_only=args.syntax_only, device=device, max_new_tokens=args.max_new_tokens,
                            debug=args.debug)

    run_prompts(args, gen, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

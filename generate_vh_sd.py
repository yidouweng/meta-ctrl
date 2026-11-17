#!/usr/bin/env python3
"""Meta-Ctrl decoding for VirtualHome subgoal decomposition.

A plan is a JSON list of predicate atoms. gamma fixes the format and the
admissible (predicate, object) atoms; beta tracks the goal predicates and
their preconditions. If beta has no accepting state after goal dropping, the
task is decoded with the syntax mask alone.

    python generate_vh_sd.py --semantic_hmm_path models/<meta-hmm> --output outputs/vh_sd.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctrlg
from ctrlg import lm_family as LF
from ctrlg.beta.backward_dp import BetaBackwardDP, SemanticVocab
from ctrlg.beta.sd_state_machine import SDBetaStateMachine
from ctrlg.beta.sd_vocab import load_sd_semantic_vocab
from ctrlg.beta.uniform_meta import resolve_uniform_logw
from ctrlg.gamma.topdown.compile import compile_topdown
from ctrlg.gamma.topdown.subgoal_decomp import SDStateMachine, enumerate_gamma_valid_predicates, parse_sd_prompt
from ctrlg.models import SemanticHMM, compute_prompt_end_embedding, load_semantic_hmm
from ctrlg.runner import FALLBACK_NO_PLAN, add_common_args, eot_id, load_lm, run_prompts
from ctrlg.two_level.candidates import SubgoalDecompCandidateTracker
from ctrlg.two_level.processor import TwoLevelProcessor
from ctrlg.two_level.tau import SubgoalDecompTauParser

PROMPTS = "data/prompts/virtualHome/subgoal_decomposition/helm_prompt.json"
META_HMM = "models/meta-hmm-sd"
MAX_BUDGET = 20
MAX_NEW_TOKENS = 2048


def build_chat_prompt(tokenizer, user_prompt: str) -> List[int]:
    return LF.prompt_ids(tokenizer, "virtualhome", "SD", user_prompt)


def build_id_mapping(objects):
    """Canonical object ids in (name, numeric id) order."""
    ordered = [(name, raw) for name in sorted(objects) for raw in sorted(objects[name], key=int)]
    return {raw: cid for cid, (_, raw) in enumerate(ordered)}, list(ordered), ordered


class SDStateMask(LogitsProcessor):
    """Syntax-only decoding: mask everything the SD state machine does not allow next."""

    def __init__(self, sm, prompt_len: int):
        self.sm, self.state, self.prompt_len, self.n_seen = sm, sm.initial_state(), prompt_len, 0

    def __call__(self, input_ids, scores):
        n = input_ids.shape[1] - self.prompt_len
        if n > 0 and n > self.n_seen:
            self.state = self.sm.step(self.state, int(input_ids[0, -1].item()))
            self.n_seen = n
        mask = torch.full_like(scores[0], float("-inf"))
        for tok in self.sm.allowed_tokens(self.state):
            if 0 <= tok < mask.shape[0]:
                mask[tok] = 0.0
        return scores + mask.unsqueeze(0)


def generate_syntax_only(model, tokenizer, prompt_text: str, max_new_tokens: int, device: str) -> str:
    objects, properties, categories, _, _, _ = parse_sd_prompt(prompt_text)
    sm = SDStateMachine(tokenizer, objects, properties, categories, eot_id(tokenizer))
    prompt_ids = LF.prefill_final(tokenizer, build_chat_prompt(tokenizer, prompt_text))
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=max_new_tokens,
                             do_sample=False, logits_processor=[SDStateMask(sm, len(prompt_ids))],
                             **LF.generate_kwargs(tokenizer))
    return LF.answer_text(tokenizer, out[0, len(prompt_ids):])


def generate_two_level(model, tokenizer, prompt_text: str, *, vocab_size: int,
                       semantic_hmm: Optional[SemanticHMM], semantic_vocab: Optional[SemanticVocab],
                       max_budget: int, max_new_tokens: int, ctrlg_scale: float, hard_mask: bool,
                       device: str, debug: bool) -> Optional[Dict[str, Any]]:
    """Two-level decoding; llm_output is None if beta has no accepting state."""
    eos_id = eot_id(tokenizer)
    objects, properties, categories, initial_states, goal_states, action_goals = parse_sd_prompt(prompt_text)
    sm = SDStateMachine(tokenizer, objects, properties, categories, eos_id)
    raw_to_cid, cid_to_obj, ordered = build_id_mapping(objects)
    gamma_valid = enumerate_gamma_valid_predicates(sm, raw_to_cid)
    dfa_model = ctrlg.DFAModel(compile_topdown(sm, vocab_size, use_bfs=True), vocab_size).to(device)
    tau = SubgoalDecompTauParser(tokenizer=tokenizer, gamma_dfa=sm, orig_id_to_canonical=raw_to_cid)
    tracker = SubgoalDecompCandidateTracker(tokenizer=tokenizer, gamma_dfa=sm, tau_parser=tau,
                                            canonical_id_to_obj=cid_to_obj, gamma_valid_predicates=gamma_valid)
    beta_sm = SDBetaStateMachine(initial_states=initial_states, goal_states=goal_states, action_goals=action_goals,
                                 objects=objects, properties=properties, categories=categories,
                                 gamma_valid_predicates=gamma_valid, orig_id_to_canonical=raw_to_cid,
                                 canonical_to_info=dict(enumerate(ordered)), debug=debug)
    n_dropped = len(getattr(beta_sm, "dropped_goal_indices", []) or [])
    if beta_sm.num_accept == 0:
        return {"llm_output": None,
                "stats": {"mode": "syntax_mask", "fallback": FALLBACK_NO_PLAN, "goals_dropped": n_dropped}}
    beta_dp = None
    if semantic_hmm is not None:
        beta_dp = BetaBackwardDP(state_machine=beta_sm, semantic_hmm=semantic_hmm, semantic_vocab=semantic_vocab,
                                 max_budget=max_budget, device=device, hard_mask=hard_mask,
                                 uniform_action_logw=resolve_uniform_logw(semantic_hmm, len(gamma_valid)))
    prompt_ids = build_chat_prompt(tokenizer, prompt_text)
    beta_init = None
    if semantic_hmm is not None and semantic_hmm.nn_proj is not None:
        emb = compute_prompt_end_embedding(model, prompt_ids, device=device)
        if int(emb.shape[-1]) == int(semantic_hmm.nn_proj.shape[1]):
            beta_init = semantic_hmm.initial_belief(emb)
    prefix_len = len(sm.prefix_seq)
    processor = TwoLevelProcessor(
        dfa_model, vocab_size=vocab_size, eos_token_id=eos_id, prompt_ids=prompt_ids, tokenizer=tokenizer,
        gamma_dfa=sm, candidate_tracker=tracker, beta_lookahead=beta_sm, beta_dp=beta_dp,
        beta_initial_belief=beta_init, max_meta_budget=max_budget, ctrlg_scale=ctrlg_scale,
        block_start_fn=lambda gs: isinstance(gs, tuple) and gs[0] == "PREFIX" and gs[1] == prefix_len,
        eos_gate_phase="AFTER_STEP", close_tokens=set(sm.close_seq), device=device, debug=debug,
        **LF.processor_kwargs(tokenizer))
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids], device=device), do_sample=False,
                             max_new_tokens=LF.generation_budget(tokenizer, max_new_tokens),
                             logits_processor=[processor], **LF.generate_kwargs(tokenizer))
    stats = dict(processor.get_stats(), mode="two_level", beta_states=int(beta_sm.num_states),
                 goals_dropped=n_dropped, dp_budget=max_budget)
    return {"llm_output": LF.answer_text(tokenizer, out[0, len(prompt_ids):]), "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, prompts=PROMPTS, meta_hmm=META_HMM, max_budget=MAX_BUDGET, max_new_tokens=MAX_NEW_TOKENS)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tokenizer, model, vocab_size, device = load_lm(args.base_model, args.cuda_device)
    semantic_hmm = semantic_vocab = None
    if not args.syntax_only:
        semantic_hmm = load_semantic_hmm(args.semantic_hmm_path, device=device)
        semantic_vocab = load_sd_semantic_vocab(args.semantic_hmm_path, "virtualhome")

    def gen(prompt):
        rec = {"identifier": prompt.get("identifier", "unknown"), "llm_output": "", "stats": {}, "error": None}
        text = prompt.get("llm_prompt", "")
        try:
            res = None
            if not args.syntax_only:
                res = generate_two_level(model, tokenizer, text, vocab_size=vocab_size, semantic_hmm=semantic_hmm,
                                         semantic_vocab=semantic_vocab, max_budget=args.max_budget,
                                         max_new_tokens=args.max_new_tokens, ctrlg_scale=args.ctrlg_scale,
                                         hard_mask=args.beta_hard_mask, device=device, debug=args.debug)
            if res is None or res["llm_output"] is None:
                stats = res["stats"] if res is not None else {"mode": "syntax_mask"}
                res = {"llm_output": generate_syntax_only(model, tokenizer, text, args.max_new_tokens, device),
                       "stats": stats}
            rec.update(res)
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            rec["error"] = str(e)
        return rec

    run_prompts(args, gen, output_json=args.output, include_failures=True)


if __name__ == "__main__":
    main()

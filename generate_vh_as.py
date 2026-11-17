#!/usr/bin/env python3
"""Meta-Ctrl decoding for VirtualHome action sequencing.

Per prompt: build the syntax automaton over tokens (gamma) from the prompt,
enumerate the grounded actions it can emit, build the semantic automaton (beta)
over those actions, the backward DP of the meta-token HMM times beta, and
decode greedily with TwoLevelProcessor. If beta cannot be built or no plan can
reach the goal, the task is decoded with the syntax mask alone.

    python generate_vh_as.py --semantic_hmm_path models/<meta-hmm> --output outputs/vh_as.json
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctrlg
from ctrlg import lm_family as LF
from ctrlg.beta.backward_dp import BetaBackwardDP, SemanticVocab
from ctrlg.beta.uniform_meta import resolve_uniform_logw
from ctrlg.beta.vh_state_machine import MAX_STATES as VH_MAX_STATES, VHBetaStateMachine, VHStep
from ctrlg.common.constants import NEG_INF
from ctrlg.gamma.topdown.compile import compile_topdown
from ctrlg.gamma.topdown.virtualhome import VHStateMachine, build_vh_state_machine
from ctrlg.models import SemanticHMM, compute_prompt_end_embedding, load_semantic_hmm
from ctrlg.runner import FALLBACK_NO_PLAN, FALLBACK_SIZE, add_common_args, eot_id, load_lm, run_prompts
from ctrlg.two_level.candidates import VirtualHomeCandidateTracker
from ctrlg.two_level.processor import TwoLevelProcessor
from ctrlg.two_level.tau import VirtualHomeTauParser
from ctrlg.two_level.token_mask import TokenMask

PROMPTS = "data/prompts/virtualHome/action_sequencing/helm_prompt.json"
META_HMM = "models/meta-hmm-as"
META_VOCAB = "data/vocab/hmm_vocab_as_actions_only_full_schema.json"
MAX_BUDGET = 20
MAX_NEW_TOKENS = 2048
MAX_BETA_STATES = VH_MAX_STATES  # semantic automaton size limit (states)


def _is_allowed(gamma_sm: VHStateMachine, action: str, role: int, name: str) -> bool:
    allowed = gamma_sm.allowed_objects_per_role.get((action, role), set())
    return name in allowed or len(allowed) == 0


def enumerate_gamma_valid_actions(gamma_sm: VHStateMachine,
                                  canonical_id_to_obj: Dict[int, Tuple[str, str]]) -> List[VHStep]:
    """Every grounded action gamma can emit, with canonical object ids."""
    out = []
    for action in gamma_sm.viable_actions:
        arity = gamma_sm.action_arity.get(action, 0)
        if arity == 0:
            out.append(VHStep(action, ()))
        elif arity == 1:
            out += [VHStep(action, (cid,)) for cid, (name, _) in canonical_id_to_obj.items()
                    if _is_allowed(gamma_sm, action, 0, name)]
        elif arity == 2:
            for cid1, (name1, _) in canonical_id_to_obj.items():
                for cid2, (name2, _) in canonical_id_to_obj.items():
                    if _is_allowed(gamma_sm, action, 0, name1) and _is_allowed(gamma_sm, action, 1, name2):
                        out.append(VHStep(action, (cid1, cid2)))
    return out


def build_canonical_mappings(gamma_sm: VHStateMachine):
    """Canonical object ids, assigned in (name, raw id) order.

    Returns {raw_id: cid}, {cid: (name, raw_id)} and the (name, raw_id) list."""
    objects = gamma_sm.scene_inventory
    ordered = [(name, raw_id) for name in sorted(objects)
               for raw_id in sorted(objects[name], key=lambda x: int(x) if str(x).isdigit() else x)]
    return ({raw: cid for cid, (_, raw) in enumerate(ordered)},
            {cid: obj for cid, obj in enumerate(ordered)}, list(ordered))


def generate_one(model, tokenizer, prompt: Dict, *, vocab_size: int, mask_eos_id: int,
                 semantic_hmm: Optional[SemanticHMM], semantic_vocab: Optional[SemanticVocab],
                 max_budget: int, max_new_tokens: int, ctrlg_scale: float, hard_mask: bool,
                 syntax_only: bool, device: str, debug: bool) -> Dict[str, Any]:
    ident, prompt_text = prompt.get("identifier", "unknown"), prompt.get("llm_prompt", "")
    result = {"identifier": ident, "llm_output": "", "stats": {}, "error": None}
    try:
        gamma_sm = build_vh_state_machine(tokenizer=tokenizer, prompt=prompt_text,
                                          eos_token_id=mask_eos_id, max_ws_tokens=5)
        raw_to_cid, cid_to_obj, obj_list = build_canonical_mappings(gamma_sm)
        tau = VirtualHomeTauParser(tokenizer=tokenizer, gamma_dfa=gamma_sm, orig_id_to_canonical=raw_to_cid)
        gamma_valid = enumerate_gamma_valid_actions(gamma_sm, cid_to_obj)

        beta_sm, beta_error = None, None
        try:
            beta_sm = VHBetaStateMachine(prompt=prompt_text, gamma_valid_actions=gamma_valid,
                                         max_states=MAX_BETA_STATES, debug=debug)
        except ValueError as e:
            if "max_states" not in str(e):
                raise
            beta_error = str(e)

        beta_dp = None
        if beta_sm is not None and semantic_hmm is not None:
            beta_dp = BetaBackwardDP(state_machine=beta_sm, semantic_hmm=semantic_hmm,
                                     semantic_vocab=semantic_vocab, max_budget=max_budget, device=device,
                                     hard_mask=hard_mask,
                                     uniform_action_logw=resolve_uniform_logw(semantic_hmm, len(gamma_valid)))
        tracker = VirtualHomeCandidateTracker(tokenizer=tokenizer, gamma_dfa=gamma_sm, tau_parser=tau,
                                              canonical_id_to_obj=obj_list)
        dfa_model = ctrlg.DFAModel(compile_topdown(gamma_sm, vocab_size, debug=debug), vocab_size).to(device)

        prompt_ids = LF.prompt_ids(tokenizer, "virtualhome", "AS", prompt_text)

        beta_init = None
        if semantic_hmm is not None and semantic_hmm.nn_proj is not None:
            emb = compute_prompt_end_embedding(model, prompt_ids, device=device)
            if int(emb.shape[-1]) == int(semantic_hmm.nn_proj.shape[1]):
                beta_init = semantic_hmm.initial_belief(emb)

        mask_only = syntax_only or beta_sm is None
        if beta_dp is not None and not mask_only:
            rho = beta_init if beta_init is not None else torch.full(
                (semantic_hmm.hidden_size,), -math.log(semantic_hmm.hidden_size), device=device, dtype=torch.float32)
            mask_only = beta_dp.compute_log_M(rho, beta_dp.initial_state_id, max_budget).item() <= NEG_INF / 2

        if mask_only:
            prompt_ids = LF.prefill_final(tokenizer, prompt_ids)
            processor = TokenMask(dfa_model, prompt_ids, vocab_size, mask_eos_id, device)
        else:
            processor = TwoLevelProcessor(
                dfa_model, vocab_size=vocab_size, eos_token_id=mask_eos_id, prompt_ids=prompt_ids,
                tokenizer=tokenizer, gamma_dfa=gamma_sm, candidate_tracker=tracker, beta_lookahead=beta_sm,
                beta_dp=beta_dp, beta_initial_belief=beta_init, max_meta_budget=max_budget,
                ctrlg_scale=ctrlg_scale, device=device, debug=debug, **LF.processor_kwargs(tokenizer))

        ids = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                max_new_tokens=max_new_tokens if mask_only else LF.generation_budget(tokenizer, max_new_tokens),
                logits_processor=LogitsProcessorList([processor]), pad_token_id=tokenizer.pad_token_id,
                **LF.generate_kwargs(tokenizer))
        result["llm_output"] = LF.answer_text(tokenizer, out[0][len(prompt_ids):])
        decoding = {"mode": "syntax_mask" if mask_only else "two_level"}
        if beta_sm is not None:
            decoding["goals_dropped"] = len(beta_sm.dropped_contradictory_goals) + len(beta_sm.dropped_edge_goals)
        if mask_only and not syntax_only:
            decoding["fallback"] = FALLBACK_SIZE if beta_sm is None else FALLBACK_NO_PLAN
        if not mask_only:
            decoding["dp_budget"] = max_budget
        result["stats"] = dict(processor.get_stats(), **decoding,
                               n_gamma_valid_actions=len(gamma_valid),
                               n_beta_states=beta_sm.num_states if beta_sm is not None else 0,
                               **({"beta_build_error": beta_error} if beta_error else {}))
        del dfa_model, processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, prompts=PROMPTS, meta_hmm=META_HMM, max_budget=MAX_BUDGET, max_new_tokens=MAX_NEW_TOKENS)
    ap.add_argument("--output", required=True, help="output JSON (list of {identifier, llm_output, stats})")
    ap.add_argument("--semantic_vocab_path", default=META_VOCAB)
    args = ap.parse_args()

    tokenizer, model, vocab_size, device = load_lm(args.base_model, args.cuda_device)
    mask_eos_id = eot_id(tokenizer)
    semantic_hmm = semantic_vocab = None
    if not args.syntax_only:
        semantic_hmm = load_semantic_hmm(args.semantic_hmm_path, device=device)
        semantic_vocab = SemanticVocab.from_json(args.semantic_vocab_path)

    def gen(prompt):
        return generate_one(model, tokenizer, prompt, vocab_size=vocab_size, mask_eos_id=mask_eos_id,
                            semantic_hmm=semantic_hmm, semantic_vocab=semantic_vocab,
                            max_budget=args.max_budget, max_new_tokens=args.max_new_tokens,
                            ctrlg_scale=args.ctrlg_scale, hard_mask=args.beta_hard_mask,
                            syntax_only=args.syntax_only, device=device, debug=args.debug)

    run_prompts(args, gen, output_json=args.output)


if __name__ == "__main__":
    main()

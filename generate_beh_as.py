#!/usr/bin/env python3
"""Meta-Ctrl decoding for BEHAVIOR action sequencing.

gamma fixes the JSON list of {"action", "object"} dictionaries and the
admissible (action, object) pairs; beta tracks object states, preconditions and
the goals after quantifier expansion. If beta is too large, trailing goals are
dropped (binary search); if no accepting state remains, the task is decoded
with the syntax mask alone.

    python generate_beh_as.py --semantic_hmm_path models/<meta-hmm> --output_dir outputs/beh_as
"""
import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctrlg
from ctrlg import lm_family as LF
from ctrlg.beta.backward_dp import BetaBackwardDP, SemanticVocab
from ctrlg.beta.beh_goal_expander import expand_goals
from ctrlg.beta.beh_state_machine import BehBetaStateMachine, parse_beh_states
from ctrlg.beta.uniform_meta import resolve_uniform_logw
from ctrlg.gamma.topdown.behavior_twolevel import build_beh_twolevel_state_machine
from ctrlg.models import compute_prompt_end_embedding, load_semantic_hmm
from ctrlg.runner import FALLBACK_NO_PLAN, add_common_args, eot_id, load_lm, run_prompts
from ctrlg.two_level.beh_candidates import BehaviorTwoLevelCandidateTracker
from ctrlg.two_level.beh_tau import BehaviorTwoLevelTauParser
from ctrlg.two_level.processor import GUARANTEE_STATS, TwoLevelProcessor
from ctrlg.two_level.token_mask import TokenMask

PROMPTS = "data/prompts/behavior/action_sequencing/action_sequence_prompts.json"
META_HMM = "models/meta-hmm-as"
META_VOCAB = "data/vocab/hmm_vocab_as_actions_only_full_schema.json"
MAX_BUDGET = 30
MAX_NEW_TOKENS = 2048
MAX_STATES = 100_000      # semantic automaton size limit (states)
MAX_DP_STATES = 50_000    # goals are dropped until the automaton fits the DP table
LARGE_BETA_STATES = 30_000  # above this many states the lookahead covers LARGE_BETA_LOOKAHEAD actions
LARGE_BETA_LOOKAHEAD = 20
BLOCKED_ACTIONS = {       # actions the evaluator handles poorly; never offered
    "LEFT_TRANSFER_CONTENTS_ONTOP", "LEFT_TRANSFER_CONTENTS_INSIDE",
    "RIGHT_TRANSFER_CONTENTS_ONTOP", "RIGHT_TRANSFER_CONTENTS_INSIDE",
    "NAVIGATE_TO", "LEFT_PLACE_NEXTTO_ONTOP", "RIGHT_PLACE_NEXTTO_ONTOP",
}
SIMPLE_GOALS = (
    "onfloor", "ontop", "inside", "nextto", "under",
    "toggled_on", "sliced", "soaked", "frozen", "cooked", "stained", "dusty", "open",
    "not_toggled_on", "not_sliced", "not_soaked", "not_frozen", "not_cooked",
    "not_stained", "not_dusty", "not_open",
)


def build_prompt(tokenizer, user_prompt: str) -> List[int]:
    return LF.prompt_ids(tokenizer, "behavior", "AS", user_prompt)


def parse_objects(prompt_text: str) -> List[Dict]:
    """The {'name', 'category'} dictionaries under "interactable objects"."""
    m = re.search(r"interactable objects:\n(.*?)(?:\n\nPlease|\nPlease)",
                  prompt_text[prompt_text.find("Your task:"):], re.DOTALL)
    objects = []
    for line in (m.group(1).strip().split("\n") if m else []):
        line = line.strip()
        if line.startswith("{"):
            try:
                objects.append(ast.literal_eval(line))
            except Exception:
                pass
    return objects


def enumerate_beh_gamma_valid(sm, inst_to_cid, inst_list) -> List[Tuple[str, Tuple[int, ...]]]:
    """Every (action, canonical ids) gamma can emit."""
    out = []
    for action in sorted(sm.viable_actions):
        arity = sm.action_arity.get(action, 1)
        if arity == 0:
            out.append((action, ()))
        elif arity == 1:
            viable = sm._viable_instance_nodes(action, 0)
            for inst in inst_list:
                seq = sm.instance_token_seqs.get(inst)
                if not seq:
                    continue
                if viable is not None:
                    node, ok = 0, True
                    for t in seq:
                        node = sm.instance_trie.next_node(node, t)
                        if node is None or node not in viable:
                            ok = False
                            break
                    if not ok:
                        continue
                out.append((action, (inst_to_cid[inst],)))
        else:
            out += [(action, (inst_to_cid[a], inst_to_cid[b])) for a in inst_list for b in inst_list]
    return out


def build_beta_with_goal_dropping(init_states, target_states, inst_list, inst_to_cid, gamma_valid, max_budget,
                                  objects, blocked_actions=BLOCKED_ACTIONS, max_states=MAX_STATES,
                                  max_dp_states=MAX_DP_STATES):
    """Semantic automaton with the longest goal prefix that fits.

    Quantified goals are expanded over `objects`; trailing simple goals are
    dropped until the automaton has an accepting state and at most
    `max_dp_states` states. Fitting is monotone in the number dropped, so the
    smallest drop is found by binary search; at least one goal is kept.
    Returns (automaton, n_dropped, n_kept).
    """
    expanded = expand_goals(target_states, objects) if objects else list(target_states)
    goals = [t for t in expanded if t and t[0] in SIMPLE_GOALS]
    n = len(goals)
    cache: Dict[int, Tuple] = {}

    def build(n_drop: int):
        if n_drop not in cache:
            sm = BehBetaStateMachine(
                initial_states=init_states, target_states=goals[:n - n_drop], all_instances=inst_list,
                inst_to_cid=inst_to_cid, gamma_valid_actions=gamma_valid, max_budget=max_budget,
                max_states=max_states, blocked_actions=blocked_actions, objects_meta=objects)
            cache[n_drop] = (sm, sm.num_accept > 0 and sm.num_states <= max_dp_states)
        return cache[n_drop]

    sm, fits = build(0)
    if fits:
        return sm, 0, n
    lo, hi, best = 1, n - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if build(mid)[1]:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    if best is not None:
        return build(best)[0], best, n - best
    return build(n - 1)[0], n - 1, 1


def lookahead_budget(num_states: int, max_budget: int) -> int:
    """Actions the lookahead covers; the plan itself may still use max_budget."""
    return min(max_budget, LARGE_BETA_LOOKAHEAD) if num_states > LARGE_BETA_STATES else max_budget


def _mask_only(model, tokenizer, prompt_ids, dfa_model, vocab_size, eos_id, device, max_new_tokens) -> str:
    """Decode with the syntax mask alone."""
    prompt_ids = LF.prefill_final(tokenizer, prompt_ids)
    mask = TokenMask(dfa_model, prompt_ids, vocab_size, eos_id, device)
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or eos_id,
                             logits_processor=LogitsProcessorList([mask]), **LF.generate_kwargs(tokenizer))
    return LF.answer_text(tokenizer, out[0, len(prompt_ids):])


def generate_one(prompt: Dict, tokenizer, model, *, vocab_size, eos_id, semantic_hmm, semantic_vocab, max_budget,
                 ctrlg_scale, hard_mask, syntax_only, device, max_new_tokens=MAX_NEW_TOKENS, debug=False) -> Dict:
    text = prompt["llm_prompt"]
    prompt_ids = build_prompt(tokenizer, text)
    sm = build_beh_twolevel_state_machine(tokenizer, text, eos_id)
    dfa_model = ctrlg.DFAModel(sm.build_graph_bfs(vocab_size), vocab_size).to(device)
    if syntax_only:
        return {"llm_output": _mask_only(model, tokenizer, prompt_ids, dfa_model, vocab_size, eos_id, device, max_new_tokens),
                "stats": {"mode": "syntax_mask"}}

    inst_list = sorted(sm.all_instances)
    inst_to_cid = {inst: i for i, inst in enumerate(inst_list)}
    gamma_valid = enumerate_beh_gamma_valid(sm, inst_to_cid, inst_list)
    init_states, target_states = parse_beh_states(text)
    beta_sm, n_dropped, n_kept = build_beta_with_goal_dropping(
        init_states, target_states, inst_list, inst_to_cid, gamma_valid, max_budget, parse_objects(text))
    stats = {"goals_dropped": n_dropped, "n_kept": n_kept, "beta_states": beta_sm.num_states}
    if beta_sm.num_accept == 0:
        return {"llm_output": _mask_only(model, tokenizer, prompt_ids, dfa_model, vocab_size, eos_id, device, max_new_tokens),
                "stats": dict(stats, mode="syntax_mask", fallback=FALLBACK_NO_PLAN)}

    dp_budget = lookahead_budget(beta_sm.num_states, max_budget)
    beta_dp = BetaBackwardDP(state_machine=beta_sm, semantic_hmm=semantic_hmm, semantic_vocab=semantic_vocab,
                             max_budget=dp_budget, device=device, hard_mask=hard_mask,
                             uniform_action_logw=resolve_uniform_logw(semantic_hmm, len(gamma_valid)))
    tau = BehaviorTwoLevelTauParser(tokenizer=tokenizer, gamma_dfa=sm, instance_to_canonical=inst_to_cid)
    tracker = BehaviorTwoLevelCandidateTracker(tokenizer=tokenizer, gamma_dfa=sm, tau_parser=tau,
                                               canonical_id_to_instance=inst_list, gamma_valid_actions=gamma_valid)
    beta_init = None
    if semantic_hmm.nn_proj is not None:
        emb = compute_prompt_end_embedding(model, prompt_ids, device=device)
        if int(emb.shape[-1]) == int(semantic_hmm.nn_proj.shape[1]):
            beta_init = semantic_hmm.initial_belief(emb)
    prefix_len = len(sm.prefix_seq)
    processor = TwoLevelProcessor(
        dfa_model, vocab_size=vocab_size, eos_token_id=eos_id, prompt_ids=prompt_ids, tokenizer=tokenizer,
        gamma_dfa=sm, candidate_tracker=tracker, beta_lookahead=beta_sm, beta_dp=beta_dp,
        beta_initial_belief=beta_init, max_meta_budget=max_budget, ctrlg_scale=ctrlg_scale,
        block_start_fn=lambda gs: gs == ("PREFIX", prefix_len), eos_gate_phase="AFTER_DICT",
        close_tokens={t for t, d in sm._after_dict_alts.items() if d == "close"}, device=device, debug=debug,
        **LF.processor_kwargs(tokenizer))
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids], device=device),
                             max_new_tokens=LF.generation_budget(tokenizer, max_new_tokens), do_sample=False,
                             logits_processor=[processor], **LF.generate_kwargs(tokenizer))
    ps = processor.get_stats()
    stats.update(mode="two_level", n_actions=ps["completed_actions"], is_accept=beta_sm.is_accept(ps["beta_state"]),
                 meta_budget_remaining=ps["meta_budget"], dp_budget=dp_budget,
                 **{k: ps[k] for k in GUARANTEE_STATS + ("warmup_tokens", "warmup_forced") if k in ps})
    return {"llm_output": LF.answer_text(tokenizer, out[0, len(prompt_ids):]), "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, prompts=PROMPTS, meta_hmm=META_HMM, max_budget=MAX_BUDGET, max_new_tokens=MAX_NEW_TOKENS)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--semantic_vocab_path", default=META_VOCAB)
    args = ap.parse_args()

    tokenizer, model, vocab_size, device = load_lm(args.base_model, args.cuda_device)
    eos_id = eot_id(tokenizer)
    semantic_hmm = load_semantic_hmm(args.semantic_hmm_path, device=device)
    semantic_vocab = SemanticVocab.from_json(args.semantic_vocab_path)

    def gen(prompt):
        return generate_one(prompt, tokenizer, model, vocab_size=vocab_size, eos_id=eos_id,
                            semantic_hmm=semantic_hmm, semantic_vocab=semantic_vocab, max_budget=args.max_budget,
                            ctrlg_scale=args.ctrlg_scale, hard_mask=args.beta_hard_mask,
                            syntax_only=args.syntax_only, device=device, max_new_tokens=args.max_new_tokens,
                            debug=args.debug)

    run_prompts(args, gen, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

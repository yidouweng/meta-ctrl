#!/usr/bin/env python
"""Search without an LM: plans from the automata alone.

For each prompt this builds the same syntax and semantic automata as the
corresponding generate_*.py and runs breadth-first search over the semantic
automaton from the initial state to an accepting one. Successors are
expanded in lexicographic order of (action, argument ids), so the plan is
the lexicographically least shortest plan. At most 1,000,000 states are
explored; if no accepting state is reached, the path to the visited state
satisfying the most goal atoms is returned (earliest discovered on ties).
The mode is recorded per prompt ("bfs_accept", "greedy_goal_bits",
"accept_at_init").

    python baselines/search_no_lm.py --module vh_as --output_dir outputs/search_vh_as --workers 16
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MAX_EXPLORED = 1_000_000

PROMPT_FILES = {
    "vh_as": "data/prompts/virtualHome/action_sequencing/helm_prompt.json",
    "vh_sd": "data/prompts/virtualHome/subgoal_decomposition/helm_prompt.json",
    "beh_as": "data/prompts/behavior/action_sequencing/action_sequence_prompts.json",
    "beh_sd": None,  # generate_beh_sd.PROMPTS
}

_TOKENIZER = None  # loaded once in the parent, shared via fork


# ---------------------------------------------------------------------------
# Per-module builders: the same automata as the generate_* drivers.
# Each returns (beta_sm, render_fn, goal_progress_fn).
# ---------------------------------------------------------------------------

def build_vh_as(tokenizer, prompt_entry):
    from generate_vh_as import build_canonical_mappings, enumerate_gamma_valid_actions
    from ctrlg.gamma.topdown.virtualhome import build_vh_state_machine
    from ctrlg.beta.vh_state_machine import VHBetaStateMachine

    prompt_text = prompt_entry["llm_prompt"]
    eos_id = _eos_of(tokenizer)
    gamma_sm = build_vh_state_machine(tokenizer=tokenizer, prompt=prompt_text,
                                      eos_token_id=eos_id, max_ws_tokens=5)
    _, canonical_id_to_obj, _ = build_canonical_mappings(gamma_sm)
    gamma_valid = enumerate_gamma_valid_actions(gamma_sm, canonical_id_to_obj)
    beta_sm = VHBetaStateMachine(prompt=prompt_text, gamma_valid_actions=gamma_valid,
                                 debug=False)

    def render(plan):
        if not plan:
            return "{}"
        lines = []
        for s in plan:
            args = []
            for cid in s.args:
                name, raw_id = canonical_id_to_obj[cid]
                args += [str(name), str(raw_id)]
            lines.append(f'"{s.action}": {json.dumps(args)}')
        return "{\n" + ",\n".join(lines) + "\n}"

    def progress(state_id):
        st = beta_sm.states[state_id]
        return _popcount(st.edge_bits) + _popcount(st.action_goal_bits)

    return beta_sm, render, progress


def build_vh_sd(tokenizer, prompt_entry):
    from generate_vh_sd import build_id_mapping
    from ctrlg.gamma.topdown.subgoal_decomp import (
        SDStateMachine, enumerate_gamma_valid_predicates, parse_sd_prompt)
    from ctrlg.beta.sd_state_machine import SDBetaStateMachine

    prompt_text = prompt_entry["llm_prompt"]
    eos_id = _eos_of(tokenizer)
    objects, properties, categories, initial_states, goal_states, action_goals = \
        parse_sd_prompt(prompt_text)
    sd_sm = SDStateMachine(tokenizer, objects, properties, categories, eos_id)
    orig_id_to_canonical, canonical_id_to_obj, sorted_objects = build_id_mapping(objects)
    gamma_valid = enumerate_gamma_valid_predicates(sd_sm, orig_id_to_canonical)
    beta_sm = SDBetaStateMachine(
        initial_states=initial_states, goal_states=goal_states,
        action_goals=action_goals, objects=objects, properties=properties,
        categories=categories, gamma_valid_predicates=gamma_valid,
        orig_id_to_canonical=orig_id_to_canonical,
        canonical_to_info=dict(enumerate(sorted_objects)), debug=False)

    def render(plan):
        atoms = []
        for s in plan:
            args = ", ".join(f"{canonical_id_to_obj[cid][0]}.{canonical_id_to_obj[cid][1]}"
                             for cid in s.args)
            atoms.append(f"{s.action}({args})")
        return json.dumps({"output": atoms})

    def progress(state_id):
        return _popcount(beta_sm.states[state_id].goal_bits)

    return beta_sm, render, progress


def build_beh_as(tokenizer, prompt_entry):
    # As generate_beh_as (budget 30, quantified goals expanded over the objects).
    import generate_beh_as as gb

    prompt_text = prompt_entry["llm_prompt"]
    eos_id = _eos_of(tokenizer)
    sm = gb.build_beh_twolevel_state_machine(tokenizer, prompt_text, eos_id)
    inst_list = sorted(sm.all_instances)
    inst_to_cid = {inst: i for i, inst in enumerate(inst_list)}
    gamma_valid = gb.enumerate_beh_gamma_valid(sm, inst_to_cid, inst_list)
    init_states, target_states = gb.parse_beh_states(prompt_text)

    objects = gb.parse_objects(prompt_text)

    beta_sm, n_dropped, n_kept = gb.build_beta_with_goal_dropping(
        init_states, target_states, inst_list, inst_to_cid, gamma_valid, 30, objects)

    def render(plan):
        items = [{"action": s.action,
                  "object": ",".join(inst_list[c] for c in s.args)}
                 for s in plan]
        return json.dumps(items, indent=1)

    def progress(state_id):
        return _popcount(beta_sm.states[state_id].goal_bits)

    return beta_sm, render, progress


def build_beh_sd(tokenizer, prompt_entry):
    import generate_beh_sd as gbs
    from ctrlg.beta.beh_sd_state_machine import BehSDBetaStateMachine

    prompt_text = prompt_entry["llm_prompt"]
    eos_id = _eos_of(tokenizer)
    sm = gbs.build_beh_sd_twolevel_state_machine(tokenizer, prompt_text, eos_id)
    inst_list = list(sm.all_instances)
    inst_to_cid = {inst: i for i, inst in enumerate(inst_list)}
    gamma_valid = gbs.enumerate_beh_sd_gamma_valid(sm, inst_to_cid, inst_list)
    init_states, target_states = gbs.parse_beh_sd_states(prompt_text)
    objects = gbs._parse_sd_objects(prompt_text)

    # as the paper's run: at most 2,000,000 states, budget 35
    beta_sm = BehSDBetaStateMachine(
        initial_states=init_states, target_states=target_states,
        all_instances=inst_list, inst_to_cid=inst_to_cid,
        gamma_valid_steps=gamma_valid, max_states=2_000_000,
        max_budget=35, objects=objects, edge_budget=gbs.DP_EDGE_BUDGET)

    def render(plan):
        atoms = []
        for s in plan:
            if s.action.startswith("NOT_"):
                pred, neg = s.action[4:].lower(), True
            else:
                pred, neg = s.action.lower(), False
            atom = f"{pred}({', '.join(inst_list[c] for c in s.args)})"
            atoms.append(f"not {atom}" if neg else atom)
        return json.dumps({"output": atoms})

    def progress(state_id):
        return _popcount(beta_sm.states[state_id].goal_bits)

    return beta_sm, render, progress


BUILDERS = {"vh_as": build_vh_as, "vh_sd": build_vh_sd,
            "beh_as": build_beh_as, "beh_sd": build_beh_sd}


# ---------------------------------------------------------------------------
# LM-free BFS over the beta SM transition graph
# ---------------------------------------------------------------------------

def _popcount(x):
    return bin(x).count("1")


def _eos_of(tokenizer):
    if "<|eot_id|>" in tokenizer.get_vocab():
        return int(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    return int(tokenizer.eos_token_id)


def _step_key(t):
    step = t[0]
    return (step.action, tuple(step.args))


def bfs_plan(beta_sm, progress_fn, max_explored=MAX_EXPLORED):
    """Deterministic LM-free BFS. Returns (plan_steps, mode, n_explored)."""
    init = beta_sm.get_initial_state_id()
    if beta_sm.is_accept(init):
        return [], "accept_at_init", 0

    parent = {init: None}          # state -> (prev_state, step) — insertion = BFS order
    queue = deque([init])
    explored = 0
    capped = False

    while queue:
        s = queue.popleft()
        explored += 1
        if explored > max_explored:
            capped = True
            break
        for step, dst in sorted(beta_sm.transitions.get(s, []), key=_step_key):
            if dst == s or dst in parent:
                continue
            parent[dst] = (s, step)
            if beta_sm.is_accept(dst):
                return _reconstruct(parent, dst), "bfs_accept", explored
            queue.append(dst)

    # Fallback: greedy-toward-goal-bits — path to the BFS-visited state with
    # max satisfied goal bits, earliest-discovered wins ties (deterministic).
    best_sid, best_prog = init, progress_fn(init)
    for sid in parent:
        p = progress_fn(sid)
        if p > best_prog:
            best_sid, best_prog = sid, p
    mode = "greedy_goal_bits" + ("_capped" if capped else "")
    return _reconstruct(parent, best_sid), mode, explored


def _reconstruct(parent, sid):
    plan = []
    while parent[sid] is not None:
        prev, step = parent[sid]
        plan.append(step)
        sid = prev
    plan.reverse()
    return plan


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(args_tuple):
    module, idx, prompt_entry, out_dir = args_tuple
    out_path = Path(out_dir) / f"prompt_{idx}.json"
    if out_path.exists():
        return idx, "cached"
    ident = prompt_entry.get("identifier", f"idx_{idx}")
    t0 = time.perf_counter()
    entry = {"identifier": ident, "llm_output": "", "dfa_only": {}}
    try:
        beta_sm, render, progress = BUILDERS[module](_TOKENIZER, prompt_entry)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        plan, mode, explored = bfs_plan(beta_sm, progress)
        t_bfs = time.perf_counter() - t0
        entry["llm_output"] = render(plan)
        entry["dfa_only"] = {
            "mode": mode, "plan_len": len(plan), "explored": explored,
            "beta_states": int(beta_sm.num_states),
            "beta_accept": int(beta_sm.num_accept),
            "build_s": round(t_build, 3), "bfs_s": round(t_bfs, 3),
        }
    except Exception as e:
        import traceback
        entry["dfa_only"] = {"mode": "error",
                             "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-2000:]}
        # emit a syntactically-parseable empty plan so the eval counts it
        entry["llm_output"] = ('{"output": []}' if module.endswith("sd")
                               else ("[]" if module == "beh_as" else "{}"))
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=1)
    return idx, entry["dfa_only"].get("mode", "?")


def main():
    global _TOKENIZER
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, choices=list(BUILDERS))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="models/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prompt_indices", default=None,
                    help="comma-separated subset (default: all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(str(REPO / args.base_model))

    pf = PROMPT_FILES[args.module]
    if pf is None:
        import generate_beh_sd as gbs
        pf = gbs.PROMPTS
    with open(REPO / pf) as f:
        prompts = json.load(f)

    idxs = (list(range(len(prompts))) if args.prompt_indices is None
            else [int(i) for i in args.prompt_indices.split(",")])
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(args.module, i, prompts[i], str(out_dir)) for i in idxs]
    print(f"[dfa_only] {args.module}: {len(tasks)} prompts, "
          f"{args.workers} workers -> {out_dir}", flush=True)

    if args.workers <= 1:
        for t in tasks:
            idx, mode = run_one(t)
            print(f"  prompt {idx}: {mode}", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for idx, mode in pool.imap_unordered(run_one, tasks):
                print(f"  prompt {idx}: {mode}", flush=True)

    # merge
    merged = []
    for i in idxs:
        p = out_dir / f"prompt_{i}.json"
        if p.exists():
            with open(p) as f:
                merged.append(json.load(f))
    with open(out_dir / "merged.json", "w") as f:
        json.dump(merged, f, indent=1)
    modes = {}
    for e in merged:
        m = e.get("dfa_only", {}).get("mode", "?")
        modes[m] = modes.get(m, 0) + 1
    print(f"[dfa_only] {args.module} done: {len(merged)} entries; modes={modes}")
    print(f"[dfa_only] merged -> {out_dir/'merged.json'}")


if __name__ == "__main__":
    main()

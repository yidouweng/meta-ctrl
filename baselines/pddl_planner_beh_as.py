#!/usr/bin/env python
"""Classical-planner baseline for BEHAVIOR action sequencing.

The planner gets the benchmark's full PDDL domain for BEHAVIOR
(virtualhome_eval/resources/behavior/behavior.pddl, 30 ADL actions) and its
problem file for each task (resources/behavior/problem_pddl/<identifier>.pddl):
complete preconditions and effects, more than Meta-Ctrl's automata encode.

Solver: Fast Downward (the `up-fast-downward` package, or a checkout named by
FAST_DOWNWARD), FF heuristic, lazy greedy best-first search with preferred
operators; the domain needs ADL (negative preconditions, `exists`, conditional
effects). Search is deterministic.

PDDL plans are converted to the benchmark's output format:
  * actions: the table ACTION_MAP. The PDDL domain is single-handed, so hand
    actions map to RIGHT_*; clean_stained_*/clean_dusty_* -> CLEAN(target),
    place_onfloor -> RIGHT_PLACE_ONTOP(floor), slice* -> SLICE, soak* -> SOAK.
  * objects: PDDL names are WordNet-style (`plywood_n_01_1`), the evaluator
    expects the prompt's scene names (`plywood_0`). Objects are matched per
    category; instances of one category are told apart by their initial state,
    and interchangeable duplicates are paired in sorted order.

Usage (EAI_ROOT = checkout of embodied-agent-interface):
  python baselines/pddl_planner_beh_as.py --output_dir outputs/planner_beh_as [--timeout 300]
"""
import argparse
import importlib.util
import os
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EAI = Path(os.environ.get("EAI_ROOT", "embodied-agent-interface")) / "src/virtualhome_eval/resources/behavior"
DOMAIN = EAI / "behavior.pddl"
PROBLEM_DIR = EAI / "problem_pddl"
PROMPTS = REPO / "data/prompts/behavior/action_sequencing/action_sequence_prompts.json"


def fast_downward() -> str:
    """Path of fast-downward.py: $FAST_DOWNWARD, else the `up-fast-downward` package."""
    if os.environ.get("FAST_DOWNWARD"):
        return os.environ["FAST_DOWNWARD"]
    spec = importlib.util.find_spec("up_fast_downward")
    if spec is None or spec.origin is None:
        sys.exit("Fast Downward not found: `pip install up-fast-downward` or set FAST_DOWNWARD")
    return str(Path(spec.origin).parent / "downward/fast-downward.py")

# PDDL action -> (benchmark action, indices of the plan args that become the
# eval "object" field). Agent args are dropped implicitly by the indices.
ACTION_MAP = {
    # navigate_to is dropped: the evaluator does not execute NAVIGATE_TO and
    # treats objects as reachable.
    "navigate_to": None,
    "grasp": ("RIGHT_GRASP", (0,)),
    "release": ("RIGHT_RELEASE", (0,)),
    "place_ontop": ("RIGHT_PLACE_ONTOP", (1,)),
    "place_inside": ("RIGHT_PLACE_INSIDE", (1,)),
    "open": ("OPEN", (0,)),
    "close": ("CLOSE", (0,)),
    "slice": ("SLICE", (0,)),
    "slice-carvingknife": ("SLICE", (0,)),
    "place_onfloor": ("RIGHT_PLACE_ONTOP", (1,)),
    "place_nextto": ("RIGHT_PLACE_NEXTTO", (1,)),
    "place_nextto_ontop": ("RIGHT_PLACE_NEXTTO_ONTOP", (1, 2)),
    "clean_stained_brush": ("CLEAN", (1,)),
    "clean_stained_cloth": ("CLEAN", (1,)),
    "clean_stained_handowel": ("CLEAN", (1,)),
    "clean_stained_towel": ("CLEAN", (1,)),
    "clean_stained_dishtowel": ("CLEAN", (1,)),
    "clean_stained_dishwasher": ("CLEAN", (1,)),
    "clean_stained_rag": ("CLEAN", (1,)),
    "soak": ("SOAK", (0,)),
    "soak_teapot": ("SOAK", (0,)),
    "place_under": ("RIGHT_PLACE_UNDER", (1,)),
    "toggle_on": ("TOGGLE_ON", (0,)),
    "clean_dusty_rag": ("CLEAN", (1,)),
    "clean_dusty_towel": ("CLEAN", (1,)),
    "clean_dusty_cloth": ("CLEAN", (1,)),
    "clean_dusty_brush": ("CLEAN", (1,)),
    "clean_dusty_vacuum": ("CLEAN", (1,)),
    "freeze": ("FREEZE", (0,)),
    "cook": ("COOK", (0,)),
}

_CAT_RE = re.compile(r"^(.*)_n_(\d+)$")


def pddl_cat_of(pddl_obj):
    """plywood_n_01_1 -> (category 'plywood.n.01', instance token)."""
    m = re.match(r"^(.*)_n_(\d+)_(\d+)$", pddl_obj)
    if not m:
        return None, None
    return f"{m.group(1)}.n.{m.group(2)}", int(m.group(3))


def parse_prompt(prompt_text):
    """-> (objects [{'name','category'}], init atoms [tuple])."""
    t = prompt_text[prompt_text.find("Your task:"):]
    objects = []
    m = re.search(r"interactable objects:\n(.*?)(?:\n\nPlease|\nPlease)", t, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    objects.append(ast.literal_eval(line))
                except Exception:
                    pass
    init = []
    m = re.search(r"initial environment state:\n(.*?)\n\n", t, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith("["):
                try:
                    init.append(tuple(ast.literal_eval(line)))
                except Exception:
                    pass
    return objects, init


def parse_problem(path):
    """-> (objects {pddl_name: pddl_type}, init atoms [tuple], goal text)."""
    txt = path.read_text()
    objs = {}
    m = re.search(r"\(:objects(.*?)\)\s*\(:init", txt, re.DOTALL)
    decl = m.group(1)
    for group in re.finditer(r"((?:\S+\s+)+)-\s*(\S+)", decl):
        for name in group.group(1).split():
            objs[name] = group.group(2)
    init = []
    m = re.search(r"\(:init(.*?)\)\s*\(:goal", txt, re.DOTALL)
    init_txt = m.group(1)
    # drop negated atoms: `(not (dusty x))` must not count as `dusty x`
    init_txt = re.sub(r"\(not\s*\([^()]*\)\s*\)", " ", init_txt)
    for atom in re.finditer(r"\(([^()]+)\)", init_txt):
        parts = atom.group(1).split()
        if parts[0] != "same_obj":
            init.append(tuple(parts))
    return objs, init


def build_object_mapping(pddl_objs, prompt_objs, pddl_init, prompt_init):
    """Map pddl object name -> prompt instance name.

    Category-exact first; same-category duplicates disambiguated by fixpoint
    constraint propagation over init atoms; leftovers by sorted order.
    Returns (mapping, mode) where mode notes how ambiguity was resolved.
    """
    by_cat_pddl = defaultdict(list)
    for name in sorted(pddl_objs):
        cat, _ = pddl_cat_of(name)
        if cat is not None and cat != "agent.n.01":
            by_cat_pddl[cat].append(name)
    by_cat_prompt = defaultdict(list)
    for o in prompt_objs:
        by_cat_prompt[o["category"]].append(o["name"])
    for v in by_cat_prompt.values():
        v.sort()

    # candidate sets
    cand = {}
    for cat, plist in by_cat_pddl.items():
        pr = by_cat_prompt.get(cat, [])
        for p in plist:
            cand[p] = set(pr)

    # index prompt init atoms:
    #   unary: pred -> set(inst); binary: pred -> set((a,b))
    prompt_unary = defaultdict(set)
    prompt_binary = defaultdict(set)
    for atom in prompt_init:
        if len(atom) == 2:
            prompt_unary[atom[0]].add(atom[1])
        elif len(atom) == 3:
            prompt_binary[atom[0]].add((atom[1], atom[2]))

    # prompt name -> category (from the interactable list; fall back to a
    # wordnet-style parse for names like agent.n.01_1)
    name_to_cat = {o["name"]: o["category"] for o in prompt_objs}

    def cat_of_prompt_name(n):
        if n in name_to_cat:
            return name_to_cat[n]
        m = re.match(r"^(.*\.n\.\d+)_\d+$", n)
        return m.group(1) if m else None

    used_init = False
    for _ in range(10):  # fixpoint
        changed = False
        assigned = {p: next(iter(c)) for p, c in cand.items() if len(c) == 1}
        taken = defaultdict(set)  # cat -> assigned prompt names
        for p, n in assigned.items():
            taken[pddl_cat_of(p)[0]].add(n)
        for p, c in cand.items():
            if len(c) <= 1:
                continue
            newc = set(c) - (taken[pddl_cat_of(p)[0]] - {assigned.get(p)})
            # unary constraints (exact predicate match)
            for atom in pddl_init:
                if len(atom) == 2 and atom[1] == p:
                    ok = prompt_unary.get(atom[0], set())
                    filt = {n for n in newc if n in ok}
                    if filt:
                        newc, used_init = filt, True
            # binary constraints
            for atom in pddl_init:
                if len(atom) == 3 and p in atom[1:]:
                    pred, a, b = atom
                    pairs = prompt_binary.get(pred, set())
                    if a == p:
                        if b in assigned:  # partner known: exact filter
                            filt = {n for n in newc if (n, assigned[b]) in pairs}
                        else:  # category-level: some partner-category object
                            bcat = pddl_cat_of(b)[0]
                            filt = {n for n in newc
                                    if any(x == n and cat_of_prompt_name(y) == bcat
                                           for x, y in pairs)}
                        if filt:
                            newc, used_init = filt, True
                    elif b == p:
                        if a in assigned:
                            filt = {n for n in newc if (assigned[a], n) in pairs}
                        else:
                            acat = pddl_cat_of(a)[0]
                            filt = {n for n in newc
                                    if any(y == n and cat_of_prompt_name(x) == acat
                                           for x, y in pairs)}
                        if filt:
                            newc, used_init = filt, True
            if newc != c:
                cand[p] = newc
                changed = True
        if not changed:
            break

    # finalize: unique first, then sorted-order fill within category
    mapping = {}
    fallback_order = False
    for cat, plist in by_cat_pddl.items():
        remaining = [n for n in by_cat_prompt.get(cat, [])]
        # pass 1: unique candidates
        for p in plist:
            c = cand.get(p, set())
            if len(c) == 1:
                n = next(iter(c))
                if n in remaining:
                    mapping[p] = n
                    remaining.remove(n)
        # pass 2: order fill
        for p in plist:
            if p in mapping:
                continue
            if remaining:
                mapping[p] = remaining.pop(0)
                fallback_order = True
    mode = ("init_aligned" if used_init else "category_unique")
    if fallback_order:
        mode += "+order_fill"
    return mapping, mode


def repair_problem(problem_path, td):
    """Repair benchmark-shipped problem files that use objects in :init/:goal
    without declaring them in :objects. Returns a path to a repaired copy (or
    the original path if nothing needs repair) and the list of added objects."""
    txt = problem_path.read_text()
    m = re.search(r"\(:objects(.*?)\)\s*\(:init", txt, re.DOTALL)
    declared = set(re.sub(r"-\s*\S+", " ", m.group(1)).split())
    used = set()
    for section in re.findall(r"\(:(?:init|goal)(.*?)\)\s*(?:\(:goal|\)\s*$)",
                              txt, re.DOTALL):
        for atom in re.finditer(r"\(([^()]+)\)", section):
            parts = atom.group(1).split()
            used.update(p for p in parts[1:] if _CAT_RE.match(re.sub(r"_\d+$", "", p)))
    missing = sorted(used - declared)
    if not missing:
        return problem_path, []
    decls = " ".join(f"{o} - {re.sub(r'_[0-9]+$', '', o)}" for o in missing)
    txt = re.sub(r"\(:objects(.*?)\)(\s*\(:init)",
                 lambda mm: f"(:objects{mm.group(1)} {decls}){mm.group(2)}",
                 txt, count=1, flags=re.DOTALL)
    out = Path(td) / problem_path.name
    out.write_text(txt)
    return out, missing


def run_fd(problem_path, timeout):
    with tempfile.TemporaryDirectory() as td:
        plan_file = Path(td) / "plan.txt"
        prob, repaired = repair_problem(problem_path, td)
        cmd = [sys.executable, fast_downward(), "--plan-file", str(plan_file),
               str(DOMAIN), str(prob),
               "--evaluator", "hff=ff()",
               "--search", "lazy_greedy([hff], preferred=[hff])"]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            return None, time.perf_counter() - t0, "timeout"
        wall = time.perf_counter() - t0
        status_suffix = f" (repaired: +{','.join(repaired)})" if repaired else ""
        if not plan_file.exists():
            return None, wall, f"no_plan (exit {proc.returncode}){status_suffix}"
        steps = []
        for line in plan_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                steps.append(line[1:-1].split())
        return steps, wall, "solved" + status_suffix


def convert_plan(steps, mapping):
    out = []
    for parts in steps:
        act, args = parts[0], parts[1:]
        if act not in ACTION_MAP:
            raise KeyError(f"unmapped PDDL action {act}")
        if ACTION_MAP[act] is None:
            continue  # navigate_to — see ACTION_MAP note
        name, idxs = ACTION_MAP[act]
        # FD sometimes emits `(navigate_to agent_n_01_1 agent_n_01_1)`
        # (self-navigation, trivially satisfies the not-in-reach precond).
        # The benchmark has no agent object — drop such steps.
        if any(args[i].startswith("agent_n_01") for i in idxs):
            continue
        objs = []
        for i in idxs:
            p = args[i]
            if p not in mapping:
                raise KeyError(f"unmapped PDDL object {p}")
            objs.append(mapping[p])
        out.append({"action": name, "object": ",".join(objs)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="outputs/planner_beh_as")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--prompt_indices", default=None)
    args = ap.parse_args()

    with open(PROMPTS) as f:
        prompts = json.load(f)
    idxs = (list(range(len(prompts))) if args.prompt_indices is None
            else [int(i) for i in args.prompt_indices.split(",")])
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, stats = [], []
    for i in idxs:
        entry = prompts[i]
        ident = entry["identifier"]
        rec = {"identifier": ident, "llm_output": "[]"}
        st = {"idx": i, "identifier": ident}
        prob = PROBLEM_DIR / f"{ident}.pddl"
        try:
            if not prob.exists():
                raise FileNotFoundError(f"no problem file {prob.name}")
            steps, wall, status = run_fd(prob, args.timeout)
            st.update(status=status, planner_wall_s=round(wall, 2))
            if steps is not None:
                prompt_objs, prompt_init = parse_prompt(entry["llm_prompt"])
                pddl_objs, pddl_init = parse_problem(prob)
                mapping, map_mode = build_object_mapping(
                    pddl_objs, prompt_objs, pddl_init, prompt_init)
                actions = convert_plan(steps, mapping)
                rec["llm_output"] = json.dumps(actions, indent=1)
                st.update(plan_len=len(steps), n_actions=len(actions),
                          map_mode=map_mode)
        except Exception as e:
            st.update(status="bridge_error", error=f"{type(e).__name__}: {e}")
        merged.append(rec)
        stats.append(st)
        print(f"[{i:3d}] {st.get('status')} "
              f"wall={st.get('planner_wall_s','-')}s "
              f"len={st.get('plan_len','-')} {st.get('error','')}",
              flush=True)

    with open(out_dir / "merged.json", "w") as f:
        json.dump(merged, f, indent=1)
    with open(out_dir / "planner_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    n_solved = sum(1 for s in stats if str(s.get("status", "")).startswith("solved"))
    walls = [s["planner_wall_s"] for s in stats if "planner_wall_s" in s]
    print(f"\nsolved {n_solved}/{len(stats)}; "
          f"mean wall {sum(walls)/max(len(walls),1):.2f}s")
    print(f"merged -> {out_dir/'merged.json'}")


if __name__ == "__main__":
    main()

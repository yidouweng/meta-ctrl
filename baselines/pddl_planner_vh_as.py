#!/usr/bin/env python
"""Classical-planner baseline for VirtualHome action sequencing.

The planner gets the benchmark's full PDDL domain for VirtualHome
(virtualhome_eval/resources/virtualhome/virtualhome.pddl, 33 ADL actions):
complete preconditions and effects, more than Meta-Ctrl's automata encode.
By default each problem (objects, initial state, goals) is grounded from the
task prompt, the same information Meta-Ctrl sees; `--problems shipped` uses
the benchmark's own problem files instead.

Solver: Fast Downward (the `up-fast-downward` package, or a checkout named by
FAST_DOWNWARD), FF heuristic, lazy greedy best-first search with preferred
operators.

PDDL plans are converted to the benchmark's output format with the table
ACTION_MAP: the character argument is dropped, wipe/drop keep only the acted-on
object, and plug_in/plug_out steps are dropped because the evaluator's action
vocabulary has no PLUGIN/PLUGOUT. PDDL object symbols are mapped back to the
prompt's (name, id) pairs.

Usage (EAI_ROOT = checkout of embodied-agent-interface):
  python baselines/pddl_planner_vh_as.py --output_dir outputs/planner_vh_as
"""
import argparse
import importlib.util
import os
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VH = Path(os.environ.get("EAI_ROOT", "embodied-agent-interface")) / "src/virtualhome_eval/resources/virtualhome"
DOMAIN = VH / "virtualhome.pddl"
PROBLEM_DIR = VH / "problem_pddl"
PROMPTS = REPO / "data/prompts/virtualHome/action_sequencing/helm_prompt.json"


def fast_downward() -> str:
    """Path of fast-downward.py: $FAST_DOWNWARD, else the `up-fast-downward` package."""
    if os.environ.get("FAST_DOWNWARD"):
        return os.environ["FAST_DOWNWARD"]
    spec = importlib.util.find_spec("up_fast_downward")
    if spec is None or spec.origin is None:
        sys.exit("Fast Downward not found: `pip install up-fast-downward` or set FAST_DOWNWARD")
    return str(Path(spec.origin).parent / "downward/fast-downward.py")

# PDDL action -> (benchmark action, indices of plan args to emit) or None to
# drop the step. Plan args include the character at index 0.
ACTION_MAP = {
    "walk_towards": ("WALK", (1,)),
    "walk_into": ("WALK", (1,)),
    "find": ("FIND", (1,)),
    "sit": ("SIT", (1,)),
    "standup": ("STANDUP", ()),
    "grab": ("GRAB", (1,)),
    "open": ("OPEN", (1,)),
    "close": ("CLOSE", (1,)),
    "put_on": ("PUTBACK", (1, 2)),
    "put_on_character": ("PUTON", (1,)),
    "put_inside": ("PUTIN", (1, 2)),
    "switch_on": ("SWITCHON", (1,)),
    "switch_off": ("SWITCHOFF", (1,)),
    "drink": ("DRINK", (1,)),
    "turn_to": ("TURNTO", (1,)),
    "look_at": ("LOOKAT", (1,)),
    "wipe": ("WIPE", (1,)),
    "drop": ("DROP", (1,)),
    "read": ("READ", (1,)),
    "touch": ("TOUCH", (1,)),
    "lie": ("LIE", (1,)),
    "pour": ("POUR", (1, 2)),
    "type": ("TYPE", (1,)),
    "watch": ("WATCH", (1,)),
    "move": ("MOVE", (1,)),
    "wash": ("WASH", (1,)),
    "squeeze": ("SQUEEZE", (1,)),
    "plug_in": None,   # not in the evaluator's action vocabulary
    "plug_out": None,
    "cut": ("CUT", (1,)),
    "eat": ("EAT", (1,)),
    "sleep": ("SLEEP", ()),
    "wake_up": ("WAKEUP", ()),
}


# ---------------------------------------------------------------------------
# Problem grounding from the prompt (default mode): objects, current state and
# node/edge goals come from the prompt. States the prompt leaves unlisted get
# defaults: HAS_SWITCH -> off, HAS_PLUG -> plugged_in, CAN_OPEN -> open.
# Action goals ("GRAB must be included") have no PDDL counterpart, so the
# planner does not address them.
# ---------------------------------------------------------------------------

NODE_STATE_PRED = {
    "CLEAN": "clean", "DIRTY": "dirty", "CLOSED": "closed", "OPEN": "open",
    "OFF": "off", "ON": "on", "PLUGGED_IN": "plugged_in",
    "PLUGGED_OUT": "plugged_out",
}
CHAR_STATE_PRED = {"SITTING": "sitting", "LYING": "lying"}


def _sym(name, oid):
    return f"{name}_{oid}"


def ground_problem(prompt_text, ident):
    t = prompt_text[prompt_text.find("Objects in the scene:",
                                     prompt_text.find("Input:")):]

    # objects: name, id, properties
    objs = []  # (name, id, [props])
    for m in re.finditer(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*), id:\s*(\d+), properties:\s*\[([^\]]*)\]",
            t, re.MULTILINE):
        props = [p.strip().strip("'\"") for p in m.group(3).split(",") if p.strip()]
        objs.append((m.group(1), m.group(2), props))
    name_first = {}
    for name, oid, _ in objs:
        name_first.setdefault(name, oid)
    char_syms = [_sym(n, i) for n, i, _ in objs if n == "character"]
    char = char_syms[0] if char_syms else "character_0"

    init = set()
    for name, oid, props in objs:
        s = _sym(name, oid)
        for p in props:
            init.add(f"({p.lower()} {s})")

    # node states
    m = re.search(r"Nodes:\n(.*?)\n\nEdges:", t, re.DOTALL)
    stated = {}  # sym -> set(states)
    if m:
        for mm in re.finditer(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*), states: \[([^\]]*)\]",
                m.group(1), re.MULTILINE):
            name = mm.group(1)
            oid = name_first.get(name)
            if oid is None:
                continue
            s = _sym(name, oid)
            states = [x.strip().strip("'\"") for x in mm.group(2).split(",") if x.strip()]
            stated.setdefault(s, set()).update(states)
            for st in states:
                if st in NODE_STATE_PRED:
                    init.add(f"({NODE_STATE_PRED[st]} {s})")
                elif st in CHAR_STATE_PRED and name == "character":
                    init.add(f"({CHAR_STATE_PRED[st]} {char})")

    # init-completion defaults
    for name, oid, props in objs:
        s = _sym(name, oid)
        st = stated.get(s, set())
        if "HAS_SWITCH" in props and not ({"ON", "OFF"} & st):
            init.add(f"(off {s})")
        if "HAS_PLUG" in props and not ({"PLUGGED_IN", "PLUGGED_OUT"} & st):
            init.add(f"(plugged_in {s})")
        # The domain's switch_on unconditionally requires plugged_in, but the
        # simulator only enforces plugging for HAS_PLUG objects. For plugless
        # switchables (e.g. a computer with HAS_SWITCH but no HAS_PLUG) mark
        # them plugged_in so switch_on is applicable — mirrors the executor.
        if "HAS_SWITCH" in props and "HAS_PLUG" not in props \
                and not ({"PLUGGED_IN", "PLUGGED_OUT"} & st):
            init.add(f"(plugged_in {s})")
        if "CAN_OPEN" in props and not ({"OPEN", "CLOSED"} & st):
            init.add(f"(open {s})")

    def rel_atom(rel, a_name, a_id, b_name, b_id):
        a, b = _sym(a_name, a_id), _sym(b_name, b_id)
        a_char, b_char = a_name == "character", b_name == "character"
        if rel in ("NEAR", "CLOSE"):
            return f"(next_to {a} {b})" if a_char else \
                   (f"(next_to {b} {a})" if b_char else f"(obj_next_to {a} {b})")
        if rel == "INSIDE":
            return f"(inside {a} {b})" if a_char else f"(obj_inside {a} {b})"
        if rel == "ON":
            if a_char:
                return f"(ontop {a} {b})"
            if b_char:
                return f"(on_char {a} {b})"
            return f"(obj_ontop {a} {b})"
        if rel == "FACING":
            return f"(facing {a} {b})" if a_char else None
        if rel in ("HOLDS_RH", "HOLDS_LH"):
            pred = rel.lower()
            return f"({pred} {a} {b})" if a_char else None
        return None

    # init edges: "<a> (id) is REL to <b> (id)"
    m = re.search(r"Edges:\n(.*?)-{5,}", t, re.DOTALL)
    if m:
        for mm in re.finditer(
                r"<([A-Za-z_][A-Za-z0-9_]*)> \((\d+)\) is ([A-Z_]+) to "
                r"<([A-Za-z_][A-Za-z0-9_]*)> \((\d+)\)", m.group(1)):
            atom = rel_atom(mm.group(3), mm.group(1), mm.group(2),
                            mm.group(4), mm.group(5))
            if atom:
                init.add(atom)

    goal = set()
    m = re.search(r"Node goals are:\n(.*?)-{5,}", t, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            mm = re.match(r"([A-Za-z_][A-Za-z0-9_]*) is ([A-Z_]+)$", line.strip())
            if not mm:
                continue
            name, st = mm.group(1), mm.group(2)
            oid = name_first.get(name)
            if oid is None:
                continue
            if st in CHAR_STATE_PRED and name == "character":
                goal.add(f"({CHAR_STATE_PRED[st]} {char})")
            elif st in NODE_STATE_PRED:
                goal.add(f"({NODE_STATE_PRED[st]} {_sym(name, oid)})")

    m = re.search(r"Edge goals are:\n(.*?)-{5,}", t, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            mm = re.match(
                r"([A-Za-z_][A-Za-z0-9_]*) is ([A-Z_]+) to ([A-Za-z_][A-Za-z0-9_]*)$",
                line.strip())
            if not mm:
                continue
            a, rel, b = mm.groups()
            if a not in name_first or b not in name_first:
                continue
            atom = rel_atom(rel, a, name_first[a], b, name_first[b])
            if atom:
                goal.add(atom)

    if not goal:
        return None  # nothing expressible (e.g. action-goals-only prompt)

    obj_decl = " ".join(_sym(n, i) for n, i, _ in objs if n != "character")
    lines = [f"(define (problem p{ident})",
             "    (:domain virtualhome)",
             "    (:objects",
             f"    {char} - character",
             f"    {obj_decl} - object",
             ")",
             "    (:init " + "\n    ".join(sorted(init)) + ")",
             "    (:goal (and " + " ".join(sorted(goal)) + "))",
             ")"]
    return "\n".join(lines)


def prompt_name_to_id(prompt_text):
    """Parse the 'Objects in the scene' block -> {name: id_str}."""
    mapping, dup = {}, set()
    for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*), id:\s*(\d+)",
                         prompt_text, re.MULTILINE):
        name, oid = m.group(1), m.group(2)
        if name in mapping and mapping[name] != oid:
            dup.add(name)
            continue  # keep first id
        mapping.setdefault(name, oid)
    return mapping, sorted(dup)


def find_problem(ident):
    hits = list(PROBLEM_DIR.glob(f"*/{ident}.pddl"))
    return hits[0] if hits else None


def run_fd(problem_path, timeout):
    with tempfile.TemporaryDirectory() as td:
        plan_file = Path(td) / "plan.txt"
        cmd = [sys.executable, fast_downward(), "--plan-file", str(plan_file),
               str(DOMAIN), str(problem_path),
               "--evaluator", "hff=ff()",
               "--search", "lazy_greedy([hff], preferred=[hff])"]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            return None, time.perf_counter() - t0, "timeout"
        wall = time.perf_counter() - t0
        if not plan_file.exists():
            return None, wall, f"no_plan (exit {proc.returncode})"
        steps = []
        for line in plan_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                steps.append(line[1:-1].split())
        return steps, wall, "solved"


def convert_plan(steps, resolve):
    """resolve(pddl_symbol) -> (name, id_str)."""
    lines = []
    for parts in steps:
        act, args = parts[0], parts[1:]
        if act not in ACTION_MAP:
            raise KeyError(f"unmapped PDDL action {act}")
        if ACTION_MAP[act] is None:
            continue  # plug_in/plug_out — eval vocabulary gap
        name, idxs = ACTION_MAP[act]
        out_args = []
        for i in idxs:
            n, oid = resolve(args[i])
            out_args += [n, oid]
        lines.append(f'"{name}": {json.dumps(out_args)}')
    return "{\n" + ",\n".join(lines) + "\n}" if lines else "{}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="outputs/planner_vh_as")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--prompt_indices", default=None)
    ap.add_argument("--problems", choices=["prompt", "shipped"], default="prompt",
                    help="prompt: ground :init/:goal from the task prompt (default); "
                         "shipped: use the benchmark's problem_pddl files")
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
        rec = {"identifier": ident, "llm_output": "{}"}
        st = {"idx": i, "identifier": ident}
        try:
            if args.problems == "prompt":
                ptxt = ground_problem(entry["llm_prompt"], ident)
                if ptxt is None:
                    st.update(status="no_expressible_goal")
                    merged.append(rec); stats.append(st)
                    print(f"[{i:3d}] {ident:8s} no_expressible_goal", flush=True)
                    continue
                tf = tempfile.NamedTemporaryFile(
                    "w", suffix=".pddl", delete=False)
                tf.write(ptxt); tf.close()
                prob = Path(tf.name)
            else:
                prob = find_problem(ident)
                if prob is None:
                    st.update(status="no_problem_file")
                    merged.append(rec); stats.append(st)
                    continue
            steps, wall, status = run_fd(prob, args.timeout)
            st.update(status=status, planner_wall_s=round(wall, 2))
            if steps is not None:
                if args.problems == "prompt":
                    def resolve(sym):
                        n, oid = sym.rsplit("_", 1)
                        return n, oid
                else:
                    name_to_id, dups = prompt_name_to_id(entry["llm_prompt"])
                    if dups:
                        st["duplicate_names_first_id"] = dups
                    def resolve(sym):
                        if sym not in name_to_id:
                            raise KeyError(f"object {sym} not in scene list")
                        return sym, name_to_id[sym]
                rec["llm_output"] = convert_plan(steps, resolve)
                n_actions = rec["llm_output"].count('": [')
                st.update(plan_len=len(steps), n_actions=n_actions)
        except Exception as e:
            st.update(status="bridge_error", error=f"{type(e).__name__}: {e}")
        merged.append(rec)
        stats.append(st)
        print(f"[{i:3d}] {ident:8s} {st.get('status')} "
              f"wall={st.get('planner_wall_s','-')}s "
              f"len={st.get('plan_len','-')} {st.get('error','')}", flush=True)

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

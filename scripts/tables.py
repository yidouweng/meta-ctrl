#!/usr/bin/env python3
"""Rebuild the paper's Llama 3 8B EAI numbers from the evaluator outputs in results/.

    python scripts/tables.py           # print every table
    python scripts/tables.py --check   # also compare each cell with the expected value

Task and execution success rates, in percent. VirtualHome cells are the mean
over six evaluator runs (PYTHONHASHSEED 0-5); BEHAVIOR is evaluated once.

Held-out identifiers (Table 6) split each module's tasks by data/splits/. The
VH AS evaluator writes per-task results (execution only) for 301 of its 342
tasks, and 4 VH AS tasks (339_1, 627_1, 84_1, 93_1) are in neither split: the
splits are defined over the VH SD identifiers, which they lack.
"""
import argparse
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

R = Path("results")  # set by --results
SUB = {"as": "action_sequencing", "sd": "subgoal_decomposition"}


def vh(run: str, module: str):
    """Mean (task, exec) over <run>_s{0..5}_eval (or <run>_eval if evaluated once).
    The AS summary stores percentages, SD fractions."""
    scale = 1.0 if module == "as" else 100.0
    t, e = [], []
    files = sorted(glob.glob(f"{R}/{run}_s[0-9]_eval/virtualhome/evaluate_results/{SUB[module]}/*/summary.json")) \
        or glob.glob(f"{R}/{run}_eval/virtualhome/evaluate_results/{SUB[module]}/*/summary.json")  # one evaluator run
    seeds = glob.glob(f"{R}/{run}_s[0-9]_eval")
    if seeds and (len(seeds) != 6 or len(files) != 6):
        # a missing run would silently change the mean; refuse instead
        sys.exit(f"{run}: expected six evaluator runs (s0-s5), found {len(seeds)} folders with {len(files)} summaries")
    for f in files:
        d = json.load(open(f))
        t.append(d["goal_evaluation"]["task_success_rate"] * scale)
        e.append(d["trajectory_evaluation"]["execution_success_rate"] * scale)
    return (st.mean(t), st.mean(e)) if t else None


def beh(run: str, module: str):
    """(task, exec) of <run>_eval. Runs evaluated in parts store a merged summary."""
    name = "merged_outputs.json" if module == "as" else "merged.json"
    f = R / f"{run}_eval/behavior/evaluate_results/{SUB[module]}/summary/{name}"
    if not f.exists():
        return None
    d = json.load(open(f))
    if "goal_evaluation" in d:
        return d["goal_evaluation"]["task_success_rate"] * 100, d["trajectory_evaluation"]["execution_success_rate"] * 100
    return d["task_success_rate"] * 100, d["execution_success_rate"] * 100


# Each row: run names for (VH AS, BEH AS, VH SD, BEH SD); None = not in this table.
def _arm(lm, arm):
    return tuple(f"{lm}_{m}_{arm}" for m in ("vh_as", "beh_as", "vh_sd", "beh_sd"))


LLAMA = _arm("llama", "trained")
ABLATION = [
    ("Classical planner, full domain", ("planner_vh_as", "planner_beh_as", None, None)),
    ("Search, no LLM (our automata)", ("search_vh_as", "search_beh_as", "search_vh_sd", "search_beh_sd")),
    ("+ syntax (gamma mask, no lookahead)", _arm("llama", "syntax")),
    ("goal reachability", _arm("llama", "mask")),
    ("valid-plan counting", _arm("llama", "uniform")),
    ("weighted counting (Meta-Ctrl)", LLAMA),
]
XGRAMMAR = [("XGrammar (syntax only)", ("xgrammar_vh_as", "xgrammar_beh_as", "xgrammar_vh_sd", "xgrammar_beh_sd"))]

# Expected values (task, exec), used by --check.
EXPECTED = {
    "Llama 3 8B + Meta-Ctrl": [(88.9, 97.7), (43.0, 94.0), (91.1, 96.2), (45.0, 61.0)],
    "Classical planner, full domain": [(47.2, 84.6), (13.0, 37.0), None, None],
    "Search, no LLM (our automata)": [(82.3, 93.1), (31.0, 77.0), (80.5, 86.7), (52.0, 64.0)],
    "+ syntax (gamma mask, no lookahead)": [(2.0, 97.4), (12.0, 17.0), (0.0, 95.9), (0.0, 61.0)],
    "goal reachability": [(80.5, 88.5), (42.0, 83.0), (90.5, 95.0), (37.0, 52.0)],
    "valid-plan counting": [(87.4, 95.7), (40.0, 90.0), (91.1, 95.3), (32.0, 51.0)],
    "weighted counting (Meta-Ctrl)": [(88.9, 97.7), (43.0, 94.0), (91.1, 96.2), (45.0, 61.0)],
    "XGrammar (syntax only)": [(36.9, 39.7), (13.0, 23.0), (58.6, 73.4), (25.0, 34.0)],
}
# Held-out table, per module: Meta-Ctrl (H, S), valid-plan counting (H, S), gap (H, S), s.e. (H, S).
EXPECTED_HOLDOUT = {
    "vh_as": ((95.7, 98.5), (92.6, 97.0), (3.2, 1.5), (3.4, 1.5)),
    "vh_sd": ((89.2, 91.9), (89.2, 91.9), (0.0, 0.0), (4.3, 2.5)),
    "beh_as": ((56.7, 37.1), (46.7, 37.1), (10.0, 0.0), (12.8, 8.2)),
    "beh_sd": ((53.3, 41.4), (36.7, 30.0), (16.7, 11.4), (12.7, 8.0)),
}


def row(runs):
    return [None if run is None else fn(run, mod)
            for run, fn, mod in zip(runs, (vh, beh, vh, beh), ("as", "as", "sd", "sd"))]


HOLDOUT_RUNS = {m: {a: f"llama_{m}_{a}" for a in ("uniform", "trained")}
                for m in ("vh_as", "vh_sd", "beh_as", "beh_sd")}


def per_task(module: str, run: str):
    """{identifier: score in [0, 1]}: execution for VH AS (its evaluator has no
    per-task goal flag), task success otherwise; VirtualHome averaged over runs."""
    if module.startswith("vh"):
        key = "executable" if module == "vh_as" else "success"
        per = {}
        for f in glob.glob(f"{R}/{run}_s[0-9]_eval/virtualhome/evaluate_results/{SUB[module[3:]]}/*/error_info.json"):
            for k, v in json.load(open(f)).items():
                k = k[len("scene_1_"):] if k.startswith("scene_1_") else k
                per.setdefault(k, []).append(bool(v.get(key)))
        return {k: st.mean(v) for k, v in per.items()}
    if module == "beh_as":
        log = json.load(open(R / f"{run}_eval/behavior/evaluate_results/action_sequencing/log/merged_outputs.json"))
        return {r["identifier"]: float(bool(r["llm_rst"]["goal_rst"]["all_goal_satisfied_graph"])) for r in log}
    log = json.load(open(R / f"{run}_eval/behavior/evaluate_results/subgoal_decomposition/log/merged.json"))
    return {k: float(bool(v.get("success"))) for k, v in log.items()}


def holdout():
    """Each arm on identifiers never sampled for the meta-HMM corpus (H) vs sampled (S),
    and the weighted - counting gap with its unpaired standard error."""
    out = {}
    for module, arms in HOLDOUT_RUNS.items():
        split = json.load(open(f"data/splits/{module}.json"))
        sets = {"H": set(split["test_identifiers"]), "S": set(split["train_identifiers"])}
        if module == "vh_sd":
            sets = {k: {i[len("scene_1_"):] if i.startswith("scene_1_") else i for i in v} for k, v in sets.items()}
        scores = {a: per_task(module, r) for a, r in arms.items()}
        res = {}
        for part, ids in sets.items():
            part_ids = set(ids)
            ids = [i for i in ids if i in scores["trained"] and i in scores["uniform"]]
            # each arm over the identifiers it has; the gap over those both counting arms have
            means = {a: 100 * st.mean(v for i, v in scores[a].items() if i in part_ids) for a in arms}
            w = [scores["trained"][i] for i in ids]
            u = [scores["uniform"][i] for i in ids]
            se = 100 * math.sqrt(st.pvariance(w) / len(ids) + st.pvariance(u) / len(ids))
            res[part] = (means, means["trained"] - means["uniform"], se, len(ids))
        (_, gh, sh, _), (_, gs, ss, _) = res["H"], res["S"]
        res["z"] = (gh - gs) / math.sqrt(sh ** 2 + ss ** 2) if sh + ss > 0 else 0.0
        out[module] = res
    return out


CALIBRATION = {c: f"calib_llama_c{c}" for c in (2, 3, 4)}
EXPECTED_CALIBRATION = {2: 9, 3: 8, 4: 9}  # successes on the 20 calibration tasks


def calibration():
    """Task successes of the lookahead calibration search: BEHAVIOR SD, the 20
    tasks with prompt index divisible by five. The evaluator scores all 100
    identifiers, so the summary rate is not a rate over the 20; count instead.
    Without a separate c = 2 run, c = 2 is read from the Meta-Ctrl cell, which
    decodes the same 20 tasks with the same settings (greedy, so identical plans)."""
    prompts = json.load(open("data/prompts/behavior/subgoal_decomposition/subgoal_decomposition_prompts.json"))
    tasks = {p["identifier"] for p in prompts[::5]}
    out = {}
    for c, run in CALIBRATION.items():
        f = R / f"{run}_eval/behavior/evaluate_results/subgoal_decomposition/log/merged.json"
        if not f.exists() and c == 2:
            f = R / f"{LLAMA[3]}_eval/behavior/evaluate_results/subgoal_decomposition/log/merged.json"
        if f.exists():
            out[c] = sum(bool(v.get("success")) for k, v in json.load(open(f)).items() if k in tasks)
    return out


def fmt(c):
    return "n/a".center(13) if c is None else f"{c[0]:5.1f} / {c[1]:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--results", default="results", help="directory of evaluator outputs")
    args = ap.parse_args()
    global R
    R = Path(args.results)
    if not R.is_dir():
        sys.exit(f"no {R}/ directory: run `scripts/download.sh outputs` first, or pass --results")
    tables = [("Main table (Meta-Ctrl row)", [("Llama 3 8B + Meta-Ctrl", LLAMA)]),
              ("Ablation (Llama 3 8B)", ABLATION), ("XGrammar appendix", XGRAMMAR)]
    bad = 0
    for title, rows in tables:
        print(f"\n{title}\n{'':38}{'VH AS':>15}{'BEH AS':>15}{'VH SD':>15}{'BEH SD':>15}")
        for label, runs in rows:
            cells = row(runs)
            line = f"{label:38}" + "".join(f"{fmt(c):>15}" for c in cells)
            if title.startswith("Main"):
                tasks = [c[0] for c in cells if c is not None]
                line += f"   avg task {sum(tasks) / len(tasks):.1f}" if len(tasks) == 4 else ""
            if args.check:
                for c, p in zip(cells, EXPECTED[label]):
                    if (c is None) != (p is None) or (c is not None and any(
                            not math.isclose(round(x, 1), y, abs_tol=0.051) for x, y in zip(c, p))):
                        bad += 1
                        line += f"  MISMATCH, expected {p}"
            print(line)
    print("\nHeld-out identifiers: tasks never sampled for the HMM corpus (H) vs sampled (S); per-task success in %\n"
          "(VH AS: execution); gap = Meta-Ctrl minus valid-plan counting, +- its unpaired standard error.")
    print(f"{'':8}{'n H / S':>11}{'Meta-Ctrl H':>13}{'S':>7}{'valid-plan H':>14}{'S':>7}{'gap H':>16}{'gap S':>16}")
    zs = []
    for module, res in holdout().items():
        (mh, gh, sh, nh), (ms, gs, ss, ns) = res["H"], res["S"]
        vals = ((mh["trained"], ms["trained"]), (mh["uniform"], ms["uniform"]), (gh, gs), (sh, ss))
        line = (f"{module:8}{f'{nh} / {ns}':>11}{mh['trained']:13.1f}{ms['trained']:7.1f}"
                f"{mh['uniform']:14.1f}{ms['uniform']:7.1f}{f'{gh:+.1f} +- {sh:.1f}':>16}{f'{gs:+.1f} +- {ss:.1f}':>16}")
        zs.append(abs(res["z"]))
        # a gap may be reported as the difference of rounded means: allow 0.1
        if args.check and any(not math.isclose(round(x, 1), y, abs_tol=0.051 + 0.1 * (j == 2))
                              for j, (v, p) in enumerate(zip(vals, EXPECTED_HOLDOUT[module])) for x, y in zip(v, p)):
            bad += 1
            line += f"  MISMATCH, expected {EXPECTED_HOLDOUT[module]}"
        print(line)
    print(f"largest |z| between the splits: {max(zs):.2f}")
    cal = calibration()
    line = "\nLookahead calibration, BEHAVIOR SD, 20 tasks (successes): " + \
        ", ".join(f"c = {c}: {n}/20" for c, n in cal.items())
    if args.check and cal != EXPECTED_CALIBRATION:
        bad += 1
        line += f"  MISMATCH, expected {EXPECTED_CALIBRATION}"
    print(line)
    if args.check:
        print("\nall cells match" if not bad else f"\n{bad} cells differ from the expected values")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

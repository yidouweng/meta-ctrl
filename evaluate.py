#!/usr/bin/env python3
"""Run the EAI evaluator (`eai-eval --mode evaluate_results`) on generations.

The input is one generation file or a folder of them. Each is copied to a
staging folder in the layout the evaluator expects,
<staging>/<dataset>/<module>/<name>_outputs.json, and evaluated into
<output-dir>/<dataset>/evaluate_results/<module>/<name>/.

Run it with the eai-eval environment's python (scripts/evaluate_run.sh does):

    python evaluate.py --module AS --dataset virtualhome --input outputs/llama_vh_as_trained.json \
        --output-dir results/llama_vh_as_trained_eval
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

MODULES = {"AS": "action_sequencing", "SD": "subgoal_decomposition"}


def stage_inputs(input_path: Path, dataset: str, module: str, keep: bool) -> Tuple[str, Optional[str]]:
    """(base path to pass to the evaluator, staging folder to delete or None)."""
    if input_path.is_dir():
        if (input_path / dataset / module).is_dir():  # already in the evaluator's layout
            return str(input_path.resolve()), None
        files: List[Path] = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix == ".json")
        if not files:
            sys.exit(f"no .json files in {input_path}")
    elif input_path.is_file():
        files = [input_path]
    else:
        sys.exit(f"input not found: {input_path}")

    staging = tempfile.mkdtemp(prefix="eai_eval_stage_")
    target = Path(staging) / dataset / module
    target.mkdir(parents=True)
    for f in files:
        name = f.name if f.stem.endswith("_outputs") else f"{f.stem}_outputs.json"
        shutil.copy2(f, target / name)
    if keep:
        print(f"[evaluate] staged in {target}")
    return staging, staging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", required=True, choices=sorted(MODULES) + sorted(MODULES.values()))
    ap.add_argument("--dataset", required=True, choices=["behavior", "virtualhome"])
    ap.add_argument("--input", required=True, help="a generation file, or a folder of them")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--keep-staging", action="store_true", help="keep the staging folder")
    args = ap.parse_args()

    if importlib.util.find_spec("eai_eval") is None:
        sys.exit("eai_eval is not importable: run this script with the eai-eval environment's python")
    module = MODULES.get(args.module, args.module)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base, staging = stage_inputs(Path(args.input).expanduser(), args.dataset, module, args.keep_staging)
    cmd = [sys.executable, "-m", "eai_eval.cli", "--dataset", args.dataset, "--mode", "evaluate_results",
           "--eval-type", module, "--llm-response-path", base, "--output-dir", str(output_dir),
           "--num-workers", str(args.num_workers)]
    print("[evaluate] " + " ".join(cmd), flush=True)
    try:
        code = subprocess.run(cmd).returncode
    finally:
        if staging and not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
    if code != 0:
        print(f"[evaluate] eai-eval exited with code {code}", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()

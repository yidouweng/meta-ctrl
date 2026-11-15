#!/usr/bin/env bash
# Evaluate one generation run with the EAI evaluator, under the names
# scripts/tables.py reads.
#
#   scripts/evaluate_run.sh <name> <module>
#     name    outputs/<name>.json (VirtualHome) or outputs/<name>/ (BEHAVIOR, prompt_*.json)
#     module  vh_as | vh_sd | beh_as | beh_sd
#
# VirtualHome is evaluated six times (PYTHONHASHSEED 0-5) into results/<name>_s<k>_eval,
# BEHAVIOR once into results/<name>_eval. EVAL_PY selects the evaluator's python;
# BEHAVIOR rendering uses the GPU in CUDA_VISIBLE_DEVICES (or GIBSON_DEVICE_ID).
set -euo pipefail
[ $# -eq 2 ] || { sed -n "4,7p" "$0"; exit 2; }
NAME=$1; MODULE=$2
EVAL_PY=${EVAL_PY:-python}
cd "$(dirname "$0")/.."
FAMILY=${MODULE#*_}
mkdir -p results logs
if [[ $MODULE == vh_* ]]; then
  for s in 0 1 2 3 4 5; do
    rm -rf results/${NAME}_s${s}_eval   # the evaluator merges into an existing directory
    PYTHONHASHSEED=$s $EVAL_PY evaluate.py --module ${FAMILY^^} --dataset virtualhome --input outputs/$NAME.json \
      --output-dir results/${NAME}_s${s}_eval --num-workers 8 > logs/eval_${NAME}_s$s.log 2>&1
  done
else
  # BEHAVIOR renders with iGibson (EGL), which defaults to the first GPU; render on
  # the job's own GPU instead (GIBSON_DEVICE_ID takes the device's minor number).
  if [ -z "${GIBSON_DEVICE_ID:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export GIBSON_DEVICE_ID=$(nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES%%,*}" | awk '/Minor Number/ {print $NF; exit}')
  fi
  $EVAL_PY - outputs/$NAME <<'EOF'
import glob, json, re, sys
d = sys.argv[1]
files = sorted(glob.glob(f"{d}/prompt_*.json"), key=lambda f: int(re.search(r"prompt_(\d+)", f).group(1)))
json.dump([r for f in files for r in json.load(open(f))], open(f"{d}/merged.json", "w"))
EOF
  rm -rf results/${NAME}_eval
  PYTHONHASHSEED=0 $EVAL_PY evaluate.py --module ${FAMILY^^} --dataset behavior --input outputs/$NAME/merged.json \
    --output-dir results/${NAME}_eval --num-workers 4 > logs/eval_${NAME}.log 2>&1
fi
echo "done: results/${NAME}*"

#!/usr/bin/env bash
# Lookahead calibration search (paper, Appendix I): Meta-Ctrl (trained meta-token HMM) on
# BEHAVIOR subgoal decomposition with lookahead exponent c, on the 20 tasks whose prompt
# index is divisible by five.
#
#   [LM=llama|gpt-oss] scripts/calibrate.sh [c ...]          (default: 2 3 4)
#
# Writes outputs/calib_<LM>_c<c>/ and evaluates it into results/calib_<LM>_c<c>_eval;
# scripts/tables.py counts the successes on the 20 tasks. PY, EVAL_PY and LM as in scripts/run.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
EVAL_PY=${EVAL_PY:-python}
LM=${LM:-llama}
case $LM in
  llama)   BASE=models/Meta-Llama-3-8B-Instruct; HMM=models/meta-hmm-sd ;;
  gpt-oss) BASE=models/gpt-oss-20b;              HMM=models/meta-hmm-sd-gpt-oss ;;
  *) echo "unknown LM $LM (llama or gpt-oss)"; exit 2 ;;
esac
TASKS=$(seq -s, 0 5 95)
mkdir -p outputs results logs
export PYTHONPATH=$PWD PYTHONHASHSEED=0
CS=("$@"); [ ${#CS[@]} -gt 0 ] || CS=(2 3 4)
for c in "${CS[@]}"; do
  NAME=calib_${LM}_c$c
  $PY generate_beh_sd.py --base_model $BASE --semantic_hmm_path $HMM \
    --ctrlg_scale $c --prompt_indices $TASKS --output_dir outputs/$NAME > logs/$NAME.log 2>&1
  EVAL_PY=$EVAL_PY scripts/evaluate_run.sh $NAME beh_sd
done

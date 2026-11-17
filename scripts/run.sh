#!/usr/bin/env bash
# Generate and evaluate one cell of the EAI experiments.
#
#   [LM=llama|gpt-oss] scripts/run.sh <module> <arm>
#     module  vh_as | vh_sd | beh_as | beh_sd
#     arm     trained   Meta-Ctrl (weighted counting; trained meta-token HMM)
#             uniform   valid-plan counting (uniform meta-token HMM)
#             mask      goal reachability
#             syntax    syntax mask only
#     LM      llama (Llama 3 8B Instruct, default) or gpt-oss (gpt-oss-20b; PY = the vLLM environment)
#
# Writes outputs/<LM>_<module>_<arm>[.json] and evaluates it with scripts/evaluate_run.sh
# into results/<LM>_<module>_<arm>_eval (VirtualHome: six runs, as _s<k>_eval).
# PY and EVAL_PY select the decoding and evaluation interpreters (REFERENCE.md, Environments).
# N=<k> decodes only the first k tasks, as a smoke test, under the name
# <LM>_<module>_<arm>_n<k> (tables.py ignores it). On VH AS use N >= 51: its evaluator
# needs a scored task with each goal type (state, relation, action), and the prompts are
# ordered by goal type.
set -euo pipefail
[ $# -eq 2 ] || { sed -n "3,10p" "$0"; exit 2; }
MODULE=$1; ARM=$2
PY=${PY:-python}
EVAL_PY=${EVAL_PY:-python}
cd "$(dirname "$0")/.."
if ! $EVAL_PY -c "import eai_eval" 2>/dev/null; then
  echo "EVAL_PY=$EVAL_PY cannot import eai_eval: set it to the evaluator's python (scripts/setup_evaluator.sh prints it)"
  exit 2
fi

LM=${LM:-llama}
FAMILY=${MODULE#*_}                       # as | sd
case $LM in
  llama)   BASE=models/Meta-Llama-3-8B-Instruct; TRAINED=models/meta-hmm-${FAMILY} ;;
  gpt-oss) BASE=models/gpt-oss-20b;              TRAINED=models/meta-hmm-${FAMILY}-gpt-oss ;;
  *) echo "unknown LM $LM (llama or gpt-oss)"; exit 2 ;;
esac
case $MODULE in
  vh_as|beh_as) UNIFORM=models/uniform-as ;;
  vh_sd)        UNIFORM=models/uniform-vh-sd ;;
  beh_sd)       UNIFORM=models/uniform-beh-sd ;;
  *) echo "unknown module $MODULE"; exit 2 ;;
esac
case $ARM in
  trained) HMM=$TRAINED; FLAGS="" ;;
  uniform) HMM=$UNIFORM; FLAGS="" ;;
  mask)    HMM=$UNIFORM; FLAGS="--beta_hard_mask" ;;
  syntax)  HMM=$UNIFORM; FLAGS="--syntax_only" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
# Meta-Ctrl on BEHAVIOR subgoal decomposition uses the calibrated lookahead c = 2 for both LMs
# (scripts/calibrate.sh); the ablation arms are uncalibrated (c = 1).
if [ "$MODULE" = beh_sd ] && [ "$ARM" = trained ]; then FLAGS="--ctrlg_scale 2"; fi

NAME=${LM}_${MODULE}_${ARM}
if [ "${N:-0}" -gt 0 ]; then FLAGS="$FLAGS --n_prompts $N"; NAME=${NAME}_n$N; fi
mkdir -p outputs results logs
export PYTHONPATH=$PWD
export PYTHONHASHSEED=0   # some automata iterate sets; a fixed seed makes decoding bit-reproducible
if [[ $MODULE == vh_* ]]; then
  $PY generate_${MODULE}.py --base_model $BASE --semantic_hmm_path $HMM $FLAGS --output outputs/$NAME.json \
    > logs/$NAME.log 2>&1
else
  $PY generate_${MODULE}.py --base_model $BASE --semantic_hmm_path $HMM $FLAGS --output_dir outputs/$NAME \
    > logs/$NAME.log 2>&1
fi
EVAL_PY=$EVAL_PY scripts/evaluate_run.sh $NAME $MODULE


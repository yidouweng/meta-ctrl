#!/usr/bin/env bash
# Generate and evaluate every Llama 3 8B run (16: 4 modules x 4 arms), then print the
# tables. Runs whose results exist are skipped.
#
#   PY=<decoding python> EVAL_PY=<evaluator python> scripts/run_all.sh
#
# N=<k> runs every cell on its first k tasks only (smoke test; tables are not rebuilt;
# k >= 51 for VH AS, see scripts/run.sh).
# The full run takes days on one GPU; REFERENCE.md lists measured times.
set -euo pipefail
cd "$(dirname "$0")/.."
for MODULE in vh_as beh_as vh_sd beh_sd; do
  for ARM in syntax mask uniform trained; do
    NAME=llama_${MODULE}_${ARM}${N:+_n$N}
    if [ -d "results/${NAME}_eval" ] || [ -d "results/${NAME}_s5_eval" ]; then echo "skip $NAME (results exist)"; continue; fi
    echo "$(date +%H:%M) $NAME"
    scripts/run.sh "$MODULE" "$ARM"
  done
done
[ -n "${N:-}" ] || ${PY:-python} scripts/tables.py --check

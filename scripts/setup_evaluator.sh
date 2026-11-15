#!/usr/bin/env bash
# Set up the EAI evaluator: the conda environment from envs/eai-eval.yml plus
# iGibson (needed for BEHAVIOR) from the EAI fork, with two build patches that do
# not change the evaluation.
#
#   scripts/setup_evaluator.sh [IGIBSON_DIR]      (default: third_party/iGibson)
#
# Needs conda, git, a C++ compiler, cmake, the system package libegl1 and ~30 GB of disk
# for the iGibson scenes (IGIBSON_ACCEPT_TERMS=1 accepts the dataset terms non-interactively).
# EAI_ENV names the conda environment (default: eai-eval). Prints the
# evaluator's python at the end; pass it to scripts/run.sh as EVAL_PY.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-third_party/iGibson}
ENV=${EAI_ENV:-eai-eval}
IGIBSON_REPO=https://github.com/embodied-agent-interface/iGibson.git
IGIBSON_COMMIT=a4f6021c
PYBIND11_TAG=v2.10.4

ldconfig -p | grep -q 'libEGL.so.1' || { echo "libEGL.so.1 not found: install libegl1 (e.g. apt install libegl1)"; exit 1; }
command -v conda > /dev/null || { echo "conda not found"; exit 1; }

if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda env create -n "$ENV" -f envs/eai-eval.yml
fi
PY=$(conda run -n "$ENV" python -c 'import sys; print(sys.executable)')

if [ ! -d "$DIR" ]; then
  git clone "$IGIBSON_REPO" "$DIR"
  git -C "$DIR" checkout "$IGIBSON_COMMIT"
  git -C "$DIR" submodule update --init --recursive
  git -C "$DIR" apply "$PWD/envs/igibson_egl_iostream.patch"
  git -C "$DIR/igibson/render/pybind11" fetch --tags --quiet
  git -C "$DIR/igibson/render/pybind11" checkout "$PYBIND11_TAG"
fi
"$PY" -m pip install --no-deps -e "$DIR"

# BEHAVIOR evaluation loads the iGibson scenes (22 GB download, about 27 GB on disk);
# without them every BEHAVIOR task scores 0. Their use is subject to the iGibson dataset terms.
DATA=$DIR/igibson/data
if [ ! -d "$DATA/ig_dataset/scenes" ]; then
  TERMS=http://svl.stanford.edu/igibson/assets/GDS_agreement.pdf
  if [ "${IGIBSON_ACCEPT_TERMS:-}" != 1 ]; then
    read -r -p "Download the iGibson dataset, accepting its terms ($TERMS)? [y/N] " ok
    [ "$ok" = y ] || { echo "skipped: BEHAVIOR evaluation needs $DATA/ig_dataset (rerun to download)"; exit 1; }
  fi
  mkdir -p "$DATA"
  GS=https://storage.googleapis.com/gibson_scenes
  curl -sSfL $GS/igibson.key -o "$DATA/igibson.key"
  curl -sSfL $GS/assets_igibson.tar.gz | tar xz -C "$DATA"
  curl -sSfL $GS/ig_dataset.tar.gz | tar xz -C "$DATA"
fi

"$PY" -c "import igibson, behavior_eval, virtualhome_eval" > /dev/null
echo "evaluator ready: EVAL_PY=$PY"

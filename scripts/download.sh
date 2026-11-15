#!/usr/bin/env bash
# Download the artifacts (REFERENCE.md, "Models and artifacts"), archives in the
# Hugging Face dataset repo $REPO:
#
#   scripts/download.sh outputs   outputs/, results/   generations + evaluator outputs
#   scripts/download.sh hmms      models/              the meta-token HMMs
#   scripts/download.sh lm        models/              Llama 3 8B Instruct
#   scripts/download.sh samples   work/samples/        the Llama samples, 0.12 GB
#   scripts/download.sh corpora   work/corpora/        the training corpora, 3.4 GB
#
# Llama 3 is gated on Hugging Face: accept its license and run
# `huggingface-cli login` (or set HF_TOKEN) first.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-gwenweng/meta-ctrl-eai}

fetch() {  # fetch <archive in $REPO> <directory to extract into>
  mkdir -p "$2"
  f=$(python -c "import sys; from huggingface_hub import hf_hub_download as d; \
print(d(sys.argv[1], sys.argv[2], repo_type='dataset'))" "$REPO" "$1")
  tar xf "$f" -C "$2"
}

case ${1:-} in
  outputs) fetch outputs.tar.gz . ;;
  hmms)    fetch models.tar.gz . ;;
  lm)      python -c "from huggingface_hub import snapshot_download as s; \
s('meta-llama/Meta-Llama-3-8B-Instruct', local_dir='models/Meta-Llama-3-8B-Instruct', \
allow_patterns=['*.json', '*.safetensors'])" ;;
  samples) fetch train_data/llama_samples.tar work/samples ;;
  corpora) fetch train_data/llama_corpus_as.tar work/corpora
           fetch train_data/llama_corpus_sd.tar work/corpora ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac

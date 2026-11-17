#!/usr/bin/env bash
# Download the artifacts (REFERENCE.md, "Models and artifacts"), archives in the
# Hugging Face dataset repo $REPO:
#
#   scripts/download.sh outputs   outputs/, results/   generations + evaluator outputs (both LMs)
#   scripts/download.sh hmms      models/              the meta-token HMMs (both LMs)
#   scripts/download.sh lm        models/              the base LM
#   scripts/download.sh samples   work/samples/        the LM's HMM training samples
#   scripts/download.sh corpora   work/corpora/        the Llama training corpora, 3.4 GB
#
# LM=llama (default) or gpt-oss selects the base LM for `lm` and `samples`. Llama 3 is gated
# on Hugging Face: accept its license and run `huggingface-cli login` (or set HF_TOKEN) first.
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
  lm)      case ${LM:-llama} in
             llama)   ID=meta-llama/Meta-Llama-3-8B-Instruct; DIR=models/Meta-Llama-3-8B-Instruct ;;
             gpt-oss) ID=openai/gpt-oss-20b;                  DIR=models/gpt-oss-20b ;;
             *) echo "unknown LM ${LM}"; exit 2 ;;
           esac
           python -c "import sys; from huggingface_hub import snapshot_download as s; \
s(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['*.json', '*.safetensors', '*.jinja'], \
ignore_patterns=['original/*', 'metal/*'])" "$ID" "$DIR" ;;
  samples) fetch train_data/${LM:-llama}_samples.tar work/samples ;;
  corpora) fetch train_data/llama_corpus_as.tar work/corpora
           fetch train_data/llama_corpus_sd.tar work/corpora ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac

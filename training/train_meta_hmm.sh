#!/usr/bin/env bash
# Train one meta-token HMM from scratch: sample -> meta-token corpus -> balance -> train.
# Writes models/<name>-retrained/ (the released checkpoint's name, suffixed) in the
# layout the decoders read.
#
#   training/train_meta_hmm.sh <as|sd>
#
# Interpreters (REFERENCE.md, Environments): SAMPLE_PY has vLLM; PY builds the corpus;
# TRAIN_PY has JAX. SAMPLES=<dir> skips sampling and builds the corpus from
# existing samples, <dir>/{behavior,virtualhome}_<as|sd> (scripts/download.sh samples).
set -euo pipefail
FAM=${1:-}
cd "$(dirname "$0")/.."
SAMPLE_PY=${SAMPLE_PY:-python}; PY=${PY:-python}; TRAIN_PY=${TRAIN_PY:-$PY}
WORK=${WORK:-work/meta_hmm_${FAM}}
MODEL=models/Meta-Llama-3-8B-Instruct
MODULE=${FAM^^}
case $FAM in
  as) TASK=action_sequencing;     VOCAB=data/vocab/hmm_vocab_as_actions_only_full_schema.json; CMAX=96 ;;
  sd) TASK=subgoal_decomposition; VOCAB=data/vocab/hmm_vocab_sd_predicates_union.json;         CMAX=384 ;;
  *) echo "usage: $0 <as|sd>"; exit 2 ;;
esac

if [ -z "${SAMPLES:-}" ]; then
  SAMPLES=$WORK/samples
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  for BENCH in behavior virtualHome; do
    B=$(echo $BENCH | tr A-Z a-z)
    case $B-$FAM in virtualhome-*) MML=4096 ;; behavior-as) MML=4608 ;; behavior-sd) MML=4096 ;; esac
    # 1M samples per benchmark, 200 tokens each, T = 1, top-p = 1 (paper, Appendix E). BEHAVIOR SD
    # draws 465,920 (13 rounds of 70 prompts x 512): about half pass its filter, which is enough to
    # balance against the VirtualHome half.
    N=1000000; [ $B-$FAM = behavior-sd ] && N=465920
    $SAMPLE_PY training/sample.py --model $MODEL --prompt_json data/prompts/augment/$BENCH/$TASK/prompt_augmented.json \
      --dataset $B --module $MODULE --output_dir $SAMPLES/${B}_$FAM --total_samples $N --continuation_length 200 \
      --temperature 1.0 --top_p 1.0 --seed 42 --batch_size 512 --prompt_batch_size 16 --max_num_seqs 2048 \
      --gpu_memory_utilization 0.94 --max_model_len $MML
  done
fi

$PY training/build_dataset.py --family $FAM --base_model $MODEL --vocab_json $VOCAB --c_max $CMAX \
  --emb_batch_size 8 --emb_dtype float16 \
  --behavior_samples $SAMPLES/behavior_$FAM --behavior_prompts data/prompts/augment/behavior/$TASK/prompt_augmented.json \
  --virtualhome_samples $SAMPLES/virtualhome_$FAM --virtualhome_prompts data/prompts/augment/virtualHome/$TASK/prompt_augmented.json \
  --out_root $WORK/corpus
DIRS=$($PY training/balance.py $WORK/corpus/behavior $WORK/corpus/virtualhome --seed 42 | tail -2)
D1=$(echo "$DIRS" | head -1); D2=$(echo "$DIRS" | tail -1)

OUT=models/meta-hmm-${FAM}-retrained
$TRAIN_PY training/train.py --data_dir $D1 --data_dir $D2 --vocab_json $VOCAB --save_dir $OUT \
  --hidden_size 128 --hmm_lr 0.1 --nn_lr 0.001 --batch_size 256 --epochs 20 --patience 5 --seed 42
cp $VOCAB $OUT/
echo "done: $OUT"

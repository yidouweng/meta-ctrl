#!/usr/bin/env python3
"""Sample LM answers to the training prompts with vLLM.

    python training/sample.py --model M --prompt_json P --dataset behavior|virtualhome --module AS|SD \
        --output_dir OUT [--total_samples 1000000 --continuation_length 200 ...]

Prompts are sorted by identifier and split 70/30 with default_rng(seed); only
the 70% (the training prompts) are sampled, equally often. Prompts are
built by ctrlg.prompts, as the decoder's are. Every generate() call gets its
own seed derived from --seed, so repeated calls draw new samples.

Writes OUT/shards/*.npz (continuations padded to --continuation_length,
prompt_indices into the training prompts, pad_id), OUT/prompts.npz (the
tokenized training prompts, left-padded), OUT/split_info.json and
OUT/metadata.json. A rerun resumes after the last complete call.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctrlg.prompts import chat_text, system_prompt  # noqa: E402


def split_prompts(prompts, train_fraction: float, seed: int):
    idx = np.arange(len(prompts))
    np.random.default_rng(seed).shuffle(idx)
    n_train = max(1, min(int(len(prompts) * train_fraction), len(prompts) - 1))
    return [prompts[i] for i in idx[:n_train]], [prompts[i] for i in idx[n_train:]]


def call_seed(seed: int, call_idx: int) -> int:
    """Well-separated deterministic seed of the call_idx-th generate() call."""
    h = (seed * 1_000_003) ^ ((call_idx + 1) * 2_654_435_761)
    return (h * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) % (2 ** 31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt_json", required=True)
    ap.add_argument("--dataset", choices=["behavior", "virtualhome"], required=True)
    ap.add_argument("--module", choices=["AS", "SD"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--total_samples", type=int, default=1_000_000)
    ap.add_argument("--continuation_length", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_fraction", type=float, default=0.70)
    ap.add_argument("--batch_size", type=int, default=512, help="samples per prompt per generate() call")
    ap.add_argument("--prompt_batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="auto", help="vLLM dtype (gpt-oss: bfloat16)")
    ap.add_argument("--reasoning_effort", default=None, help="gpt-oss chat template: low|medium|high")
    ap.add_argument("--date", default=None,
                    help="gpt-oss: the 'Current date:' the chat template stamps in (default: today)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.94)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--max_num_seqs", type=int, default=None)
    ap.add_argument("--shard_size", type=int, default=10_000)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    prompts = sorted(json.load(open(args.prompt_json)), key=lambda p: str(p["identifier"]))
    train, test = split_prompts(prompts, args.train_fraction, args.seed)
    system = system_prompt(args.dataset, args.module)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = [chat_text(tok, args.dataset, args.module, p["llm_prompt"],
                       reasoning_effort=args.reasoning_effort, date=args.date) for p in train]

    out = args.output_dir
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    split_info = {"seed": args.seed, "train_fraction": args.train_fraction,
                  "num_train_prompts": len(train), "num_test_prompts": len(test),
                  "train_identifiers": [str(p["identifier"]) for p in train],
                  "test_identifiers": [str(p["identifier"]) for p in test], "identifier_field": "identifier"}
    json.dump(split_info, open(os.path.join(out, "split_info.json"), "w"), indent=2)
    # The prompt tokens the meta-token HMM is conditioned on (tokenized as for sampling,
    # special tokens included), left-padded.
    ids = [tok(t, add_special_tokens=True, return_attention_mask=False)["input_ids"] for t in texts]
    width = max(len(x) for x in ids)
    arr = np.full((len(ids), width), tok.pad_token_id, dtype=np.int32)
    for i, x in enumerate(ids):
        arr[i, -len(x):] = x
    np.savez_compressed(os.path.join(out, "prompts.npz"), prompts=arr,
                        identifiers=np.array(split_info["train_identifiers"], dtype=object), t_max=width,
                        pad_id=tok.pad_token_id, vocab_size=tok.vocab_size)
    json.dump({"model": args.model, "dataset": args.dataset, "module": args.module, "system_prompt": system,
               "reasoning_effort": args.reasoning_effort, "total_samples": args.total_samples,
               "continuation_length": args.continuation_length, "temperature": args.temperature,
               "top_p": args.top_p, "seed": args.seed, "train_fraction": args.train_fraction,
               "pad_id": tok.pad_token_id, "eos_id": tok.eos_token_id},
              open(os.path.join(out, "metadata.json"), "w"), indent=2)

    per_prompt = max(1, args.total_samples // len(train))
    n_calls = (per_prompt + args.batch_size - 1) // args.batch_size
    prefix = f"{args.dataset}_{args.module}"
    done = sorted(glob.glob(os.path.join(out, "shards", f"{prefix}_shard_*.npz")))
    shard_counter = len(done)
    # Resume after the last complete call: every call draws batch_size samples per prompt
    # with its own seed, and the shards of a complete call are all on disk.
    done_rows = sum(int(np.load(f)["continuations"].shape[0]) for f in done)
    if done_rows >= per_prompt * len(train):
        print("sampling already complete")
        return
    first_call = done_rows // (len(train) * args.batch_size)
    if done_rows != first_call * len(train) * args.batch_size:
        raise SystemExit(f"{out}/shards ends inside a call ({done_rows} rows); delete the shards of the "
                         f"incomplete call or start over")

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
              **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
              seed=args.seed, trust_remote_code=True, disable_custom_all_reduce=True, enable_prefix_caching=True)
    vtok = llm.get_tokenizer()
    pad_id = vtok.pad_token_id if vtok.pad_token_id is not None else vtok.eos_token_id
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.continuation_length,
                            n=args.batch_size, seed=call_seed(args.seed, 0), stop_token_ids=[vtok.eos_token_id])

    buf_c, buf_p = [], []

    def flush():
        nonlocal shard_counter, buf_c, buf_p
        if buf_c:
            np.savez_compressed(os.path.join(out, "shards", f"{prefix}_shard_{shard_counter:05d}.npz"),
                                continuations=np.stack(buf_c).astype(np.int32),
                                prompt_indices=np.asarray(buf_p, dtype=np.int32), pad_id=int(pad_id))
            buf_c, buf_p, shard_counter = [], [], shard_counter + 1

    L = args.continuation_length
    for call_idx in range(first_call, n_calls):
        params.n = min(args.batch_size, per_prompt - call_idx * args.batch_size)
        params.seed = call_seed(args.seed, call_idx)
        for s in range(0, len(train), args.prompt_batch_size):
            for k, res in enumerate(llm.generate(texts[s:s + args.prompt_batch_size], params, use_tqdm=False)):
                for comp in res.outputs:
                    t = np.asarray(comp.token_ids, dtype=np.int32)[:L]
                    buf_c.append(np.pad(t, (0, L - len(t)), constant_values=int(pad_id)))
                    buf_p.append(s + k)
            if len(buf_c) >= args.shard_size:
                flush()
        flush()  # a finished call is entirely on disk
        print(f"call {call_idx + 1}/{n_calls}: {shard_counter} shards", flush=True)


if __name__ == "__main__":
    main()

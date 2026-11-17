#!/usr/bin/env python3
"""Keep the final channel of gpt-oss samples.

    python training/final_channel.py SAMPLES_DIR OUT_DIR [--tokenizer PATH]

gpt-oss writes an analysis channel before its answer. Each sampled
continuation is cut to the tokens after `<|channel|>final<|message|>` up to and
including `<|return|>`; samples without either marker are dropped. OUT_DIR has
the layout of a sample.py output (shards/, prompts.npz, split_info.json,
metadata.json), so build_dataset.py reads it unchanged.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
from transformers import AutoTokenizer

SHARD_ROWS = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tokenizer", default="models/gpt-oss-20b")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ch, msg, ret = (tok.convert_tokens_to_ids(t) for t in ("<|channel|>", "<|message|>", "<|return|>"))
    (final,) = tok.encode("final", add_special_tokens=False)
    os.makedirs(os.path.join(a.dst, "shards"), exist_ok=True)

    rows, prompt_idx, n_out, dropped = [], [], 0, {"no_signpost": 0, "no_return": 0}
    lens = []

    def flush():
        nonlocal rows, prompt_idx, n_out
        if not rows:
            return
        width = max(len(r) for r in rows)
        arr = np.full((len(rows), width), pad, dtype=np.int32)
        for i, r in enumerate(rows):
            arr[i, :len(r)] = r
        np.savez_compressed(os.path.join(a.dst, "shards", f"final_shard_{n_out:05d}.npz"), continuations=arr,
                            prompt_indices=np.asarray(prompt_idx, dtype=np.int32), pad_id=pad)
        rows, prompt_idx, n_out = [], [], n_out + 1

    for path in sorted(glob.glob(os.path.join(a.src, "shards", "*.npz"))):
        d = np.load(path)
        pad = int(d["pad_id"])
        for r, p in zip(d["continuations"], d["prompt_indices"]):
            # As in the paper's corpus: a row is cut to its number of non-pad tokens, then
            # pads are removed (this drops the rare sample with a pad token mid-answer).
            r = r[:int((r != pad).sum())]
            r = r[r != pad].tolist()
            pos = next((j for j in range(len(r) - 2) if r[j] == ch and r[j + 1] == final and r[j + 2] == msg), None)
            if pos is None:
                dropped["no_signpost"] += 1
                continue
            body = r[pos + 3:]
            if not body or body[-1] != ret:
                dropped["no_return"] += 1
                continue
            rows.append(body)
            prompt_idx.append(int(p))
            lens.append(len(body))
            if len(rows) == SHARD_ROWS:
                flush()
    flush()

    for f in ("prompts.npz", "split_info.json"):
        shutil.copy(os.path.join(a.src, f), a.dst)
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    meta["final_only"] = {"kept": len(lens), "dropped": dropped, "len_max": max(lens, default=0)}
    json.dump(meta, open(os.path.join(a.dst, "metadata.json"), "w"), indent=2)
    print(json.dumps(meta["final_only"]))


if __name__ == "__main__":
    main()

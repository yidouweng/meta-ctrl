#!/usr/bin/env python3
"""Balance the BEHAVIOR and VirtualHome halves of a corpus.

    python training/balance.py DIR_A DIR_B [--seed 42]

Subsamples the larger half to the size of the smaller one (uniform, without
replacement, sorted; default_rng(seed)) into <larger>_sub<N>. Prints the two
directories to train on, smaller first, as the last two lines.
"""
import argparse
import json
import os
import shutil

import numpy as np


def n_samples(d):
    return int(json.load(open(os.path.join(d, "dataset_info.json")))["num_samples"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    na, nb = n_samples(a.dir_a), n_samples(a.dir_b)
    small, big = (a.dir_a, a.dir_b) if na <= nb else (a.dir_b, a.dir_a)
    k, nbig = min(na, nb), max(na, nb)
    if k == nbig:
        print(small)
        print(big)
        return

    dst = f"{big.rstrip('/')}_sub{k}"
    os.makedirs(dst, exist_ok=True)
    sel = np.sort(np.random.default_rng(a.seed).choice(nbig, size=k, replace=False))
    seq = np.load(os.path.join(big, "semantic_continuations.npy"), mmap_mode="r")[:nbig]
    mp = np.load(os.path.join(big, "mapping.npz"))["continuation_to_prompt_idx"][:nbig]
    out = np.lib.format.open_memmap(os.path.join(dst, "semantic_continuations.npy"), mode="w+",
                                    dtype=np.int32, shape=(k, seq.shape[1]))
    out[:] = seq[sel]
    out.flush()
    np.savez_compressed(os.path.join(dst, "mapping.npz"), continuation_to_prompt_idx=mp[sel].astype(np.int32))
    for f in ["prompts.npz", "split_info.json", "prompt_end_embedding.npz", "source_metadata.json"]:
        if os.path.exists(os.path.join(big, f)):
            shutil.copy2(os.path.join(big, f), os.path.join(dst, f))
    info = json.load(open(os.path.join(big, "dataset_info.json")))
    info.update(num_samples=k, capacity=k, subsampled_from=big, subsample_seed=a.seed, subsample_of_N=nbig)
    json.dump(info, open(os.path.join(dst, "dataset_info.json"), "w"), indent=2)
    print(f"[balance] {big}: {nbig} -> {k} rows in {dst}")
    print(small)
    print(dst)


if __name__ == "__main__":
    main()

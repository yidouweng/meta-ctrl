#!/usr/bin/env python3
"""Train a meta-token HMM.

    python training/train.py --data_dir D1 --data_dir D2 --vocab_json V --save_dir OUT

Model: hidden states z_1..z_N (H of them), meta-tokens y_1..y_N,
    p(y | prompt) = sum_z p(z_1 | e) prod_n p(z_n | z_{n-1}) prod_n p(y_n | z_n),
where e is the LM's hidden state at the end of the prompt and
p(z_1 | e) = softmax(W e + b). The loss is the negative log-likelihood per
meta-token (EOS included); the HMM (JAX) and the projection (PyTorch) are
trained jointly with AdamW. Each epoch ends with a full pass over the dev
split (5% of the training prompts' samples); the best epoch is kept, and
training stops after `patience` epochs without improvement.

Writes OUT/hmm_params.pt (log_alpha, log_beta, log_gamma; unnormalized) and
OUT/nn_checkpoint.pt (proj.weight, proj.bias) of the best epoch, the format
ctrlg.models.load_semantic_hmm reads.
"""
import argparse
import json
import os
import sys
from functools import partial
from typing import List

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# --deterministic must act before JAX and PyTorch initialize the GPU.
DETERMINISTIC = "--deterministic" in sys.argv
if DETERMINISTIC:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import torch  # noqa: E402
from jax import jit, value_and_grad, vmap  # noqa: E402
from jax.nn import logsumexp  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402


# ---------------------------------------------------------------- HMM (log space)
# params = (alpha, beta, gamma): alpha (H, H) transitions, beta (H, V) emissions,
# gamma (1, H) unused initial weights (kept for the checkpoint format); rows are
# normalized with log-softmax wherever they are used.

def init_hmm(hidden_size: int, vocab_size: int, key):
    """Dirichlet(1) rows, as the cookbook's dense / categorical layers."""
    k_alpha, k_beta, k_gamma = jax.random.split(key, 3)
    alpha = jnp.log(jax.random.dirichlet(k_alpha, jnp.ones(hidden_size), shape=(hidden_size,)))
    rows = []
    for i in range(hidden_size // 1000 + 1):
        k_beta, sub = jax.random.split(k_beta)
        rows.append(jnp.log(jax.random.dirichlet(sub, jnp.ones((vocab_size,)),
                                                 shape=(min(1000, hidden_size - i * 1000),))))
    beta = jnp.concatenate(rows, axis=0)
    gamma = jnp.log(jax.random.dirichlet(k_gamma, jnp.ones(hidden_size), shape=(1,)))
    return alpha, beta, gamma


def _stable_mvm(log_w, log_x):
    """log(W @ x) from log W and log x."""
    x_max = jnp.max(log_x)
    w_max = jnp.max(log_w, axis=1, keepdims=True)
    z = jnp.exp(log_w - w_max) @ jnp.exp(log_x - x_max)
    return jnp.log(z) + x_max + jnp.squeeze(w_max, axis=-1)


def backward_log_z(params, sequence, mask):
    """log p(y_{1:n} | z_1) for every z_1; positions with mask False are skipped."""
    alpha, beta, _ = params
    emit = vmap(lambda y: beta[..., y] - logsumexp(beta, axis=-1))(sequence)  # (n, H)
    n = len(sequence)

    def step(i, state):
        trans = alpha - logsumexp(alpha, axis=-1, keepdims=True)
        new = _stable_mvm(trans, jnp.ravel(state + emit[n - 1 - i]))
        return jnp.where(mask[n - 1 - i], new, state)

    return jax.lax.fori_loop(0, n, step, jnp.zeros((alpha.shape[0],))).ravel()


@jit
def nll_per_token(params, prior_log_z, sequences, masks, lengths):
    """Mean over the batch of -log p(y | prompt) / len(y)."""
    back = vmap(backward_log_z, in_axes=(None, 0, 0))(params, sequences, masks)
    ll = logsumexp(prior_log_z + back, axis=1)
    return jnp.mean(-ll / jnp.maximum(lengths, 1.0))


class PromptProjection(nn.Module):
    """p(z_1 | e) = softmax(W e + b)."""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(embeddings), dim=-1)


# ---------------------------------------------------------------- data

def load_data(data_dirs: List[str], vocab_json: str, seed: int, train_fraction: float = 0.95) -> dict:
    """Samples of the training prompts (split_info.json), split per dataset into
    train / dev with default_rng(seed); global indices are offsets into the
    concatenation of the datasets in the given order."""
    vocab = json.load(open(vocab_json))["full_vocab"]
    eos_id = vocab.index("EOS")
    datasets = []
    for d in data_dirs:
        emb = np.load(os.path.join(d, "prompt_end_embedding.npz"), allow_pickle=True)
        seqs = np.load(os.path.join(d, "semantic_continuations.npy"), mmap_mode="r")
        prompt_idx = np.load(os.path.join(d, "mapping.npz"))["continuation_to_prompt_idx"].astype(np.int64)
        identifiers = np.load(os.path.join(d, "prompts.npz"), allow_pickle=True)["identifiers"]
        train_ids = set(str(x) for x in json.load(open(os.path.join(d, "split_info.json")))["train_identifiers"])
        n = min(int(seqs.shape[0]), int(prompt_idx.shape[0]))
        prompt_idx = prompt_idx[:n]
        is_train = np.zeros(len(identifiers), dtype=bool)
        is_train[[i for i, x in enumerate(identifiers) if str(x) in train_ids]] = True
        idx = np.where(is_train[prompt_idx])[0].astype(np.int64)
        np.random.default_rng(seed).shuffle(idx)
        n_dev = max(1, int(len(idx) * (1 - train_fraction))) if len(idx) > 1 else 0
        info_path = os.path.join(d, "dataset_info.json")
        max_len = json.load(open(info_path)).get("max_semantic_len_observed", 0) if os.path.exists(info_path) else 0
        datasets.append(dict(prompt_embeddings=emb["prompt_embeddings"], prompt_indices=prompt_idx,
                             sequences=seqs[:n].astype(np.int32, copy=False), dev=idx[:n_dev], train=idx[n_dev:],
                             c_max=min(int(max_len), int(seqs.shape[1])) if max_len else int(seqs.shape[1])))
        print(f"{d}: {len(idx) - n_dev} train, {n_dev} dev samples")
    offsets = np.cumsum([0] + [len(ds["sequences"]) for ds in datasets])
    return dict(datasets=datasets, offsets=offsets, eos_id=eos_id, vocab_size=len(vocab),
                embed_dim=int(datasets[0]["prompt_embeddings"].shape[-1]),
                c_max=max(ds["c_max"] for ds in datasets),
                train=np.concatenate([o + ds["train"] for o, ds in zip(offsets, datasets)]),
                dev=np.concatenate([o + ds["dev"] for o, ds in zip(offsets, datasets)]))


def make_batch(data: dict, indices: np.ndarray) -> dict:
    """Prompt embeddings and meta-token rows cut after the first EOS (an EOS is
    forced at c_max - 1 if a row has none)."""
    c_max, eos_id = data["c_max"], data["eos_id"]
    ds_ids = np.searchsorted(data["offsets"][1:], indices, side="right")
    local = indices - data["offsets"][ds_ids]
    emb = np.zeros((len(indices), data["embed_dim"]), dtype=np.float32)
    seqs = np.full((len(indices), c_max), eos_id, dtype=np.int32)
    masks = np.zeros((len(indices), c_max), dtype=bool)
    lengths = np.zeros((len(indices),), dtype=np.int32)
    for k, ds in enumerate(data["datasets"]):
        sel = ds_ids == k
        if sel.any():
            emb[sel] = np.asarray(ds["prompt_embeddings"][ds["prompt_indices"][local[sel]]], dtype=np.float32)
    for j, (k, li) in enumerate(zip(ds_ids, local)):
        row = data["datasets"][k]["sequences"][li][:c_max]
        eos = np.where(row == eos_id)[0]
        if eos.size:
            n = int(eos[0]) + 1
        else:
            row = row.copy()
            row[c_max - 1] = eos_id
            n = c_max
        seqs[j, :n] = row[:n]
        masks[j, :n] = True
        lengths[j] = n
    return dict(embeddings=emb, sequences=seqs, masks=masks, lengths=lengths)


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", action="append", required=True,
                    help="build_dataset.py output; give the smaller (balanced) half first")
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--hmm_lr", type=float, default=0.1)
    ap.add_argument("--nn_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic GPU kernels (XLA and PyTorch): bitwise-reproducible runs, about 2.5x slower")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir, args.vocab_json, args.seed)
    print(f"c_max={data['c_max']}  train={len(data['train'])}  dev={len(data['dev'])}")

    params = init_hmm(args.hidden_size, data["vocab_size"], jax.random.PRNGKey(args.seed))
    optimizer = optax.MultiSteps(optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.hmm_lr)),
                                 every_k_schedule=1)
    opt_state = optimizer.init(params)
    proj = PromptProjection(data["embed_dim"], args.hidden_size).to(device)
    proj_opt = torch.optim.AdamW(proj.parameters(), lr=args.nn_lr, weight_decay=0.01)

    def batch_loss(params, batch):
        with torch.no_grad():
            prior = jnp.array(proj(torch.from_numpy(batch["embeddings"]).to(device)).cpu().numpy())
        return float(nll_per_token(params, prior, jnp.array(batch["sequences"]), jnp.array(batch["masks"]),
                                   jnp.array(batch["lengths"])))

    def dev_loss(params):
        proj.eval()
        total, dev = 0.0, data["dev"]
        for s in range(0, len(dev), args.batch_size):
            part = dev[s:s + args.batch_size]
            total += batch_loss(params, make_batch(data, part)) * len(part)
        proj.train()
        return total / max(len(dev), 1)

    def save(params):
        os.makedirs(args.save_dir, exist_ok=True)
        alpha, beta, gamma = params
        torch.save({"log_alpha": torch.from_numpy(np.array(alpha)).float(),
                    "log_beta": torch.from_numpy(np.array(beta)).float(),
                    "log_gamma": torch.from_numpy(np.array(gamma)).float(),
                    "hidden_size": args.hidden_size, "vocab_size": data["vocab_size"]},
                   os.path.join(args.save_dir, "hmm_params.pt"))
        torch.save(proj.state_dict(), os.path.join(args.save_dir, "nn_checkpoint.pt"))

    rng = np.random.default_rng(args.seed)
    train = data["train"]
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        total, n_batches = 0.0, (len(train) + args.batch_size - 1) // args.batch_size
        for b in range(n_batches):
            batch = make_batch(data, train[b * args.batch_size:(b + 1) * args.batch_size])
            proj_opt.zero_grad()
            prior_t = proj(torch.from_numpy(batch["embeddings"]).to(device))
            prior = jnp.array(prior_t.detach().cpu().numpy())
            loss, (g_params, g_prior) = value_and_grad(
                partial(nll_per_token, sequences=jnp.array(batch["sequences"]), masks=jnp.array(batch["masks"]),
                        lengths=jnp.array(batch["lengths"])), argnums=(0, 1))(params, prior)
            updates, opt_state = optimizer.update(g_params, opt_state, params)
            params = optax.apply_updates(params, updates)
            prior_t.backward(torch.from_numpy(np.array(g_prior)).to(device))
            proj_opt.step()
            total += float(loss)
        d = dev_loss(params)
        print(f"epoch {epoch}  train {total / max(n_batches, 1):.4f}  dev {d:.4f}", flush=True)
        if d < best:
            best, stale = d, 0
            save(params)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"best dev {best:.4f} -> {args.save_dir}")


if __name__ == "__main__":
    main()

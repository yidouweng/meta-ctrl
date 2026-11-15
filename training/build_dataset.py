#!/usr/bin/env python3
"""Turn sampled continuations into meta-token training data.

    python training/build_dataset.py --family as|sd --base_model M --vocab_json V --c_max N \
        --behavior_samples DIR --behavior_prompts JSON \
        --virtualhome_samples DIR --virtualhome_prompts JSON --out_root OUT

Each samples DIR is a sample.py output (shards/, prompts.npz, split_info.json).
Writes OUT/behavior and OUT/virtualhome, each with
    semantic_continuations.npy  (rows, c_max) int32, first num_samples rows valid
    mapping.npz                 continuation_to_prompt_idx
    prompts.npz, split_info.json, source_metadata.json   copied from DIR
    prompt_end_embedding.npz    the LM's last hidden state at the end of each prompt
    dataset_info.json           counts; num_samples, max_semantic_len_observed

A continuation is kept if
  AS: some prefix starting at the first "[" (BEHAVIOR) or "{" (VirtualHome)
      is a well-formed action list over the prompt's actions and objects,
      read token by token (training/corpus_filter.py); the accepted prefix
      is converted (training/semantic.py);
  SD: its "output" list parses and is closed; each atom becomes
      PRED|NOT_PRED followed by one ID per argument (canonical ids as in the
      decoder);
and yields at least one meta-token before EOS.
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_filter import build_spec, first_accept  # noqa: E402
from semantic import (  # noqa: E402
    behavior_output_text_to_semantic_tokens,
    parse_virtualhome_json_pairs_prefix,
    virtualhome_pairs_to_semantic_tokens,
)


# ---------------------------------------------------------------- shared helpers

def load_vocab(vocab_json: str) -> Tuple[Dict[str, int], int, int]:
    full_vocab = json.load(open(vocab_json))["full_vocab"]
    tok2id = {str(tok): i for i, tok in enumerate(full_vocab)}
    return tok2id, tok2id["EOS"], tok2id["UNK"]


_SPECIAL_RE = re.compile(r"<\|[^|<>]{1,40}\|>")
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def answer_text(raw_with_specials: str) -> str:
    """The answer without special tokens and without a surrounding code fence."""
    body = _SPECIAL_RE.sub("", raw_with_specials).strip()
    cf = _CODE_FENCE_RE.match(body)
    return cf.group(1).strip() if cf else body


def extract_balanced(text: str, open_ch: str, close_ch: str) -> str:
    """The first balanced open_ch ... close_ch substring (JSON string aware), or ""."""
    s = str(text)
    start = s.find(open_ch)
    if start < 0:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


@torch.inference_mode()
def prompt_end_embeddings(base_model: str, prompts_npz: str, device: torch.device, batch_size: int,
                          dtype: torch.dtype) -> np.ndarray:
    """Last hidden state at the last prompt token (prompts are left-padded)."""
    d = np.load(prompts_npz, allow_pickle=True)
    prompts = d["prompts"].astype(np.int64, copy=False)
    pad_id = int(d.get("pad_id", 0))
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=None).to(device)
    model.eval()
    out = np.zeros((prompts.shape[0], model.config.hidden_size), dtype=np.float32)
    for start in range(0, prompts.shape[0], batch_size):
        batch = torch.from_numpy(prompts[start:start + batch_size]).to(device)
        hs = model.model(input_ids=batch, attention_mask=(batch != pad_id).long(), return_dict=True).last_hidden_state
        out[start:start + batch_size] = hs[:, -1, :].float().cpu().numpy()
    return out


# ---------------------------------------------------------------- AS

def as_converter(domain: str, tokenizer, stats: Dict[str, int]) -> Callable:
    specs: Dict[str, object] = {}

    def convert(toks: List[int], ident: str, prompt_text: str) -> Optional[List[str]]:
        if ident not in specs:
            specs[ident] = build_spec(domain, tokenizer, prompt_text)
        span = first_accept(domain, tokenizer, specs[ident], toks)
        if span is None:
            return None
        stats["num_gamma_accepted"] += 1
        text = answer_text(tokenizer.decode(toks[span[0]:span[1] + 1], skip_special_tokens=False))
        if domain == "behavior":
            return behavior_output_text_to_semantic_tokens(
                prompt_text=prompt_text, output_text=extract_balanced(text, "[", "]") or text)
        pairs = parse_virtualhome_json_pairs_prefix(extract_balanced(text, "{", "}") or text)
        return virtualhome_pairs_to_semantic_tokens(pairs, prompt_text=prompt_text)

    return convert


# ---------------------------------------------------------------- SD

_BEH_OBJ_NAME_RE = re.compile(r"'name':\s*'([^']+)'")
_INST_RE = re.compile(r"^[a-zA-Z0-9_]+\.\d+$")
_PRED_CALL_RE = re.compile(r"(\w+)\(([^)]+)\)")
_REL_OBJ_BLOCK_RE = re.compile(r"##\s*Relevant objects.*?\n(.*?)(?:\n##|\Z)", re.DOTALL | re.IGNORECASE)
_ATOM_RE = re.compile(r"^\s*(not\s+)?([A-Za-z_][\w]*)\s*(?:\(([^()]*)\))?\s*$", re.IGNORECASE)
_CONNECTIVE_SPLIT_RE = re.compile(r"\s+(?:and|or)\s+", re.IGNORECASE)


def behavior_canonical_id_map(prompt_text: str) -> Dict[str, int]:
    """{instance: canonical id}, the order of generate_beh_sd's state machine:
    relevant objects and every <base>.<n> argument of the prompt, sorted by
    base name, then by instance."""
    instances = set()
    m = _REL_OBJ_BLOCK_RE.search(str(prompt_text))
    instances.update(_BEH_OBJ_NAME_RE.findall(m.group(1) if m else ""))
    for mm in _PRED_CALL_RE.finditer(str(prompt_text)):
        for arg in mm.group(2).split(","):
            arg = arg.strip()
            if "." in arg and _INST_RE.match(arg):
                instances.add(arg)
    by_base: Dict[str, List[str]] = {}
    for inst in sorted(instances):
        by_base.setdefault(inst.split(".", 1)[0], []).append(inst)
    ordered = [inst for base in sorted(by_base) for inst in sorted(by_base[base])]
    return {inst: i for i, inst in enumerate(ordered)}


def virtualhome_canonical_id_map(prompt_text: str) -> Dict[str, int]:
    """{raw id: canonical id}, as generate_vh_sd.build_id_mapping."""
    from ctrlg.gamma.topdown.subgoal_decomp import parse_sd_prompt

    objects, *_ = parse_sd_prompt(str(prompt_text))
    ordered = [raw_id for name in sorted(objects) for raw_id in sorted(objects[name], key=int)]
    return {raw_id: cid for cid, raw_id in enumerate(ordered)}


def _output_array_content(text: str) -> Optional[str]:
    """Text inside the closed `output` array, or None."""
    s = str(text)
    key = s.find('"output"')
    if key < 0:
        key = s.find("output")
    lb = s.find("[", key) if key >= 0 else s.find("[")
    if lb < 0:
        return None
    depth = 0
    for i in range(lb, len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return s[lb + 1:i]
    return None


def _split_top_level_commas(content: str) -> List[str]:
    parts, cur, depth = [], [], 0
    for ch in content:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def parse_output_list(text: str) -> Optional[List[str]]:
    """Subgoal strings of an SD answer: strict JSON if possible, else the items of
    the closed `output` array (the model often leaves atoms unquoted)."""
    blob = extract_balanced(text, "{", "}")
    if blob:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and isinstance(obj.get("output"), list):
                return [str(x) for x in obj["output"]]
        except Exception:
            pass
    content = _output_array_content(text)
    if content is None:
        return None
    items = []
    for it in _split_top_level_commas(content):
        if len(it) >= 2 and it[0] == '"' and it[-1] == '"':
            it = it[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if it:
            items.append(it)
    return items or None


def sd_semantic_tokens(output_list: Sequence[str], domain: str, id_map: Dict[str, int], id_cap: int) -> List[str]:
    """[PRED|NOT_PRED, ID|UNK, ..., EOS]; conjunctions and disjunctions are flattened."""
    y: List[str] = []
    for subgoal in output_list:
        for part in _CONNECTIVE_SPLIT_RE.split(str(subgoal).strip()):
            m = _ATOM_RE.match(part)
            if not m:
                continue
            pred = m.group(2).upper()
            y.append("NOT_" + pred if m.group(1) else pred)
            for arg in (a.strip() for a in (m.group(3) or "").split(",")):
                if not arg:
                    continue
                if domain == "behavior":
                    cid = id_map.get(arg)
                else:  # name.raw_id
                    cid = id_map.get(arg.rsplit(".", 1)[1]) if "." in arg else None
                y.append("UNK" if cid is None or not 0 <= int(cid) < id_cap else f"ID{int(cid)}")
    return y + ["EOS"]


def sd_converter(domain: str, tokenizer, id_cap: int, stats: Dict[str, int]) -> Callable:
    id_maps: Dict[str, Dict[str, int]] = {}

    def convert(toks: List[int], ident: str, prompt_text: str) -> Optional[List[str]]:
        output_list = parse_output_list(answer_text(tokenizer.decode(toks, skip_special_tokens=False)))
        if output_list is None:
            stats["num_parse_failed_or_unclosed"] += 1
            return None
        if ident not in id_maps:
            try:
                id_maps[ident] = (behavior_canonical_id_map if domain == "behavior"
                                  else virtualhome_canonical_id_map)(prompt_text)
            except Exception:
                id_maps[ident] = {}
        return sd_semantic_tokens(output_list, domain, id_maps[ident], id_cap)

    return convert


# ---------------------------------------------------------------- writer

def write_dataset(args, domain: str, samples_dir: str, prompt_json: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for src, dst in (("prompts.npz", "prompts.npz"), ("split_info.json", "split_info.json"),
                     ("metadata.json", "source_metadata.json")):
        if os.path.exists(os.path.join(samples_dir, src)):
            shutil.copy2(os.path.join(samples_dir, src), os.path.join(out_dir, dst))
    id2prompt = {str(r["identifier"]): str(r["llm_prompt"]) for r in json.load(open(prompt_json))
                 if r.get("identifier") and r.get("llm_prompt")}
    train_ids = [str(x) for x in np.load(os.path.join(samples_dir, "prompts.npz"), allow_pickle=True)["identifiers"]]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tok2id, eos_id, unk_id = load_vocab(args.vocab_json)
    stats = {"num_gamma_accepted": 0, "num_parse_failed_or_unclosed": 0}
    if args.family == "as":
        convert = as_converter(domain, tokenizer, stats)
    else:
        convert = sd_converter(domain, tokenizer, int(json.load(open(args.vocab_json)).get("id_cap", 60)), stats)

    shard_paths = sorted(str(p) for p in Path(samples_dir, "shards").glob("*.npz"))
    cap = sum(int(np.load(p)["continuations"].shape[0]) for p in shard_paths)
    if args.max_samples:
        cap = min(cap, args.max_samples)
    c_max = int(args.c_max)
    seq = np.lib.format.open_memmap(os.path.join(out_dir, "semantic_continuations.npy"), mode="w+",
                                    dtype=np.int32, shape=(cap, c_max))
    prompt_idx = np.zeros((cap,), dtype=np.int32)
    seen = kept = max_len = truncated = 0
    for sp in shard_paths:
        d = np.load(sp)
        cont, pidx, pad_id = d["continuations"], d["prompt_indices"], int(d["pad_id"])
        for j in range(cont.shape[0]):
            if seen >= cap:
                break
            seen += 1
            train_idx = int(pidx[j])
            if not 0 <= train_idx < len(train_ids) or not id2prompt.get(train_ids[train_idx]):
                continue
            ident = train_ids[train_idx]
            toks = cont[j].tolist()
            while toks and toks[-1] == pad_id:
                toks.pop()
            y = convert(toks, ident, id2prompt[ident])
            if y is None or len(y) <= 1:
                continue
            ids = [tok2id.get(t, unk_id) for t in y]
            max_len = max(max_len, len(ids))
            truncated += len(ids) > c_max
            row = seq[kept]
            row[:] = eos_id
            y0 = ids[:c_max]
            if eos_id not in y0:
                y0[-1] = eos_id
            row[:len(y0)] = y0
            prompt_idx[kept] = train_idx
            kept += 1
    seq.flush()
    np.savez_compressed(os.path.join(out_dir, "mapping.npz"), continuation_to_prompt_idx=prompt_idx[:kept])
    info = dict(domain=domain, family=args.family, created_at=datetime.now().isoformat(timespec="seconds"),
                source_samples=str(samples_dir), base_model=args.base_model, vocab_json=args.vocab_json,
                c_max=c_max, num_seen=seen, num_samples=kept, capacity=cap,
                max_semantic_len_observed=max_len, num_truncated_to_c_max=truncated, **stats)
    json.dump(info, open(os.path.join(out_dir, "dataset_info.json"), "w"), indent=2)
    print(f"[{domain}] seen={seen} kept={kept} -> {out_dir}")

    if not args.no_embeddings:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        emb = prompt_end_embeddings(args.base_model, os.path.join(out_dir, "prompts.npz"), device,
                                    args.emb_batch_size, getattr(torch, args.emb_dtype))
        np.savez_compressed(os.path.join(out_dir, "prompt_end_embedding.npz"), prompt_embeddings=emb,
                            embed_dim=emb.shape[1], num_prompts=emb.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["as", "sd"], required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--vocab_json", required=True)
    ap.add_argument("--c_max", type=int, required=True, help="stored width of a meta-token row")
    ap.add_argument("--behavior_samples", required=True)
    ap.add_argument("--behavior_prompts", required=True)
    ap.add_argument("--virtualhome_samples", required=True)
    ap.add_argument("--virtualhome_prompts", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="read at most this many rows per domain")
    ap.add_argument("--no_embeddings", action="store_true")
    ap.add_argument("--emb_batch_size", type=int, default=8)
    ap.add_argument("--emb_dtype", default="float16", help="dtype of the LM for the prompt embeddings")
    args = ap.parse_args()
    write_dataset(args, "behavior", args.behavior_samples, args.behavior_prompts,
                  os.path.join(args.out_root, "behavior"))
    write_dataset(args, "virtualhome", args.virtualhome_samples, args.virtualhome_prompts,
                  os.path.join(args.out_root, "virtualhome"))


if __name__ == "__main__":
    main()

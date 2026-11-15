"""Vocabulary of a subgoal-decomposition meta-token HMM.

An SD meta-HMM is read with the vocabulary file stored in its own directory
(the union vocabulary of the trained HMMs, or the per-domain vocabulary of a
uniform one); any mismatch with the HMM's emission columns is an error.

The BEHAVIOR predicates `nextto` and `touching` (and their negations) have no
entry of their own in the union vocabulary; the corpus represents them as UNK
and decoding does the same, so the HMM scores them as it was trained.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .backward_dp import SemanticVocab

OWN_VOCAB_FILES = ("hmm_vocab_sd_predicates_union.json", "sd_semantic_vocab.json")
_BEH_UNK = ("NEXTTO", "TOUCHING", "NOT_NEXTTO", "NOT_TOUCHING")


def _full_vocab(path: Path) -> List[str]:
    data = json.load(open(path))
    if isinstance(data, dict) and "full_vocab" in data:
        return [str(t) for t in data["full_vocab"]]
    if isinstance(data, list):
        return [str(t) for t in data]
    raise ValueError(f"SD vocab {path} has no full_vocab list")


def load_sd_semantic_vocab(semantic_hmm_path: str, domain: str) -> SemanticVocab:
    """Vocabulary next to the HMM; domain is "virtualhome" or "behavior"."""
    d = Path(semantic_hmm_path)
    found = [d / f for f in OWN_VOCAB_FILES if (d / f).exists()]
    if len(found) != 1:
        raise ValueError(f"expected one SD vocabulary file in {d}, found {[f.name for f in found]}")
    vocab = SemanticVocab({tok: i for i, tok in enumerate(_full_vocab(found[0]))})
    vocab.source_path = str(found[0])
    vocab.strict_actions = True  # an unknown predicate is an error, not UNK
    if domain == "behavior" and "NEXT_TO" in vocab.token_to_idx:
        vocab.aliases = {k: "UNK" for k in _BEH_UNK if k not in vocab.token_to_idx}
    return vocab


def check_vocab_matches_hmm(semantic_hmm, vocab: SemanticVocab) -> None:
    """Error unless `vocab` has the HMM's emission width and, when the HMM
    directory stores a vocabulary file, exactly its symbols in its order."""
    v0 = getattr(semantic_hmm, "trained_vocab_size", None)
    if v0 is not None and int(v0) != len(vocab):
        raise ValueError(f"semantic vocab has {len(vocab)} symbols, the HMM {int(v0)} emission columns "
                         f"(hmm={getattr(semantic_hmm, 'source_dir', '?')})")
    hdir = getattr(semantic_hmm, "source_dir", None)
    if hdir is None:
        return
    own = [Path(hdir) / f for f in OWN_VOCAB_FILES + ("hmm_vocab_as_actions_only_full_schema.json",)
           if (Path(hdir) / f).exists()]
    if own and _full_vocab(own[0]) != [vocab.idx_to_token[i] for i in range(len(vocab))]:
        raise ValueError(f"semantic vocab order differs from {own[0]}")

"""Hard mask from the syntax automaton over the LM's tokens."""
from typing import List

import numpy as np
import torch
from transformers import LogitsProcessor

from ..common.constants import NEG_INF


class TokenMask(LogitsProcessor):
    """Tracks the syntax-automaton state along the generated tokens and masks
    every token the automaton does not allow next. Greedy decoding only."""

    def __init__(self, dfa_model, prompt_ids: List[int], vocab_size: int, eos_token_id: int, device: str):
        self.dfa = dfa_model
        self.vocab_size = int(vocab_size)
        self.eos_token_id = int(eos_token_id)
        self.device = device
        self.prompt_len = len(prompt_ids)
        self._n_seen = 0
        self.state = dfa_model.initial_state

    def get_stats(self) -> dict:
        return {"generated_tokens": self._n_seen}

    def _advance(self, input_ids: torch.LongTensor) -> None:
        gen = input_ids[0, self.prompt_len:].tolist()
        for tok in gen[self._n_seen:]:
            self.state = self.dfa.next_state(self.state, tok)
        self._n_seen = len(gen)

    def _valid_mask(self) -> torch.BoolTensor:
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for _dst, bitset in self.dfa.G.get(self.state, []):
            for tok in np.where(bitset)[0]:
                if 0 <= tok < self.vocab_size:
                    mask[tok] = True
        return mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert input_ids.shape[0] == 1, "greedy decoding with batch size 1 only"
        self._advance(input_ids)
        valid = self._valid_mask()
        if not valid.any():
            out = torch.full_like(scores, NEG_INF)
            out[:, self.eos_token_id] = 0.0
            return out
        out = scores.clone()
        out[:, ~valid] = NEG_INF
        return out

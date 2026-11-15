"""
Token set utilities for top-down DFA compilation.

This module computes tokenizer-derived token sets (WS_TOKENS, QUOTE_ALTS, etc.)
without any prompt dependency. These sets are purely tokenizer-dependent and
can be cached per tokenizer.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

# =============================================================================
# Caches (per-tokenizer)
# =============================================================================

_TOKEN_TEXT_CACHE: Dict = {}
_TOKEN_SETS_CACHE: Dict = {}

# JSON whitespace characters
_JSON_WS_CHARS = " \t\n\r"


def _is_json_ws(ch: str) -> bool:
    return ch in _JSON_WS_CHARS


def _vocab_lim(tokenizer, eos_token_id: int) -> int:
    """Get vocabulary limit for iteration."""
    try:
        v = int(getattr(tokenizer, "vocab_size", 0) or 0)
    except Exception:
        v = 0
    try:
        lim = max(v, int(eos_token_id) + 1)
    except Exception:
        lim = v
    return max(lim, 1)


def _token_texts(tokenizer, *, vocab_lim: int) -> List[str]:
    """
    Cache per-token decoded text with leading whitespace preserved.
    Uses the "special-prefix decode" trick to avoid Llama-style stripping.
    """
    key = (id(tokenizer), int(vocab_lim))
    if key in _TOKEN_TEXT_CACHE:
        return _TOKEN_TEXT_CACHE[key]

    sp_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    sp = int(sp_ids[0]) if sp_ids else int(getattr(tokenizer, "eos_token_id", 0) or 0)
    pref = tokenizer.decode([sp])

    out = []
    for tid in range(int(vocab_lim)):
        try:
            s = tokenizer.decode([sp, int(tid)])
        except Exception:
            out.append("")
            continue
        if not s.startswith(pref):
            try:
                out.append(tokenizer.decode([int(tid)]))
            except Exception:
                out.append("")
            continue
        out.append(s[len(pref):])

    _TOKEN_TEXT_CACHE[key] = out
    return out


def get_eos_token_ids(tokenizer, *, eos_token_id: int) -> FrozenSet[int]:
    """
    Return token IDs that should be treated as "end tokens" for DFAs.

    For Llama-3 chat models, generations often terminate with <|eot_id|>
    instead of `eos_token_id`. When constraining decoding, we must allow both.
    """
    out = set()
    try:
        out.add(int(eos_token_id))
    except Exception:
        pass
    # Llama-3 convention: <|eot_id|> marks end-of-turn
    try:
        tid = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if tid is not None:
            out.add(int(tid))
    except Exception:
        pass
    return frozenset(int(x) for x in out if isinstance(x, int) and int(x) >= 0)


# =============================================================================
# Token Set Computation Functions
# =============================================================================

def _ws_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens that are pure JSON whitespace."""
    ws = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        if s.strip(_JSON_WS_CHARS) == "" and all(_is_json_ws(ch) for ch in s):
            ws.add(int(tid))
    return frozenset(ws)


def _punct_tokens(texts: List[str], ch: str, allow_trailing_ws: bool = True) -> FrozenSet[int]:
    """Tokens that are: optional WS + ch + (optional WS if allow_trailing_ws)."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        if s.strip(_JSON_WS_CHARS) != ch:
            continue
        if not allow_trailing_ws and not s.endswith(ch):
            continue
        if all(_is_json_ws(c) or c == ch for c in s):
            out.add(int(tid))
    return frozenset(out)


def _punct_tokens_no_leading_ws(texts: List[str], ch: str, allow_trailing_ws: bool = True) -> FrozenSet[int]:
    """Like _punct_tokens but require token to START with ch (no leading WS)."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        if not s.startswith(ch):
            continue
        if s.strip(_JSON_WS_CHARS) != ch:
            continue
        if not allow_trailing_ws and not s.endswith(ch):
            continue
        if all(_is_json_ws(c) or c == ch for c in s):
            out.add(int(tid))
    return frozenset(out)


def _brace_quote_tokens(texts: List[str], brace: str) -> FrozenSet[int]:
    """Tokens: optional WS + brace + optional WS + '"' (ends with quote)."""
    out = set()
    for tid, s in enumerate(texts):
        if not s or not s.endswith('"'):
            continue
        s1 = s.lstrip(_JSON_WS_CHARS)
        if not s1.startswith(brace):
            continue
        rest = s1[len(brace):]
        if rest.strip(_JSON_WS_CHARS) == '"' and s.endswith('"'):
            out.add(int(tid))
    return frozenset(out)


def _brace_quote_tokens_no_leading_ws(texts: List[str], brace: str) -> FrozenSet[int]:
    """Like _brace_quote_tokens but token must START with brace."""
    out = set()
    for tid, s in enumerate(texts):
        if not s or not s.endswith('"'):
            continue
        if not s.startswith(brace):
            continue
        rest = s[len(brace):]
        if rest.strip(_JSON_WS_CHARS) == '"' and s.endswith('"'):
            if all(_is_json_ws(c) or c in {brace, '"'} for c in s):
                out.add(int(tid))
    return frozenset(out)


def _quote_colon_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens: optional WS + '"' + optional WS + ':' + optional WS."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        s1 = s.lstrip(_JSON_WS_CHARS)
        if not s1.startswith('"'):
            continue
        rest = s1[1:].lstrip(_JSON_WS_CHARS)
        if not rest.startswith(":"):
            continue
        tail = rest[1:]
        if tail.strip(_JSON_WS_CHARS) != "":
            continue
        if all(_is_json_ws(c) or c in {'"', ':'} for c in s):
            out.add(int(tid))
    return frozenset(out)


def _quote_comma_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens: optional WS + '"' + optional WS + ',' + optional WS."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        s1 = s.lstrip(_JSON_WS_CHARS)
        if not s1.startswith('"'):
            continue
        rest = s1[1:].lstrip(_JSON_WS_CHARS)
        if not rest.startswith(","):
            continue
        tail = rest[1:]
        if tail.strip(_JSON_WS_CHARS) != "":
            continue
        if all(_is_json_ws(c) or c in {'"', ','} for c in s):
            out.add(int(tid))
    return frozenset(out)


def _quote_rbrack_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens: optional WS + '"' + optional WS + ']' + optional WS."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        s1 = s.lstrip(_JSON_WS_CHARS)
        if not s1.startswith('"'):
            continue
        rest = s1[1:].lstrip(_JSON_WS_CHARS)
        if not rest.startswith(']'):
            continue
        tail = rest[1:]
        if tail.strip(_JSON_WS_CHARS) != "":
            continue
        if all(_is_json_ws(c) or c in {'"', ']'} for c in s):
            out.add(int(tid))
    return frozenset(out)


def _quote_rbrack_comma_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens: optional WS + '"' + optional WS + ']' + optional WS + ','."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        s1 = s.lstrip(_JSON_WS_CHARS)
        if not s1.startswith('"'):
            continue
        rest = s1[1:].lstrip(_JSON_WS_CHARS)
        if not rest.startswith(']'):
            continue
        rest2 = rest[1:].lstrip(_JSON_WS_CHARS)
        if not rest2.startswith(','):
            continue
        tail = rest2[1:]
        if tail.strip(_JSON_WS_CHARS) != "":
            continue
        if all(_is_json_ws(c) or c in {'"', ']', ','} for c in s):
            out.add(int(tid))
    return frozenset(out)


def _quote_rbrack_rbrace_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens: optional WS + '"' + optional WS + ']' + optional WS + '}'."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        s1 = s.lstrip(_JSON_WS_CHARS)
        if not s1.startswith('"'):
            continue
        rest = s1[1:].lstrip(_JSON_WS_CHARS)
        if not rest.startswith(']'):
            continue
        rest2 = rest[1:].lstrip(_JSON_WS_CHARS)
        if not rest2.startswith('}'):
            continue
        tail = rest2[1:]
        if tail.strip(_JSON_WS_CHARS) != "":
            continue
        if all(_is_json_ws(c) or c in {'"', ']', '}'} for c in s):
            out.add(int(tid))
    return frozenset(out)


def _rbrack_leading_ws_tokens(texts: List[str]) -> FrozenSet[int]:
    """Tokens: whitespace + ']' (leading WS required before bracket)."""
    out = set()
    for tid, s in enumerate(texts):
        if not s:
            continue
        # Must have leading whitespace
        if s[0] not in _JSON_WS_CHARS:
            continue
        stripped = s.lstrip(_JSON_WS_CHARS)
        if not stripped.startswith(']'):
            continue
        tail = stripped[1:]
        if tail.strip(_JSON_WS_CHARS) != "":
            continue
        if all(_is_json_ws(c) or c == ']' for c in s):
            out.add(int(tid))
    return frozenset(out)


# =============================================================================
# Main Token Sets Class
# =============================================================================

@dataclass(frozen=True)
class VHTokenSets:
    """Token sets for VirtualHome JSON format: {"ACTION": ["name", "id", ...], ...}"""

    WS_TOKENS: FrozenSet[int]
    LBRACE_START_ALTS: FrozenSet[int]      # { at start (no leading WS)
    LBRACE_QUOTE_START_ALTS: FrozenSet[int] # {" at start (no leading WS)
    QUOTE_ALTS: FrozenSet[int]
    QUOTE_COLON_ALTS: FrozenSet[int]
    COLON_ALTS: FrozenSet[int]
    LBRACK_ALTS: FrozenSet[int]
    LBRACK_QUOTE_ALTS: FrozenSet[int]
    RBRACK_ALTS: FrozenSet[int]
    RBRACK_LEADING_WS_ALTS: FrozenSet[int]  # ] with leading WS
    COMMA_ALTS: FrozenSet[int]
    RBRACE_ALTS: FrozenSet[int]
    QUOTE_COMMA_ALTS: FrozenSet[int]
    QUOTE_RBRACK_ALTS: FrozenSet[int]
    QUOTE_RBRACK_COMMA_ALTS: FrozenSet[int]
    QUOTE_RBRACK_RBRACE_ALTS: FrozenSet[int]
    eos_token_ids: FrozenSet[int]

    @classmethod
    def from_tokenizer(cls, tokenizer, eos_token_id: int) -> "VHTokenSets":
        """Build all VH token sets from tokenizer."""
        lim = _vocab_lim(tokenizer, eos_token_id)
        texts = _token_texts(tokenizer, vocab_lim=lim)
        eos = get_eos_token_ids(tokenizer, eos_token_id=eos_token_id)

        return cls(
            WS_TOKENS=_ws_tokens(texts),
            LBRACE_START_ALTS=_punct_tokens_no_leading_ws(texts, "{", allow_trailing_ws=True),
            LBRACE_QUOTE_START_ALTS=_brace_quote_tokens_no_leading_ws(texts, "{"),
            QUOTE_ALTS=_punct_tokens(texts, '"', allow_trailing_ws=False),
            QUOTE_COLON_ALTS=_quote_colon_tokens(texts),
            COLON_ALTS=_punct_tokens(texts, ":", allow_trailing_ws=True),
            LBRACK_ALTS=_punct_tokens(texts, "[", allow_trailing_ws=True),
            LBRACK_QUOTE_ALTS=_brace_quote_tokens(texts, "["),
            RBRACK_ALTS=_punct_tokens(texts, "]", allow_trailing_ws=True),
            RBRACK_LEADING_WS_ALTS=_rbrack_leading_ws_tokens(texts),
            COMMA_ALTS=_punct_tokens(texts, ",", allow_trailing_ws=True),
            RBRACE_ALTS=_punct_tokens(texts, "}", allow_trailing_ws=True),
            QUOTE_COMMA_ALTS=_quote_comma_tokens(texts),
            QUOTE_RBRACK_ALTS=_quote_rbrack_tokens(texts),
            QUOTE_RBRACK_COMMA_ALTS=_quote_rbrack_comma_tokens(texts),
            QUOTE_RBRACK_RBRACE_ALTS=_quote_rbrack_rbrace_tokens(texts),
            eos_token_ids=eos,
        )


def get_vh_token_sets(tokenizer, eos_token_id: int) -> VHTokenSets:
    """Get cached VH token sets for tokenizer."""
    key = ("vh", id(tokenizer), int(eos_token_id))
    if key not in _TOKEN_SETS_CACHE:
        _TOKEN_SETS_CACHE[key] = VHTokenSets.from_tokenizer(tokenizer, eos_token_id)
    return _TOKEN_SETS_CACHE[key]

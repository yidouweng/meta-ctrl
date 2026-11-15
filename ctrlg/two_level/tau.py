"""Parsers (tau) that turn a completed token block into a MetaStep (action, object ids):
VirtualHome AS and VirtualHome SD. The BEHAVIOR parsers are in beh_tau.py and beh_sd_tau.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MetaStep:
    """A grounded action, emitted by the parser tau when gamma completes a block.

    - action: upper-case action name
    - ids: 0-based canonical object ids, one per argument role
    """

    action: str
    ids: Tuple[int, ...]


def _unwrap_ws_state(state):
    """Split a gamma state wrapped as ("_WS", base_state, ws_run) (used to bound
    consecutive whitespace) into (base_state, ws_run); unwrapped states give run 0."""
    if isinstance(state, tuple) and state and state[0] == "_WS":
        try:
            return state[1], int(state[2])
        except Exception:
            return state[1], 0
    return state, 0


class VirtualHomeTauParser:
    """Parser tau for the VirtualHome Action Sequencing output format (a JSON
    dict of arrays).

    A block is one entry `"ACT": ["name", "id", ...]`; it ends when gamma enters
    BLOCK_BOUNDARY after the value list closes. tau emits exactly once per block.
    Raw ids are mapped to canonical ids through `orig_id_to_canonical`.
    """

    def __init__(
        self,
        *,
        tokenizer,
        gamma_dfa,
        orig_id_to_canonical: Dict[str, int],
    ):
        self.tokenizer = tokenizer
        self.gamma_dfa = gamma_dfa
        self.orig_id_to_canonical = {str(k): int(v) for k, v in (orig_id_to_canonical or {}).items()}

        self._state = self.gamma_dfa.initial()
        self._in_block = False
        self._cur_action: Optional[str] = None
        self._cur_raw_ids: List[str] = []

    def reset(self) -> None:
        self._state = self.gamma_dfa.initial()
        self._in_block = False
        self._cur_action = None
        self._cur_raw_ids = []

    def advance_with_states(self, *, tok_id: int, prev_state, new_state) -> Optional[MetaStep]:
        """Update tau from a token and the gamma transition it caused (the
        candidate tracker has already stepped gamma)."""
        self._state = new_state

        prev_phase = getattr(self.gamma_dfa, "get_phase", lambda s: _unwrap_ws_state(s)[0][0])(prev_state)
        new_phase = getattr(self.gamma_dfa, "get_phase", lambda s: _unwrap_ws_state(s)[0][0])(new_state)

        # Record the action and enter the block once gamma has committed the action.
        act = getattr(self.gamma_dfa, "get_action", lambda s: None)(new_state)
        if act is not None and new_phase in {"ARRAY_OPEN", "ARRAY_BODY", "EXPECT_NAME", "VALUE_QUOTE_OPEN_ID", "EXPECT_ID", "PAIRS_TAIL"}:
            self._cur_action = str(act).upper()
            self._in_block = True

        # On leaving a completed id, recover the raw id from the id-trie leaf label.
        if prev_phase == "EXPECT_ID":
            name, node = getattr(self.gamma_dfa, "get_vh_expect_id_fields", lambda s: (None, None))(prev_state)
            if name is not None and node is not None:
                trie = getattr(self.gamma_dfa, "id_trie_by_name", {}).get(str(name))
                if trie is not None:
                    nxt = trie.next_node(int(node), int(tok_id))
                    if nxt is None and trie.is_leaf_node(int(node)):
                        rid = trie.get_leaf_label(int(node))
                        if rid is not None:
                            self._cur_raw_ids.append(str(rid))

        # Emit once when gamma enters BLOCK_BOUNDARY.
        if self._in_block and new_phase == "BLOCK_BOUNDARY" and prev_phase != "BLOCK_BOUNDARY":
            if not self._cur_action:
                raise ValueError("τ reached block boundary but missing action.")

            # Unknown raw ids map to -1.
            ids: List[int] = []
            for rid in self._cur_raw_ids:
                if str(rid) in self.orig_id_to_canonical:
                    ids.append(int(self.orig_id_to_canonical[str(rid)]))
                else:
                    ids.append(-1)

            step = MetaStep(action=str(self._cur_action).upper(), ids=tuple(ids))

            # Reset; tokens until the next block are glue.
            self._in_block = False
            self._cur_action = None
            self._cur_raw_ids = []
            return step

        return None

    def advance(self, tok_id: int) -> Optional[MetaStep]:
        """Standalone use: step gamma internally, then `advance_with_states`."""
        prev_state = self._state
        new_state = self.gamma_dfa.step(self._state, int(tok_id))
        if new_state == getattr(self.gamma_dfa, "DEAD", ("DEAD",)):
            raise ValueError("τ advanced into DEAD γ state (prefix not γ-valid).")
        return self.advance_with_states(tok_id=int(tok_id), prev_state=prev_state, new_state=new_state)


class SubgoalDecompTauParser:
    """Parser tau for the VirtualHome Subgoal Decomposition output format.

    Format: {"output": ["PRED(name.id)", "PRED(name1.id1, name2.id2)", ...]}

    A predicate is complete when gamma moves from CLOSE_QUOTE_SEQ to AFTER_STEP;
    tau emits once per predicate. The predicate, object names and raw ids are
    read from the PRED_NAME, ARG_NAME and ARG_ID trie leaves.
    """

    def __init__(
        self,
        *,
        tokenizer,
        gamma_dfa,
        orig_id_to_canonical: Dict[str, int],
    ):
        self.tokenizer = tokenizer
        self.gamma_dfa = gamma_dfa
        self.orig_id_to_canonical = {str(k): int(v) for k, v in (orig_id_to_canonical or {}).items()}

        self._state = self.gamma_dfa.initial()
        self._in_block = False
        self._cur_pred: Optional[str] = None
        self._cur_raw_ids: List[str] = []
        self._cur_name: Optional[str] = None  # tracks current arg's object name

    def reset(self) -> None:
        self._state = self.gamma_dfa.initial()
        self._in_block = False
        self._cur_pred = None
        self._cur_raw_ids = []
        self._cur_name = None

    def advance_with_states(self, *, tok_id: int, prev_state, new_state) -> Optional[MetaStep]:
        self._state = new_state

        prev_base, _ = _unwrap_ws_state(prev_state)
        new_base, _ = _unwrap_ws_state(new_state)
        prev_phase = prev_base[0] if isinstance(prev_base, tuple) and prev_base else str(prev_base)
        new_phase = new_base[0] if isinstance(new_base, tuple) and new_base else str(new_base)

        # Capture predicate name when trie leaf is consumed (transition out of PRED_NAME)
        if prev_phase == 'PRED_NAME' and new_phase in ('LPAREN_SEQ', 'RPAREN_SEQ', 'ARG_NAME'):
            # prev_state was at pred trie leaf; extract label
            pred_node = prev_base[1] if len(prev_base) > 1 else None
            if pred_node is not None:
                pred_trie = getattr(self.gamma_dfa, 'pred_trie', None)
                if pred_trie and pred_trie.is_leaf_node(int(pred_node)):
                    label = pred_trie.get_leaf_label(int(pred_node))
                    if label:
                        self._cur_pred = str(label).upper()
                        self._in_block = True
                        self._cur_raw_ids = []

        # PRED_NAME -> ARG_NAME (a single-token '(') is already handled above.
        if prev_phase == 'PRED_NAME' and new_phase == 'ARG_NAME':
            pass

        # Capture object name when ARG_NAME trie leaf is consumed (transition to DOT_SEQ or ARG_ID)
        if prev_phase == 'ARG_NAME' and new_phase in ('DOT_SEQ', 'ARG_ID'):
            name_node = prev_base[3] if len(prev_base) > 3 else None
            if name_node is not None:
                name_trie = getattr(self.gamma_dfa, 'name_trie', None)
                if name_trie and name_trie.is_leaf_node(int(name_node)):
                    label = name_trie.get_leaf_label(int(name_node))
                    if label:
                        self._cur_name = str(label)

        # Capture raw ID when ARG_ID trie leaf is consumed
        # (transition to ARG_SEP_SEQ, RPAREN_SEQ, or CLOSE_QUOTE_SEQ)
        if prev_phase == 'ARG_ID' and new_phase in ('ARG_SEP_SEQ', 'RPAREN_SEQ', 'CLOSE_QUOTE_SEQ'):
            id_node = prev_base[4] if len(prev_base) > 4 else None
            name = prev_base[3] if len(prev_base) > 3 else self._cur_name
            if id_node is not None and name is not None:
                id_trie = getattr(self.gamma_dfa, 'id_trie_by_name', {}).get(str(name))
                if id_trie and id_trie.is_leaf_node(int(id_node)):
                    raw_id = id_trie.get_leaf_label(int(id_node))
                    if raw_id is not None:
                        self._cur_raw_ids.append(str(raw_id))

        # Emit when predicate is complete: CLOSE_QUOTE_SEQ → AFTER_STEP
        if self._in_block and new_phase == 'AFTER_STEP' and prev_phase == 'CLOSE_QUOTE_SEQ':
            if not self._cur_pred:
                raise ValueError("τ reached step boundary but missing predicate name.")

            ids: List[int] = []
            for rid in self._cur_raw_ids:
                if str(rid) in self.orig_id_to_canonical:
                    ids.append(int(self.orig_id_to_canonical[str(rid)]))
                else:
                    ids.append(-1)

            step = MetaStep(action=str(self._cur_pred).upper(), ids=tuple(ids))

            self._in_block = False
            self._cur_pred = None
            self._cur_raw_ids = []
            self._cur_name = None
            return step

        return None

    def advance(self, tok_id: int) -> Optional[MetaStep]:
        prev_state = self._state
        new_state = self.gamma_dfa.step(self._state, int(tok_id))
        if new_state == getattr(self.gamma_dfa, "DEAD", ("DEAD",)):
            raise ValueError("τ advanced into DEAD γ state (prefix not γ-valid).")
        return self.advance_with_states(tok_id=int(tok_id), prev_state=prev_state, new_state=new_state)



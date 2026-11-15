"""Parser tau for the BEHAVIOR Action Sequencing output format.

Detects action block boundaries and emits MetaStep(action, canonical_ids).
Works with BehaviorTwoLevelStateMachine from ctrlg/gamma/topdown/behavior_twolevel.py.

Block boundary: when EXPECT_INSTANCE leaf is consumed and DFA transitions
to CLOSE_DICT or AFTER_DICT, the action block is complete.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from .tau import MetaStep, _unwrap_ws_state


class BehaviorTwoLevelTauParser:
    """tau for the format [{"action": "...", "object": "..."}, ...]."""

    def __init__(self, *, tokenizer, gamma_dfa, instance_to_canonical: Dict[str, int]):
        self.tokenizer = tokenizer
        self.gamma_dfa = gamma_dfa
        self.instance_to_canonical = {str(k): int(v) for k, v in (instance_to_canonical or {}).items()}

        self._state = self.gamma_dfa.initial()
        self._in_block = False
        self._cur_action: Optional[str] = None
        self._cur_instances: List[str] = []

    def reset(self) -> None:
        self._state = self.gamma_dfa.initial()
        self._in_block = False
        self._cur_action = None
        self._cur_instances = []

    def advance_with_states(self, *, tok_id: int, prev_state, new_state) -> Optional[MetaStep]:
        self._state = new_state

        prev_base, _ = _unwrap_ws_state(prev_state)
        new_base, _ = _unwrap_ws_state(new_state)
        prev_phase = prev_base[0] if isinstance(prev_base, tuple) and prev_base else str(prev_base)
        new_phase = new_base[0] if isinstance(new_base, tuple) and new_base else str(new_base)

        # Capture action name when action trie leaf is consumed
        # (transition out of EXPECT_ACTION to MID_SEQ)
        if prev_phase == 'EXPECT_ACTION' and new_phase == 'MID_SEQ':
            action = new_base[1] if len(new_base) > 1 else None
            if action:
                self._cur_action = str(action).upper()
                self._in_block = True
                self._cur_instances = []

        # Capture instance name when instance trie leaf is consumed
        # Transitions: EXPECT_INSTANCE → CLOSE_DICT/AFTER_DICT/OBJ_SEP/SEP_SEQ/END
        # Also: EXPECT_INSTANCE(role=N, leaf) → EXPECT_INSTANCE(role=N+1, 0) when obj_sep is 1 token
        if prev_phase == 'EXPECT_INSTANCE' and new_phase in ('CLOSE_DICT', 'AFTER_DICT', 'OBJ_SEP', 'SEP_SEQ', 'END'):
            node = prev_base[3] if len(prev_base) > 3 else None
            if node is not None:
                inst_trie = getattr(self.gamma_dfa, 'instance_trie', None)
                if inst_trie and inst_trie.is_leaf_node(int(node)):
                    label = inst_trie.get_leaf_label(int(node))
                    if label:
                        self._cur_instances.append(str(label))

        # Arity-2: EXPECT_INSTANCE(role=N, leaf) → EXPECT_INSTANCE(role=N+1, 0)
        # when obj_sep is 1 token, the comma transitions directly to next role
        if (prev_phase == 'EXPECT_INSTANCE' and new_phase == 'EXPECT_INSTANCE'
                and isinstance(prev_base, tuple) and isinstance(new_base, tuple)
                and len(prev_base) > 3 and len(new_base) > 3
                and new_base[3] == 0  # new role starts at trie root
                and prev_base[2] != new_base[2]):  # role changed
            node = prev_base[3]
            inst_trie = getattr(self.gamma_dfa, 'instance_trie', None)
            if inst_trie and inst_trie.is_leaf_node(int(node)):
                label = inst_trie.get_leaf_label(int(node))
                if label:
                    self._cur_instances.append(str(label))

        # Emit when dict closes (AFTER_DICT or SEP_SEQ or END reached after instance)
        if self._in_block and new_phase in ('AFTER_DICT', 'SEP_SEQ', 'END'):
            if not self._cur_action:
                return None

            # Map instance names to canonical IDs
            ids: List[int] = []
            for inst in self._cur_instances:
                cid = self.instance_to_canonical.get(str(inst))
                if cid is not None:
                    ids.append(cid)
                else:
                    ids.append(-1)

            step = MetaStep(action=str(self._cur_action).upper(), ids=tuple(ids))

            self._in_block = False
            self._cur_action = None
            self._cur_instances = []
            return step

        return None

    def advance(self, tok_id: int) -> Optional[MetaStep]:
        prev_state = self._state
        new_state = self.gamma_dfa.step(self._state, int(tok_id))
        if new_state == getattr(self.gamma_dfa, "DEAD", ("DEAD",)):
            raise ValueError("τ advanced into DEAD γ state.")
        return self.advance_with_states(tok_id=int(tok_id), prev_state=prev_state, new_state=new_state)

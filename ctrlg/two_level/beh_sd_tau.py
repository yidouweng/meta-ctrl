"""Parser tau for the BEHAVIOR Subgoal Decomposition output format.

Counterpart of `beh_tau.py` for the gamma automaton in
`ctrlg/gamma/topdown/behavior_sd_twolevel.py`.

Output format:
    {"output": ["pred(arg)", "pred(arg1, arg2) and not pred(arg3)", ...]}

Each emitted MetaStep represents one predicate atom (positive or negated):
    MetaStep(action="OPEN", ids=(7,))            ← `open(jar.0)`
    MetaStep(action="NOT_INSIDE", ids=(8, 4))    ← `not inside(peach.0, jar.0)`

A conjunct may contain multiple atoms joined by ` and `; we emit one
MetaStep per atom (boundary = entering `AFTER_PRED`).

Negation is tracked across the optional `NEG_SEQ` phase that precedes
`EXPECT_PRED`. The flag is reset at every `PRED_START` so subsequent
conjuncts within the same step start fresh.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .tau import MetaStep, _unwrap_ws_state


def _phase(state) -> str:
    base, _ = _unwrap_ws_state(state)
    if isinstance(base, tuple) and base:
        return str(base[0])
    return str(base)


def _base(state):
    base, _ = _unwrap_ws_state(state)
    return base


class BehaviorSDTwoLevelTauParser:
    """Parser tau for the BEHAVIOR Subgoal Decomposition format.

    Walks the gamma automaton in lockstep with the LM and emits a MetaStep each time a
    complete `pred(args)` (or `not pred(args)`) atom is consumed.
    """

    def __init__(self, *, tokenizer, gamma_dfa,
                 instance_to_canonical: Dict[str, int]):
        self.tokenizer = tokenizer
        self.gamma_dfa = gamma_dfa
        self.instance_to_canonical = {
            str(k): int(v) for k, v in (instance_to_canonical or {}).items()
        }

        self._state = self._dfa_initial()
        self._cur_pred: Optional[str] = None
        self._cur_neg: bool = False
        self._cur_args: List[str] = []

    def reset(self) -> None:
        self._state = self._dfa_initial()
        self._cur_pred = None
        self._cur_neg = False
        self._cur_args = []

    def _dfa_initial(self):
        # Some DFAs expose `initial()`, others `initial_state()`. Support both.
        if hasattr(self.gamma_dfa, 'initial'):
            return self.gamma_dfa.initial()
        return self.gamma_dfa.initial_state()

    # ------------------------------------------------------------------

    def advance_with_states(self, *, tok_id: int, prev_state, new_state
                            ) -> Optional[MetaStep]:
        """Consume one token (gamma already stepped); return a MetaStep when an
        atom completes."""
        self._state = new_state
        prev_phase = _phase(prev_state)
        new_phase = _phase(new_state)
        prev_b = _base(prev_state)
        new_b = _base(new_state)

        # Atom boundary on the LM side: every PRED_START is the start of
        # a fresh conjunct. Reset the per-atom buffer.
        if new_phase == 'PRED_START' and prev_phase != 'PRED_START':
            self._cur_pred = None
            self._cur_neg = False
            self._cur_args = []

        # Negation: entering NEG_SEQ marks this atom as `not pred(...)`.
        # We set the flag once on the transition into NEG_SEQ; it stays
        # true until the next PRED_START reset.
        if new_phase == 'NEG_SEQ' and prev_phase != 'NEG_SEQ':
            self._cur_neg = True

        # Predicate name capture: leaving EXPECT_PRED with a chosen pred.
        # Several outgoing phases all carry the predicate label:
        #   EXPECT_PRED → LPAREN_SEQ            (canonical leaf + '(')
        #   EXPECT_PRED → EXPECT_INSTANCE       (BPE-merged '(X' or pred
        #                                         leaf with TK_PAREN entry)
        if prev_phase == 'EXPECT_PRED' and new_phase != 'EXPECT_PRED':
            # gamma stores the chosen predicate in the destination state's
            # second slot for LPAREN_SEQ / EXPECT_INSTANCE.
            if isinstance(new_b, tuple) and len(new_b) > 1:
                cand = new_b[1]
                if isinstance(cand, str):
                    self._cur_pred = cand.lower()

        # Instance capture, role-by-role.
        # Two transition shapes deposit a complete instance leaf:
        #
        # (a) Leaving EXPECT_INSTANCE entirely:
        #     EXPECT_INSTANCE → {OBJ_SEP_SEQ, RPAREN_SEQ, AFTER_PRED,
        #                        AND_SEQ, NEED_QUOTE, NEED_RBRACE, END,
        #                        NEED_SPACE_PRED1, PRED_START, AFTER_STEP}
        #     (merged `)X` tokens jump straight to the latter phases.)
        #
        # (b) Mid-arity role advance via merged `, X`:
        #     EXPECT_INSTANCE(role=N, leaf) → EXPECT_INSTANCE(role=N+1, 0)
        if prev_phase == 'EXPECT_INSTANCE' and new_phase != 'EXPECT_INSTANCE':
            label = self._instance_leaf_label(prev_b)
            if label:
                self._cur_args.append(label)

        elif (prev_phase == 'EXPECT_INSTANCE' and new_phase == 'EXPECT_INSTANCE'
                and isinstance(prev_b, tuple) and isinstance(new_b, tuple)
                and len(prev_b) >= 5 and len(new_b) >= 5
                and prev_b[2] != new_b[2]):  # role changed
            label = self._instance_leaf_label(prev_b)
            if label:
                self._cur_args.append(label)

        # Atom emission. gamma closes a predicate atom via several
        # alternative phase transitions, depending on which BPE-merged token
        # carried the `)`:
        #
        #   canonical `)`          : EXPECT_INSTANCE → RPAREN_SEQ → AFTER_PRED
        #   merged `) and `        : EXPECT_INSTANCE → AND_SEQ
        #   merged `)"`            : EXPECT_INSTANCE → NEED_QUOTE
        #   merged `)",`           : EXPECT_INSTANCE → NEED_QUOTE
        #   merged `)"]}`          : EXPECT_INSTANCE → NEED_RBRACE / END
        #   merged `) and ` (sp)   : EXPECT_INSTANCE → NEED_SPACE_PRED1
        #   merged `)" ` → next    : EXPECT_INSTANCE → PRED_START
        #
        # All are emission boundaries: emit when the previous phase was inside
        # the atom (EXPECT_INSTANCE or RPAREN_SEQ) and the new one is outside it.
        _atom_inside = {'EXPECT_INSTANCE', 'RPAREN_SEQ'}
        _atom_outside = {'AFTER_PRED', 'AND_SEQ', 'NEED_QUOTE', 'NEED_RBRACE',
                         'NEED_SPACE_PRED1', 'PRED_START', 'AFTER_STEP', 'END'}
        if (prev_phase in _atom_inside and new_phase in _atom_outside
                and self._cur_pred is not None):
            if self._cur_pred is None:
                return None

            ids: List[int] = []
            for inst in self._cur_args:
                cid = self.instance_to_canonical.get(str(inst))
                ids.append(int(cid) if cid is not None else -1)

            action_token = self._cur_pred.upper()
            if self._cur_neg:
                action_token = "NOT_" + action_token

            step = MetaStep(action=action_token, ids=tuple(ids))
            # Reset after emitting. The next atom may be:
            #   - in the same step (separated by ` and ` → AND_SEQ → EXPECT_PRED)
            #   - in the next step (separated by `", "` → PRED_START → EXPECT_PRED)
            # Either way, fresh pred/neg/args will be captured from the new
            # NEG_SEQ entry (if any) and the EXPECT_PRED leaf transition.
            self._cur_pred = None
            self._cur_neg = False
            self._cur_args = []
            return step

        return None

    # ------------------------------------------------------------------

    def advance(self, tok_id: int) -> Optional[MetaStep]:
        prev_state = self._state
        new_state = self.gamma_dfa.step(self._state, int(tok_id))
        if new_state == getattr(self.gamma_dfa, "DEAD", ("DEAD",)):
            raise ValueError("τ advanced into DEAD γ state.")
        return self.advance_with_states(
            tok_id=int(tok_id), prev_state=prev_state, new_state=new_state,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _instance_leaf_label(self, prev_b) -> Optional[str]:
        """Extract the instance-name label from an EXPECT_INSTANCE base tuple,
        if its trie node is a leaf.

        EXPECT_INSTANCE base layout (see behavior_sd_twolevel.py):
            ('EXPECT_INSTANCE', pred, role, kind, node)
        """
        if not (isinstance(prev_b, tuple) and len(prev_b) >= 5):
            return None
        kind, node = prev_b[3], prev_b[4]
        trie = self._trie_for_kind(kind)
        if trie is None:
            return None
        try:
            if not trie.is_leaf_node(int(node)):
                return None
            label = trie.get_leaf_label(int(node))
            if label:
                return str(label)
        except Exception:
            return None
        return None

    def _trie_for_kind(self, kind):
        """Return gamma's instance trie for a TK_* kind constant:
          - instance_trie         (TK_CANON)
          - sp_instance_trie      (TK_SP)
          - paren_instance_trie   (TK_PAREN)
          - commasp_instance_trie (TK_COMMASP)
        """
        gd = self.gamma_dfa
        try:
            if kind == gd.TK_CANON:
                return getattr(gd, 'instance_trie', None)
            if kind == gd.TK_SP:
                return getattr(gd, 'sp_instance_trie', None)
            if kind == gd.TK_PAREN:
                return getattr(gd, 'paren_instance_trie', None)
            if kind == gd.TK_COMMASP:
                return getattr(gd, 'commasp_instance_trie', None)
        except AttributeError:
            return getattr(gd, 'instance_trie', None)
        return getattr(gd, 'instance_trie', None)

"""BEHAVIOR Subgoal Decomposition γ DFA (syntax automaton over tokens).

Counterpart of `behavior_twolevel.py` (action sequencing) for SD outputs.

Output format:
    {"output": ["pred(arg)", "pred(arg1, arg2) and not pred(arg3)", ...]}

Content phases (where β consumes (pred, neg, args) tuples):
    EXPECT_PRED      — predicate name trie
    EXPECT_INSTANCE  — instance trie filtered by per-(predicate, role) affordance
    NEG_SEQ          — optional 'not ' prefix before a conjunct

All other phases are structural JSON / punctuation (tracked, but not content).

The design uses a single instance trie
(no name/ID split) and applies a per-predicate affordance filter at the
EXPECT_INSTANCE position so β can score candidates cleanly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import CanonicalStateMachine
from .trie_builder import TokenTrie, canonical_tokenization



# =============================================================================
# SD predicates (BEHAVIOR — distinct from VH SD predicate set)
# =============================================================================

SD_PREDICATES: Dict[str, int] = {
    # Unary
    'holds_rh': 1, 'holds_lh': 1,
    'open': 1, 'cooked': 1, 'dusty': 1, 'frozen': 1, 'sliced': 1,
    'soaked': 1, 'stained': 1, 'toggledon': 1,
    # Binary
    'inside': 2, 'ontop': 2, 'onfloor': 2, 'nextto': 2, 'under': 2,
    'touching': 2,
    # `on` is deliberately absent: the evaluator's parser only accepts
    # `ontop` (evaluator precondition), so the trie forces that spelling.
}

SD_LOC_PREDS = frozenset({'inside', 'ontop', 'onfloor', 'nextto',
                          'under', 'touching'})
SD_PROP_PREDS = frozenset({'open', 'cooked', 'dusty', 'frozen', 'sliced',
                            'soaked', 'stained', 'toggledon'})
SD_HAND_PREDS = frozenset({'holds_rh', 'holds_lh'})


# =============================================================================
# Per-predicate role affordance
# =============================================================================

# Floor / room keywords (objects that can be floor targets of `onfloor`).
# PROMPT-structural (instance-naming convention), not domain knowledge.
_FLOOR_KW = ('room_floor', 'floor')


def _kw_match_any(name: str, kws: Tuple[str, ...]) -> bool:
    """Word-boundary kw match: kw is a complete `_`-delimited segment of name.
    Also matches if kw is a strict prefix or suffix of name with `_` boundary.
    """
    for kw in kws:
        if (name == kw
                or name.startswith(kw + '_')
                or name.endswith('_' + kw)
                or ('_' + kw + '_') in name):
            return True
    return False


def compute_sd_affordances(
    objects: Dict[str, List[str]],
    predicates_in_prompt: Dict[str, Set[str]],
    membership=None,
    container_role_bases: Optional[Set[str]] = None,
    repositioned_bases: Optional[Set[str]] = None,
) -> Dict[Tuple[str, int], Set[str]]:
    """Compute viable instance bases per (predicate, role).

    Returns {(pred, role): set_of_base_names}. Every entry is a set (never
    None); a set may be empty, in which case the caller drops the predicate.

    Membership (openable / toggleable / graspable) comes from the shared
    taxonomy layer `ctrlg.beta.beh_affordances.BehAffordances`, the same
    layer the β SM gates on, so γ and β agree by construction.

    Args:
        objects: {base_name: [instance_name, ...]} from the prompt.
        predicates_in_prompt: {base_name: {pred1, pred2, ...}} mentions.
        membership: BehAffordances layer (built from the prompt's
            {'name','category'} list). If None, a category-less layer is
            built from the instance names (name-derived synset fallback).
        container_role_bases: bases appearing as arg-1 of an `inside` atom
            in the prompt's initial or goal states.
        repositioned_bases: bases the goal moves (arg-0 of a goal location
            atom); treated as graspable.
    """
    if membership is None:
        from ctrlg.beta.beh_affordances import BehAffordances
        membership = BehAffordances(
            objects=[{'name': i} for insts in objects.values() for i in insts])

    bases = list(objects.keys())
    graspable = {b for b in bases if not membership.non_graspable(b)}
    # Goal-implied graspability: a goal that repositions X presupposes X is
    # carryable, overriding the taxonomy default for
    # goal-repositioned bases (never for floors/rooms/agent).
    for b in repositioned_bases or ():
        if b in objects and not (
                b.startswith('room_') or b.startswith('floor')
                or 'agent' in b or b in ('wall', 'ceiling')):
            graspable.add(b)

    aff: Dict[Tuple[str, int], Set[str]] = {}

    # Hand predicates: any graspable base
    for pred in SD_HAND_PREDS:
        aff[(pred, 0)] = set(graspable)

    # Property predicates — narrow only when we have strong typing info.
    for pred in SD_PROP_PREDS:
        if pred == 'open':
            # `open`/`not open` on X is admissible iff the prompt mentions
            # an open state for X, OR X is an open-gated container in
            # container role (the set β gates on, which keeps β's
            # open-before-inside chain satisfiable). No liberal fallback:
            # a scene with no such object cannot use `open`.
            role = container_role_bases or set()
            aff[(pred, 0)] = {
                b for b in bases
                if 'open' in predicates_in_prompt.get(b, set())
                or (membership.open_gated(b)
                    and b in role)}
        elif pred in {'dusty', 'stained', 'sliced', 'soaked',
                       'cooked', 'frozen'}:
            aff[(pred, 0)] = set(bases)  # liberal — common state predicates
        else:
            # toggledon: taxonomy-toggleable ∪ prompt mention; liberal
            # fallback when the scene has none.
            viable = {b for b in bases
                      if pred in predicates_in_prompt.get(b, set())
                      or membership.toggleable(b)}
            aff[(pred, 0)] = viable if viable else set(bases)

    # Location predicates: subject (arg-0) is liberal; targets (arg-1)
    # only restricted where the predicate is type-strict.
    for pred in SD_LOC_PREDS:
        aff[(pred, 0)] = set(bases)  # any base can be a subject

    # arg-1 strictness:
    # - onfloor: must be a floor (this is structural)
    # - others: liberal (β tightens via specific containment rules)
    aff[('onfloor', 1)] = {b for b in bases if _kw_match_any(b, _FLOOR_KW)}
    aff[('inside',  1)] = set(bases)
    aff[('ontop',   1)] = set(bases)
    aff[('nextto',  1)] = set(bases)
    aff[('under',   1)] = set(bases)
    aff[('touching',1)] = set(bases)

    # Empty sets are kept empty (not relaxed to all bases): the DFA drops
    # any predicate whose arg-0 set is empty (`usable_predicates`).

    return aff


# =============================================================================
# Two-level γ DFA
# =============================================================================

class BehaviorSDTwoLevelStateMachine(CanonicalStateMachine):
    """γ DFA for BEHAVIOR SD output with per-(predicate, role) affordance
    filtering at instance positions."""

    DEAD = ('DEAD',)

    def __init__(
        self,
        tokenizer,
        objects: Dict[str, List[str]],
        predicates_in_prompt: Dict[str, Set[str]],
        eos_token_id: int,
        objects_meta: Optional[List[Dict]] = None,
        container_role_bases: Optional[Set[str]] = None,
        repositioned_bases: Optional[Set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.objects = objects
        self.predicates_in_prompt = predicates_in_prompt

        # Flatten instances
        self.all_instances: List[str] = []
        for base in sorted(objects.keys()):
            for inst in sorted(objects[base]):
                self.all_instances.append(inst)

        # Shared taxonomy membership layer (γ and β read the same sets).
        # objects_meta = prompt [{'name','category'}] list; instances that
        # only appear in state blocks are added category-less (fallback).
        from ctrlg.beta.beh_affordances import BehAffordances
        self.membership = BehAffordances(objects=objects_meta)

        # Compute affordances per (pred, role)
        self.affordances = compute_sd_affordances(
            objects, predicates_in_prompt, membership=self.membership,
            container_role_bases=container_role_bases,
            repositioned_bases=repositioned_bases)

        # Determine which predicates are *actually usable* in this scene:
        # a predicate is dropped from the trie if its affordance excludes
        # every base in the scene (i.e. arg-0 allowed set is empty).
        # Required because `_viable_instance_nodes` returns None ("no
        # restriction") for an empty set.
        usable_preds = []
        for pred in sorted(SD_PREDICATES.keys()):
            arity = SD_PREDICATES[pred]
            allowed_arg0 = self.affordances.get((pred, 0))
            if arity >= 1 and not allowed_arg0:
                continue  # no viable arg → predicate is unusable in this scene
            usable_preds.append(pred)
        self.usable_predicates = frozenset(usable_preds)

        # Build tries with canonical predicate spellings only (`toggledon`,
        # not `toggled.on`); no aliases.
        self.pred_trie = TokenTrie()
        self.pred_token_seqs: Dict[str, Tuple[int, ...]] = {}
        for pred in usable_preds:
            seq = canonical_tokenization(tokenizer, pred)
            if seq:
                self.pred_trie.insert(seq, label=pred)
                self.pred_token_seqs[pred] = tuple(seq)

        self.instance_trie = TokenTrie()
        self.instance_token_seqs: Dict[str, Tuple[int, ...]] = {}
        for inst in self.all_instances:
            seq = canonical_tokenization(tokenizer, inst)
            if seq:
                self.instance_trie.insert(seq, label=inst)
                self.instance_token_seqs[inst] = tuple(seq)

        # Build space-prefixed predicate trie for after " and "
        # (same usable_preds restriction as canonical pred_trie above).
        self.sp_pred_trie = TokenTrie()
        for pred in usable_preds:
            spaced = canonical_tokenization(tokenizer, ' ' + pred)
            if spaced:
                self.sp_pred_trie.insert(spaced, label=pred)

        # Build space-prefixed instance trie for after ", " (binary 2nd arg)
        self.sp_instance_trie = TokenTrie()
        for inst in self.all_instances:
            spaced = canonical_tokenization(tokenizer, ' ' + inst)
            if spaced:
                self.sp_instance_trie.insert(spaced, label=inst)

        # Paren-prefixed and comma-space-prefixed instance tries cover BPE
        # merges such as '(b' or ', biscuit'. The DFA transitions below are
        # canonical-only and do not enter these tries; they are exposed for
        # the τ parser's trie-kind lookup.
        self.paren_instance_trie = TokenTrie()
        for inst in self.all_instances:
            seq = canonical_tokenization(tokenizer, '(' + inst)
            if seq:
                self.paren_instance_trie.insert(seq, label=inst)
        self.commasp_instance_trie = TokenTrie()
        for inst in self.all_instances:
            seq = canonical_tokenization(tokenizer, ', ' + inst)
            if seq:
                self.commasp_instance_trie.insert(seq, label=inst)

        # Precompute structural sequences
        self._precompute_structural(tokenizer)

        # Viability cache
        self._viable_inst_cache: Dict[Tuple[str, int], Optional[Set[int]]] = {}
        self._states = None

    # ------------------------------------------------------------------

    def _precompute_structural(self, tokenizer):
        """Canonical structural token sequences."""
        self.prefix_seq = canonical_tokenization(tokenizer, '{"output": ["')
        self.subgoal_sep_seq = canonical_tokenization(tokenizer, '", "')
        self.close_list_seq = canonical_tokenization(tokenizer, '"]}')
        self.not_seq = canonical_tokenization(tokenizer, 'not ')
        self.sp_not_seq = canonical_tokenization(tokenizer, ' not ')
        self.lparen_seq = canonical_tokenization(tokenizer, '(')
        self.rparen_seq = canonical_tokenization(tokenizer, ')')
        self.obj_sep_seq = canonical_tokenization(tokenizer, ', ')

        self.eos_tokens = {self.eos_token_id}

    # ------------------------------------------------------------------

    # Trie-kind constants (used in EXPECT_INSTANCE state):
    TK_CANON = 0       # canonical instance trie (after standalone '(')
    TK_SP = 1          # space-prefixed (after standalone ', ')
    TK_PAREN = 2       # paren-merged (after pred-leaf via '(X' token)
    TK_COMMASP = 3     # comma-space-merged (after instance-leaf via ', X')

    def _trie_for_kind(self, kind: int) -> TokenTrie:
        if kind == self.TK_SP:      return self.sp_instance_trie
        if kind == self.TK_PAREN:   return self.paren_instance_trie
        if kind == self.TK_COMMASP: return self.commasp_instance_trie
        return self.instance_trie

    def _seq_for_kind(self, kind: int, inst: str):
        if kind == self.TK_SP:
            return canonical_tokenization(self.tokenizer, ' ' + inst)
        if kind == self.TK_PAREN:
            return canonical_tokenization(self.tokenizer, '(' + inst)
        if kind == self.TK_COMMASP:
            return canonical_tokenization(self.tokenizer, ', ' + inst)
        return self.instance_token_seqs.get(inst)

    def _viable_instance_nodes(self, pred: str, role: int,
                               kind: int = 0) -> Optional[Set[int]]:
        """Get instance trie nodes leading to allowed instances for (pred, role)."""
        cache_key = (pred, role, kind)
        cache = self._viable_inst_cache
        if cache_key in cache:
            return cache[cache_key]
        allowed_bases = self.affordances.get((pred, role))
        if allowed_bases is None:
            cache[cache_key] = None
            return None
        allowed_instances = set()
        for base in allowed_bases:
            for inst in self.objects.get(base, []):
                allowed_instances.add(inst)
        trie = self._trie_for_kind(kind)
        viable = set()
        for inst in allowed_instances:
            seq = self._seq_for_kind(kind, inst)
            if not seq:
                continue
            node = 0
            viable.add(node)
            for tok in seq:
                nxt = trie.next_node(node, tok)
                if nxt is None:
                    break
                viable.add(nxt)
                node = nxt
        cache[cache_key] = viable if viable else None
        return viable if viable else None

    # ------------------------------------------------------------------

    def initial_state(self) -> Any:
        return ('PREFIX', 0)

    def is_accept(self, state: Any) -> bool:
        return isinstance(state, tuple) and state[0] == 'END'

    def step(self, state: Any, token: int) -> Any:
        nxt = self.next_state(state, token)
        return nxt if nxt is not None else self.DEAD

    # allowed_tokens / next_state walk the canonical structural sequences,
    # with content filtering at EXPECT_PRED / EXPECT_INSTANCE.

    def allowed_tokens(self, state: Any) -> Set[int]:
        phase = state[0]

        if phase == 'PREFIX':
            idx = state[1]
            if idx < len(self.prefix_seq):
                return {self.prefix_seq[idx]}
            return self.pred_trie.next_tokens(0) | set(self.not_seq[:1] if self.not_seq else ())

        if phase == 'PRED_START':
            # Start of a (possibly negated) conjunct, after `["` or after
            # ' and '. `sp` flag indicates whether we're in space-prefix mode.
            sp = state[1]
            tokens = set()
            # Optional: `not ` (or ` not ` if sp)
            seq = self.sp_not_seq if sp else self.not_seq
            if seq:
                tokens.add(seq[0])
            # Or jump directly into predicate trie
            pred_trie = self.sp_pred_trie if sp else self.pred_trie
            tokens |= set(pred_trie.next_tokens(0))
            return tokens

        if phase == 'NEG_SEQ':
            sp, idx = state[1], state[2]
            seq = self.sp_not_seq if sp else self.not_seq
            if idx < len(seq):
                return {seq[idx]}
            return set()

        if phase == 'EXPECT_PRED':
            sp, node = state[1], state[2]
            trie = self.sp_pred_trie if sp else self.pred_trie
            tokens = set(trie.next_tokens(node))
            if trie.is_leaf_node(node):
                # Leaf: canonical '(' only. The candidate tracker counts
                # canonical content tokens; a BPE-merged `(X` token would
                # desynchronise it, and β could then not bias the LM at
                # any later argument position.
                tokens.add(self.lparen_seq[0])
            return tokens

        if phase == 'LPAREN_SEQ':
            idx = state[2]
            if idx < len(self.lparen_seq):
                return {self.lparen_seq[idx]}
            return set()

        if phase == 'EXPECT_INSTANCE':
            pred, role, kind, node = state[1], state[2], state[3], state[4]
            trie = self._trie_for_kind(kind)
            viable = self._viable_instance_nodes(pred, role, kind=kind)
            raw = trie.next_tokens(node)
            if viable is not None:
                tokens = set()
                for tok in raw:
                    nxt = trie.next_node(node, tok)
                    if nxt is not None and nxt in viable:
                        tokens.add(tok)
            else:
                tokens = set(raw)
            if trie.is_leaf_node(node):
                arity = SD_PREDICATES.get(pred, 1)
                if role < arity - 1:
                    # Canonical `, ` only, for the same tracker-alignment
                    # reason as the canonical '(' in EXPECT_PRED.
                    tokens.add(self.obj_sep_seq[0])
                else:
                    tokens.add(self.rparen_seq[0])
            return tokens

        if phase == 'OBJ_SEP_SEQ':
            pred, idx = state[1], state[2]
            if idx < len(self.obj_sep_seq):
                return {self.obj_sep_seq[idx]}
            return set()

        if phase == 'RPAREN_SEQ':
            idx = state[1]
            if idx < len(self.rparen_seq):
                return {self.rparen_seq[idx]}
            return set()

        if phase == 'AFTER_PRED':
            # Canonical-only close boundary. Exactly two continuations,
            # each the first token of a fixed canonical sequence:
            #   `", "`   → SUBGOAL_SEP_SEQ  (next subgoal)
            #   `"]}`    → CLOSE_LIST_SEQ   (terminate)
            # ` and ` is not offered: the evaluator executes subgoals
            # sequentially, while `and` requires its atoms to hold in one
            # simultaneous state, which non-simultaneous atoms (e.g. several
            # `holds_lh` steps) cannot satisfy (evaluator property). One atom
            # per subgoal also lets β place `open(C)` in a strictly earlier
            # subgoal than its dependents.
            tokens = set()
            if self.subgoal_sep_seq:
                tokens.add(self.subgoal_sep_seq[0])
            if self.close_list_seq:
                tokens.add(self.close_list_seq[0])
            return tokens

        if phase == 'SUBGOAL_SEP_SEQ':
            idx = state[1]
            if idx < len(self.subgoal_sep_seq):
                return {self.subgoal_sep_seq[idx]}
            return set()

        if phase == 'CLOSE_LIST_SEQ':
            idx = state[1]
            if idx < len(self.close_list_seq):
                return {self.close_list_seq[idx]}
            return set()

        if phase == 'END':
            return set(self.eos_tokens)

        return set()

    def next_state(self, state: Any, token: int) -> Optional[Any]:
        token = int(token)
        phase = state[0]

        if phase == 'PREFIX':
            idx = state[1]
            if idx < len(self.prefix_seq):
                if token == self.prefix_seq[idx]:
                    new_idx = idx + 1
                    if new_idx >= len(self.prefix_seq):
                        return ('PRED_START', 0)  # sp=0 (no space-prefix at start of list)
                    return ('PREFIX', new_idx)
                return None
            return None

        if phase == 'PRED_START':
            sp = state[1]
            seq = self.sp_not_seq if sp else self.not_seq
            # Try negation
            if seq and token == seq[0]:
                if len(seq) == 1:
                    # 'not ' was a single token (includes trailing space).
                    # Next pred token is CANONICAL.
                    return ('EXPECT_PRED', 0, 0)
                return ('NEG_SEQ', sp, 1)
            # Otherwise: enter predicate trie
            pred_trie = self.sp_pred_trie if sp else self.pred_trie
            nxt = pred_trie.next_node(0, token)
            if nxt is not None:
                return ('EXPECT_PRED', sp, nxt)
            return None

        if phase == 'NEG_SEQ':
            sp, idx = state[1], state[2]
            seq = self.sp_not_seq if sp else self.not_seq
            if idx < len(seq) and token == seq[idx]:
                if idx + 1 >= len(seq):
                    return ('EXPECT_PRED', 0, 0)
                return ('NEG_SEQ', sp, idx + 1)
            # Canonical `not ` only, keeping generation aligned with the
            # candidate tracker.
            return None

        if phase == 'EXPECT_PRED':
            sp, node = state[1], state[2]
            trie = self.sp_pred_trie if sp else self.pred_trie
            # leaf transition: '(' canonical
            if trie.is_leaf_node(node) and token == self.lparen_seq[0]:
                pred = trie.get_leaf_label(node)
                if pred is None:
                    return None
                if len(self.lparen_seq) == 1:
                    return ('EXPECT_INSTANCE', pred, 0, self.TK_CANON, 0)
                return ('LPAREN_SEQ', pred, 1)
            # Continue predicate trie
            nxt = trie.next_node(node, token)
            if nxt is not None:
                return ('EXPECT_PRED', sp, nxt)
            return None

        if phase == 'LPAREN_SEQ':
            pred, idx = state[1], state[2]
            if idx < len(self.lparen_seq) and token == self.lparen_seq[idx]:
                if idx + 1 >= len(self.lparen_seq):
                    return ('EXPECT_INSTANCE', pred, 0, 0, 0)
                return ('LPAREN_SEQ', pred, idx + 1)
            return None

        if phase == 'EXPECT_INSTANCE':
            pred, role, kind, node = state[1], state[2], state[3], state[4]
            trie = self._trie_for_kind(kind)
            arity = SD_PREDICATES.get(pred, 1)
            # Leaf transitions: , (arg sep) or ) (close)
            if trie.is_leaf_node(node):
                if role < arity - 1:
                    # canonical ', ' fully consumed in one token → next is TK_CANON
                    if token == self.obj_sep_seq[0]:
                        if len(self.obj_sep_seq) == 1:
                            return ('EXPECT_INSTANCE', pred, role + 1,
                                    self.TK_CANON, 0)
                        return ('OBJ_SEP_SEQ', pred, 1, role)
                if role == arity - 1 and token == self.rparen_seq[0]:
                    if len(self.rparen_seq) == 1:
                        return ('AFTER_PRED',)
                    return ('RPAREN_SEQ', 1)
                # Only canonical ')' closes the atom; merged ')X' tokens are
                # not accepted, so the close always routes through
                # AFTER_PRED and the tracker stays aligned.
            # Continue instance trie
            nxt = trie.next_node(node, token)
            if nxt is not None:
                viable = self._viable_instance_nodes(pred, role, kind=kind)
                if viable is not None and nxt not in viable:
                    return None
                return ('EXPECT_INSTANCE', pred, role, kind, nxt)
            return None

        if phase == 'OBJ_SEP_SEQ':
            pred, idx, role = state[1], state[2], state[3]
            if idx < len(self.obj_sep_seq) and token == self.obj_sep_seq[idx]:
                if idx + 1 >= len(self.obj_sep_seq):
                    # Full ', ' consumed → next instance starts CANONICAL
                    return ('EXPECT_INSTANCE', pred, role + 1, self.TK_CANON, 0)
                return ('OBJ_SEP_SEQ', pred, idx + 1, role)
            # Canonical `, ` only (no `,` + ` X` split).
            return None

        if phase == 'RPAREN_SEQ':
            idx = state[1]
            if idx < len(self.rparen_seq) and token == self.rparen_seq[idx]:
                if idx + 1 >= len(self.rparen_seq):
                    return ('AFTER_PRED',)
                return ('RPAREN_SEQ', idx + 1)
            return None

        if phase == 'AFTER_PRED':
            # Canonical-only close boundary (mirrors allowed_tokens; see
            # there for why ` and ` is not offered). The separator includes
            # its trailing space, so the next pred is always canonical
            # (sp=0).
            if self.subgoal_sep_seq and token == self.subgoal_sep_seq[0]:
                if len(self.subgoal_sep_seq) == 1:
                    return ('PRED_START', 0)
                return ('SUBGOAL_SEP_SEQ', 1)
            if self.close_list_seq and token == self.close_list_seq[0]:
                if len(self.close_list_seq) == 1:
                    return ('END',)
                return ('CLOSE_LIST_SEQ', 1)
            return None

        if phase == 'SUBGOAL_SEP_SEQ':
            idx = state[1]
            if idx < len(self.subgoal_sep_seq) and token == self.subgoal_sep_seq[idx]:
                if idx + 1 >= len(self.subgoal_sep_seq):
                    return ('PRED_START', 0)
                return ('SUBGOAL_SEP_SEQ', idx + 1)
            return None

        if phase == 'CLOSE_LIST_SEQ':
            idx = state[1]
            if idx < len(self.close_list_seq) and token == self.close_list_seq[idx]:
                if idx + 1 >= len(self.close_list_seq):
                    return ('END',)
                return ('CLOSE_LIST_SEQ', idx + 1)
            return None

        if phase == 'END':
            if token in self.eos_tokens:
                return ('END',)
            return None

        return None

    def get_phase(self, state: Any) -> str:
        if isinstance(state, tuple) and state:
            return str(state[0])
        return str(state)

    # ---- Aliases matching BehaviorTwoLevelStateMachine's interface ----
    def initial(self) -> Any:
        return self.initial_state()

    def get_action(self, state: Any) -> Optional[str]:
        """Predicate-name accessor for τ / candidate tracker. SD's content
        cursor stores `pred` in slot 1 of state tuples for content phases."""
        if isinstance(state, tuple) and len(state) > 1:
            cand = state[1]
            if isinstance(cand, str):
                return cand
        return None

    def enumerate_states(self) -> List[Any]:
        if self._states is not None:
            return self._states
        self._states = [self.initial_state()]
        return self._states


# =============================================================================
# Factory
# =============================================================================

def parse_sd_objects(prompt: str) -> List[str]:
    """Object instance names (e.g. 'notebook.59') listed under "Relevant objects" in a BEH-SD prompt."""
    m = re.search(r"##\s*Relevant objects.*?\n(.*?)(?:\n##|\Z)", str(prompt or ""),
                  re.DOTALL | re.IGNORECASE)
    return sorted(set(re.findall(r"'name':\s*'([^']+)'", m.group(1)))) if m else []


def build_beh_sd_twolevel_state_machine(
    tokenizer, prompt: str, eos_token_id: int,
) -> BehaviorSDTwoLevelStateMachine:
    """Factory: parse SD prompt and build the two-level γ DFA."""
    instances = set(parse_sd_objects(prompt))
    # {'name','category'} pairs for the shared taxonomy membership layer.
    objects_meta = [
        {'name': n, 'category': c}
        for n, c in re.findall(
            r"\{'name':\s*'([^']+)',\s*'category':\s*'([^']+)'\}", prompt)
    ]
    # Also pull instances mentioned in init/target state blocks (e.g. agent_n_01.1
    # which isn't always listed in the relevant-objects section).
    for m in re.finditer(r'(\w+)\(([^)]+)\)', prompt):
        for arg in m.group(2).split(','):
            arg = arg.strip()
            if '.' in arg and re.match(r'^[a-zA-Z0-9_]+\.\d+$', arg):
                instances.add(arg)
    # Group instances by base
    objects: Dict[str, List[str]] = {}
    for inst in sorted(instances):
        base = inst.split('.', 1)[0]
        objects.setdefault(base, []).append(inst)
    # Predicates per base
    predicates_in_prompt: Dict[str, Set[str]] = {}
    for m in re.finditer(r'(\w+)\(([^)]+)\)', prompt):
        pred = m.group(1).lower()
        if pred not in SD_PREDICATES:
            continue
        for arg in m.group(2).split(','):
            arg = arg.strip()
            base = arg.split('.', 1)[0]
            predicates_in_prompt.setdefault(base, set()).add(pred)
    # Container-ROLE bases for the `open` evidence rule: arg-1 of any
    # `inside` atom in the prompt's Initial/Goal sections (including inside
    # quantifier bodies, where the container may be a category synset —
    # resolved to every scene base carrying that category).
    from ctrlg.beta.beh_sd_state_machine import parse_beh_sd_states
    cat_to_bases: Dict[str, Set[str]] = {}
    for om in objects_meta:
        c = str(om.get('category', '')).strip()
        if c:
            cat_to_bases.setdefault(c, set()).add(
                om['name'].split('.', 1)[0])

    _SD_LOC = {'inside', 'ontop', 'onfloor', 'nextto', 'under', 'touching'}

    def _resolve_bases(token, out):
        t = str(token)
        if t in cat_to_bases:               # category synset token
            out |= cat_to_bases[t]
        else:                               # instance name
            out.add(t.split('.', 1)[0])

    def _walk_ast(node, roles, repos):
        if not isinstance(node, (list, tuple)) or not node:
            return
        if node[0] == 'atom':
            _, pred, _neg, args = node
            if pred == 'inside' and len(args) >= 2:
                _resolve_bases(args[1], roles)
            if pred in _SD_LOC and len(args) >= 1:
                _resolve_bases(args[0], repos)
            return
        # ('and' | 'or', [children]); ('not', child); quantifiers carry their
        # variables before the body, which the walk skips (not tuples).
        # A negated atom still names C as a container / X as moved.
        children = node[1] if node[0] in ('and', 'or') else node[1:]
        for child in children:
            _walk_ast(child, roles, repos)

    init_ast, goal_ast = parse_beh_sd_states(prompt)
    container_role_bases: Set[str] = set()
    repositioned_bases: Set[str] = set()
    for entry in list(init_ast) + list(goal_ast):
        _walk_ast(entry, container_role_bases, set())
    # Repositioned bases come from the goal side only (the goal moves X).
    for entry in goal_ast:
        _walk_ast(entry, set(), repositioned_bases)

    return BehaviorSDTwoLevelStateMachine(
        tokenizer, objects, predicates_in_prompt, eos_token_id,
        objects_meta=objects_meta,
        container_role_bases=container_role_bases,
        repositioned_bases=repositioned_bases,
    )

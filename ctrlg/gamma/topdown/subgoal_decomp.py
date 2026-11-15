"""Top-down VirtualHome Subgoal Decomposition (SD) state machine.

Output format:
    {"output": ["PRED1(obj.id)", "PRED2(obj1.id, obj2.id)", ...]}

Each list element is exactly one predicate — no and/or compound steps.
Predicate names, arities, and object affordances are enforced.

Phase diagram:
    START → '{"output": ["' → PRED_START

    PRED_START → PRED_NAME_TRIE → '(' → ARG_BODY
    ARG_BODY (arity=0): ')' → AFTER_PRED
    ARG_BODY (arity>0): NAME_TRIE → '.' → ID_TRIE → ARG_SEP_OR_CLOSE
    ARG_SEP_OR_CLOSE:
        ', ' → next arg (NAME_TRIE)
        ')'  → AFTER_PRED

    AFTER_PRED → '"' → AFTER_STEP
    AFTER_STEP:
        ', "' → PRED_START   (next predicate)
        '"]}'  → END → EOS
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .base import CanonicalStateMachine
from .sd_predicate_defs import (
    SD_ALL_PREDICATES, SD_PREDICATE_ARITY,
    CROSS_ARG_PREDICATES,
    compute_sd_allowed_objects,
    compute_sd_arg1_allowed_objects,
)
from .trie_builder import TokenTrie, canonical_tokenization


class SDStateMachine(CanonicalStateMachine):
    """
    Top-down state machine for VirtualHome Subgoal Decomposition.

    Grammar: {"output": ["PRED(args)", ...]}
    Each list element is one predicate with typed, affordance-checked arguments.
    """

    def __init__(
        self,
        tokenizer,
        objects: Dict[str, List[str]],
        properties: Dict[str, Set[str]],
        categories: Dict[str, str],
        eos_token_id: int,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            objects: {name: [id1, id2, ...]} from prompt parsing
            properties: {name: {PROP1, PROP2, ...}} from prompt parsing
            categories: {name: category_string} e.g. "Rooms", "Characters", "placable_objects"
            eos_token_id: EOS token ID
        """
        import time as _t
        _t0 = _t.time()
        def _p(msg): print(f"  [SDStateMachine {_t.time()-_t0:.2f}s] {msg}", flush=True)

        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.objects = objects
        self.object_names = sorted(objects.keys())
        self.properties = properties
        self.categories = categories

        # Compute affordances
        _p("compute_sd_allowed_objects...")
        self.allowed_objects = compute_sd_allowed_objects(objects, properties, categories)

        # Compute viable predicates (predicates where all args have ≥1 valid object)
        _p("compute viable predicates...")
        self.viable_predicates = self._compute_viable_predicates()
        _p(f"  {len(self.viable_predicates)}/{len(SD_ALL_PREDICATES)} predicates viable")

        # Build tries
        _p("build predicate name trie...")
        self.pred_trie = self._build_pred_trie(tokenizer)

        _p("build object name tries...")
        self.name_trie, self.id_trie_by_name = self._build_object_tries(tokenizer)

        # Precompute token sequences for structural elements
        _p("precompute token seqs...")
        self._precompute_structural_seqs(tokenizer)

        # Precompute viable name nodes per (pred, arg_idx) for affordance filtering
        self._viable_name_cache: Dict[Tuple[str, int], Optional[Set[int]]] = {}

        # Cache for cross-arg context: (pred, arg_idx=1, arg0_name) → viable node set
        self._viable_name_cache_ctx: Dict[Tuple[str, int, str], Optional[Set[int]]] = {}

        # Precompute canonical token sequences for candidate tracking
        _p("precompute canonical token seqs...")
        self.pred_token_seqs: Dict[str, Tuple[int, ...]] = {}
        for pred in self.viable_predicates:
            seq = canonical_tokenization(tokenizer, pred)
            if seq:
                self.pred_token_seqs[pred] = seq

        self.name_token_seqs: Dict[str, Tuple[int, ...]] = {}
        for name in self.object_names:
            seq = canonical_tokenization(tokenizer, name)
            if seq:
                self.name_token_seqs[name] = seq

        self.id_token_seqs: Dict[Tuple[str, str], Tuple[int, ...]] = {}
        for name, ids in objects.items():
            for obj_id in ids:
                seq = canonical_tokenization(tokenizer, obj_id)
                if seq:
                    self.id_token_seqs[(name, obj_id)] = seq

        # Store scene_inventory for candidate tracking (same interface as VH)
        self.scene_inventory = objects

        # State cache
        self._states = None
        _p("done")

    def _compute_viable_predicates(self) -> Set[str]:
        """Predicates where all argument roles have ≥1 valid object."""
        viable = set()
        for pred in SD_ALL_PREDICATES:
            arity = SD_PREDICATE_ARITY[pred]
            if arity == 0:
                viable.add(pred)
                continue
            all_ok = True
            for arg_idx in range(arity):
                allowed = self.allowed_objects.get((pred, arg_idx), set())
                if not allowed:
                    all_ok = False
                    break
            if all_ok:
                viable.add(pred)
        return viable

    def _build_pred_trie(self, tokenizer) -> TokenTrie:
        """Build trie over viable predicate names."""
        trie = TokenTrie()
        for pred in sorted(self.viable_predicates):
            seq = canonical_tokenization(tokenizer, pred)
            if seq:
                trie.insert(seq, label=pred)
        return trie

    def _build_object_tries(self, tokenizer):
        """Build name trie and per-name ID tries."""
        name_trie = TokenTrie()
        for name in self.object_names:
            seq = canonical_tokenization(tokenizer, name)
            if seq:
                name_trie.insert(seq, label=name)

        id_trie_by_name: Dict[str, TokenTrie] = {}
        for name, ids in self.objects.items():
            id_trie = TokenTrie()
            for obj_id in ids:
                seq = canonical_tokenization(tokenizer, obj_id)
                if seq:
                    id_trie.insert(seq, label=obj_id)
            if id_trie.next_id > 1:  # has entries
                id_trie_by_name[name] = id_trie

        return name_trie, id_trie_by_name

    def _precompute_structural_seqs(self, tokenizer):
        """Precompute canonical token sequences and BPE alternatives for structural elements."""
        self.prefix_seq = canonical_tokenization(tokenizer, '{"output": ["')
        self.lparen_seq = canonical_tokenization(tokenizer, '(')
        self.rparen_seq = canonical_tokenization(tokenizer, ')')
        self.dot_seq = canonical_tokenization(tokenizer, '.')
        self.arg_sep_seq = canonical_tokenization(tokenizer, ', ')
        self.quote_seq = canonical_tokenization(tokenizer, '"')
        self.next_step_seq = canonical_tokenization(tokenizer, ', "')
        self.close_seq = canonical_tokenization(tokenizer, ']}')
        self.eos_tokens = {self.eos_token_id}

        

    def _name_allowed(self, pred: str, arg_idx: int, name: str, arg0_name_ctx=None) -> bool:
        """Is `name` an admissible object for (pred, arg_idx[, arg0])? Used to gate the
        name→id terminator at a trie LEAF: a leaf label may be a prefix of an allowed
        name (`wall` ⊂ `wall_clock`) while not being allowed itself, so the leaf check
        cannot rely on node viability alone."""
        if pred in CROSS_ARG_PREDICATES and arg_idx == 1 and arg0_name_ctx is not None:
            cache = getattr(self, '_arg1_allowed_cache', None)
            if cache is None:
                cache = self._arg1_allowed_cache = {}
            key = (pred, arg0_name_ctx)
            if key not in cache:
                cache[key] = compute_sd_arg1_allowed_objects(
                    pred, arg0_name_ctx, self.objects, self.properties, self.categories)
            allowed = cache[key]
        else:
            allowed = self.allowed_objects.get((pred, arg_idx))
        return allowed is None or name in allowed

    def _viable_name_nodes(self, pred: str, arg_idx: int) -> Optional[Set[int]]:
        """Get name trie nodes that lead to affordance-valid objects."""
        key = (pred, arg_idx)
        if key in self._viable_name_cache:
            return self._viable_name_cache[key]

        allowed_names = self.allowed_objects.get(key)
        if allowed_names is None:
            self._viable_name_cache[key] = None
            return None

        # Viable nodes = every node on the trie path of some allowed name.
        viable = set()
        for name in allowed_names:
            seq = self.name_token_seqs.get(name)
            if not seq:
                continue
            node = 0
            viable.add(node)
            for tok in seq:
                next_node = self.name_trie.next_node(node, tok)
                if next_node is not None:
                    viable.add(next_node)
                    node = next_node
                else:
                    break

        self._viable_name_cache[key] = viable if viable else None
        return viable if viable else None

    def _viable_name_nodes_with_context(
        self, pred: str, arg_idx: int, arg0_name: str
    ) -> Optional[Set[int]]:
        """
        Get name trie nodes for arg1 of a cross-arg predicate given arg0_name.

        Used by ONTOP and INSIDE where the set of valid arg1 objects depends on
        which arg0 was chosen.
        """
        cache_key = (pred, arg_idx, arg0_name)
        if cache_key in self._viable_name_cache_ctx:
            return self._viable_name_cache_ctx[cache_key]

        allowed_names = compute_sd_arg1_allowed_objects(
            pred, arg0_name, self.objects, self.properties, self.categories
        )

        if allowed_names is None:
            # No restriction — use the context-free cache
            result = self._viable_name_nodes(pred, arg_idx)
            self._viable_name_cache_ctx[cache_key] = result
            return result

        if not allowed_names:
            # Empty set = nothing valid. Return empty set, NOT None.
            # None means "no restriction"; empty set means "block everything".
            self._viable_name_cache_ctx[cache_key] = set()
            return set()

        viable: Set[int] = set()
        for name in allowed_names:
            seq = self.name_token_seqs.get(name)
            if not seq:
                continue
            node = 0
            viable.add(node)
            for tok in seq:
                next_node = self.name_trie.next_node(node, tok)
                if next_node is not None:
                    viable.add(next_node)
                    node = next_node
                else:
                    break

        result = viable if viable else None
        self._viable_name_cache_ctx[cache_key] = result
        return result

    # =========================================================================
    # State enumeration
    # =========================================================================

    def enumerate_states(self) -> List[Any]:
        if self._states is not None:
            return self._states

        states = []

        # Prefix states: generating '{"output": ["' token by token
        for i in range(len(self.prefix_seq) + 1):
            states.append(('PREFIX', i))
        # PREFIX at len(prefix_seq) = PRED_START (enter pred trie)

        # Predicate name trie states
        for node in self.pred_trie.all_nodes():
            states.append(('PRED_NAME', node))

        # After predicate name identified: parameterized by pred
        for pred in self.viable_predicates:
            arity = SD_PREDICATE_ARITY[pred]
            is_cross_arg = pred in CROSS_ARG_PREDICATES

            # '(' sequence (usually 1 token but handle multi)
            for i in range(len(self.lparen_seq)):
                states.append(('LPAREN_SEQ', pred, i))

            if arity == 0:
                # ')' sequence for nullary
                for i in range(len(self.rparen_seq)):
                    states.append(('RPAREN_SEQ', pred, i))
            else:
                for arg_idx in range(arity):
                    if is_cross_arg and arg_idx == 1:
                        # For cross-arg predicates, arg1 states are parameterized
                        # by the arg0 name that was chosen.
                        arg0_allowed = self.allowed_objects.get((pred, 0), set())
                        for arg0_name in sorted(arg0_allowed):
                            # ARG_NAME for arg1 carries arg0_name context
                            for node in self.name_trie.all_nodes():
                                states.append(('ARG_NAME', pred, 1, node, arg0_name))

                            # DOT_SEQ / ARG_ID for arg1 carry arg0_name context
                            for name in self.object_names:
                                for i in range(len(self.dot_seq)):
                                    states.append(('DOT_SEQ', pred, 1, name, i, arg0_name))

                                id_trie = self.id_trie_by_name.get(name)
                                if id_trie:
                                    for id_node in id_trie.all_nodes():
                                        states.append(('ARG_ID', pred, 1, name, id_node, arg0_name))
                    else:
                        # Normal (context-free) arg states
                        for node in self.name_trie.all_nodes():
                            states.append(('ARG_NAME', pred, arg_idx, node))

                        for name in self.object_names:
                            for i in range(len(self.dot_seq)):
                                states.append(('DOT_SEQ', pred, arg_idx, name, i))

                            id_trie = self.id_trie_by_name.get(name)
                            if id_trie:
                                for id_node in id_trie.all_nodes():
                                    states.append(('ARG_ID', pred, arg_idx, name, id_node))

                    # ', ' separator sequence between args (multi-token)
                    if arg_idx < arity - 1:
                        if is_cross_arg and arg_idx == 0:
                            # ARG_SEP_SEQ carries arg0_name for cross-arg predicates
                            arg0_allowed = self.allowed_objects.get((pred, 0), set())
                            for arg0_name in sorted(arg0_allowed):
                                for i in range(len(self.arg_sep_seq)):
                                    states.append(('ARG_SEP_SEQ', pred, arg_idx, i, arg0_name))
                        else:
                            for i in range(len(self.arg_sep_seq)):
                                states.append(('ARG_SEP_SEQ', pred, arg_idx, i))

                # ')' sequence after last arg
                for i in range(len(self.rparen_seq)):
                    states.append(('RPAREN_SEQ', pred, i))

        # '"' sequence after predicate content
        for i in range(len(self.quote_seq)):
            states.append(('CLOSE_QUOTE_SEQ', i))

        # After step closed: choice point
        states.append(('AFTER_STEP',))

        # ', "' sequence for next step (multi-token)
        for i in range(len(self.next_step_seq)):
            states.append(('NEXT_STEP_SEQ', i))

        # ']}' sequence for close (may be single token)
        for i in range(len(self.close_seq)):
            states.append(('CLOSE_SEQ', i))

        states.append(('END',))

        self._states = states
        return states

    def initial_state(self) -> Any:
        return ('PREFIX', 0)

    def is_accept(self, state: Any) -> bool:
        return isinstance(state, tuple) and state[0] == 'END'

    # =========================================================================
    # Transitions
    # =========================================================================

    def _seq_allowed(self, seq, idx):
        """Return allowed tokens for position idx in a fixed sequence."""
        if idx < len(seq):
            return {seq[idx]}
        return set()

    def allowed_tokens(self, state: Any) -> Set[int]:
        phase = state[0]

        if phase == 'PREFIX':
            idx = state[1]
            if idx < len(self.prefix_seq):
                return {self.prefix_seq[idx]}
            # At PRED_START: enter predicate name trie
            return self.pred_trie.next_tokens(0)

        if phase == 'PRED_NAME':
            node = state[1]
            tokens = set(self.pred_trie.next_tokens(node))
            if self.pred_trie.is_leaf_node(node):
                # Pred complete — allow first token of '('
                tokens.add(self.lparen_seq[0])
            return tokens

        if phase == 'LPAREN_SEQ':
            return self._seq_allowed(self.lparen_seq, state[2])

        if phase == 'RPAREN_SEQ':
            return self._seq_allowed(self.rparen_seq, state[2])

        if phase == 'ARG_NAME':
            pred, arg_idx, node = state[1], state[2], state[3]
            # Cross-arg predicates carry arg0_name in state[4] for arg_idx=1
            if pred in CROSS_ARG_PREDICATES and arg_idx == 1:
                arg0_name = state[4]
                viable = self._viable_name_nodes_with_context(pred, arg_idx, arg0_name)
            else:
                viable = self._viable_name_nodes(pred, arg_idx)
            raw_tokens = self.name_trie.next_tokens(node)
            if viable is not None:
                tokens = set()
                for tok in raw_tokens:
                    next_n = self.name_trie.next_node(node, tok)
                    if next_n is not None and next_n in viable:
                        tokens.add(tok)
            else:
                tokens = set(raw_tokens)
            if self.name_trie.is_leaf_node(node):
                name = self.name_trie.get_leaf_label(node)
                arg0_ctx = state[4] if (pred in CROSS_ARG_PREDICATES and arg_idx == 1) else None
                if (name and name in self.id_trie_by_name
                        and self._name_allowed(pred, arg_idx, name, arg0_ctx)):
                    tokens.add(self.dot_seq[0])
            return tokens

        if phase == 'DOT_SEQ':
            # seq_idx is state[4] in both layouts:
            # cross-arg: (DOT_SEQ, pred, arg_idx, name, seq_idx, arg0_name)
            # normal:    (DOT_SEQ, pred, arg_idx, name, seq_idx)
            return self._seq_allowed(self.dot_seq, state[4])

        if phase == 'ARG_ID':
            # Normal layout: ('ARG_ID', pred, arg_idx, name, id_node)
            # Cross-arg layout: ('ARG_ID', pred, arg_idx, name, id_node, arg0_name)
            pred, arg_idx, name, id_node = state[1], state[2], state[3], state[4]
            id_trie = self.id_trie_by_name.get(name)
            if not id_trie:
                return set()
            tokens = set(id_trie.next_tokens(id_node))
            if id_trie.is_leaf_node(id_node):
                arity = SD_PREDICATE_ARITY[pred]
                if arg_idx < arity - 1:
                    tokens.add(self.arg_sep_seq[0])  # first token of ', '
                else:
                    tokens.add(self.rparen_seq[0])   # first token of ')'
            return tokens

        if phase == 'ARG_SEP_SEQ':
            # Normal layout: ('ARG_SEP_SEQ', pred, arg_idx, seq_idx)
            # Cross-arg layout: ('ARG_SEP_SEQ', pred, arg_idx, seq_idx, arg0_name)
            return self._seq_allowed(self.arg_sep_seq, state[3])

        if phase == 'CLOSE_QUOTE_SEQ':
            return self._seq_allowed(self.quote_seq, state[1])

        if phase == 'AFTER_STEP':
            tokens = set()
            tokens.add(self.next_step_seq[0])  # first of ', "'
            tokens.add(self.close_seq[0])       # first of ']}'
            return tokens

        if phase == 'NEXT_STEP_SEQ':
            return self._seq_allowed(self.next_step_seq, state[1])

        if phase == 'CLOSE_SEQ':
            return self._seq_allowed(self.close_seq, state[1])

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
                    return ('PREFIX', idx + 1)
                return None
            # PRED_START: enter pred name trie
            next_node = self.pred_trie.next_node(0, token)
            if next_node is not None:
                return ('PRED_NAME', next_node)
            return None

        if phase == 'PRED_NAME':
            node = state[1]
            # Check if pred name complete and token is start of '('
            if self.pred_trie.is_leaf_node(node) and token == self.lparen_seq[0]:
                pred = self.pred_trie.get_leaf_label(node)
                if pred:
                    if len(self.lparen_seq) == 1:
                        # '(' consumed — enter args
                        arity = SD_PREDICATE_ARITY.get(pred, 1)
                        if arity == 0:
                            return ('RPAREN_SEQ', pred, 0)
                        return ('ARG_NAME', pred, 0, 0)
                    return ('LPAREN_SEQ', pred, 1)
            # Continue trie
            next_node = self.pred_trie.next_node(node, token)
            if next_node is not None:
                return ('PRED_NAME', next_node)
            return None

        if phase == 'LPAREN_SEQ':
            pred, idx = state[1], state[2]
            if idx < len(self.lparen_seq) and token == self.lparen_seq[idx]:
                if idx + 1 >= len(self.lparen_seq):
                    arity = SD_PREDICATE_ARITY.get(pred, 1)
                    if arity == 0:
                        return ('RPAREN_SEQ', pred, 0)
                    return ('ARG_NAME', pred, 0, 0)
                return ('LPAREN_SEQ', pred, idx + 1)
            return None

        if phase == 'RPAREN_SEQ':
            pred, idx = state[1], state[2]
            if idx < len(self.rparen_seq) and token == self.rparen_seq[idx]:
                if idx + 1 >= len(self.rparen_seq):
                    return ('CLOSE_QUOTE_SEQ', 0)
                return ('RPAREN_SEQ', pred, idx + 1)
            return None

        if phase == 'ARG_NAME':
            pred, arg_idx, node = state[1], state[2], state[3]
            is_cross_arg = pred in CROSS_ARG_PREDICATES
            # For cross-arg predicates at arg_idx=1, state carries arg0_name
            arg0_name_ctx = state[4] if (is_cross_arg and arg_idx == 1) else None

            # Check dot transition (name complete)
            if self.name_trie.is_leaf_node(node) and token == self.dot_seq[0]:
                name = self.name_trie.get_leaf_label(node)
                if (name and name in self.id_trie_by_name
                        and self._name_allowed(pred, arg_idx, name, arg0_name_ctx)):
                    if len(self.dot_seq) == 1:
                        if arg0_name_ctx is not None:
                            return ('ARG_ID', pred, arg_idx, name, 0, arg0_name_ctx)
                        return ('ARG_ID', pred, arg_idx, name, 0)
                    if arg0_name_ctx is not None:
                        return ('DOT_SEQ', pred, arg_idx, name, 1, arg0_name_ctx)
                    return ('DOT_SEQ', pred, arg_idx, name, 1)
            # Continue name trie
            next_node = self.name_trie.next_node(node, token)
            if next_node is not None:
                if arg0_name_ctx is not None:
                    viable = self._viable_name_nodes_with_context(pred, arg_idx, arg0_name_ctx)
                else:
                    viable = self._viable_name_nodes(pred, arg_idx)
                if viable is not None and next_node not in viable:
                    return None
                if arg0_name_ctx is not None:
                    return ('ARG_NAME', pred, arg_idx, next_node, arg0_name_ctx)
                return ('ARG_NAME', pred, arg_idx, next_node)
            return None

        if phase == 'DOT_SEQ':
            pred, arg_idx, name, idx = state[1], state[2], state[3], state[4]
            # Cross-arg states have arg0_name at state[5]
            is_cross_arg = pred in CROSS_ARG_PREDICATES
            arg0_name_ctx = state[5] if (is_cross_arg and arg_idx == 1 and len(state) > 5) else None
            if idx < len(self.dot_seq) and token == self.dot_seq[idx]:
                if idx + 1 >= len(self.dot_seq):
                    if arg0_name_ctx is not None:
                        return ('ARG_ID', pred, arg_idx, name, 0, arg0_name_ctx)
                    return ('ARG_ID', pred, arg_idx, name, 0)
                if arg0_name_ctx is not None:
                    return ('DOT_SEQ', pred, arg_idx, name, idx + 1, arg0_name_ctx)
                return ('DOT_SEQ', pred, arg_idx, name, idx + 1)
            return None

        if phase == 'ARG_ID':
            pred, arg_idx, name, id_node = state[1], state[2], state[3], state[4]
            # Cross-arg states: arg_idx=0 will produce ARG_SEP_SEQ with arg0_name=name;
            #                   arg_idx=1 carries arg0_name at state[5]
            is_cross_arg = pred in CROSS_ARG_PREDICATES
            arg0_name_ctx = state[5] if (is_cross_arg and arg_idx == 1 and len(state) > 5) else None

            id_trie = self.id_trie_by_name.get(name)
            if not id_trie:
                return None
            arity = SD_PREDICATE_ARITY[pred]

            if id_trie.is_leaf_node(id_node):
                if arg_idx < arity - 1 and token == self.arg_sep_seq[0]:
                    if is_cross_arg and arg_idx == 0:
                        # Carry the chosen arg0 name into the separator sequence
                        if len(self.arg_sep_seq) == 1:
                            return ('ARG_NAME', pred, 1, 0, name)
                        return ('ARG_SEP_SEQ', pred, arg_idx, 1, name)
                    # Normal (non-cross-arg) separator
                    if len(self.arg_sep_seq) == 1:
                        return ('ARG_NAME', pred, arg_idx + 1, 0)
                    return ('ARG_SEP_SEQ', pred, arg_idx, 1)
                if arg_idx == arity - 1 and token == self.rparen_seq[0]:
                    if len(self.rparen_seq) == 1:
                        return ('CLOSE_QUOTE_SEQ', 0)
                    return ('RPAREN_SEQ', pred, 1)
            # Continue ID trie
            next_id_node = id_trie.next_node(id_node, token)
            if next_id_node is not None:
                if arg0_name_ctx is not None:
                    return ('ARG_ID', pred, arg_idx, name, next_id_node, arg0_name_ctx)
                return ('ARG_ID', pred, arg_idx, name, next_id_node)
            return None

        if phase == 'ARG_SEP_SEQ':
            pred, arg_idx, idx = state[1], state[2], state[3]
            # Cross-arg states carry arg0_name at state[4]
            is_cross_arg = pred in CROSS_ARG_PREDICATES
            arg0_name_ctx = state[4] if (is_cross_arg and arg_idx == 0 and len(state) > 4) else None
            if idx < len(self.arg_sep_seq) and token == self.arg_sep_seq[idx]:
                if idx + 1 >= len(self.arg_sep_seq):
                    if arg0_name_ctx is not None:
                        return ('ARG_NAME', pred, arg_idx + 1, 0, arg0_name_ctx)
                    return ('ARG_NAME', pred, arg_idx + 1, 0)
                if arg0_name_ctx is not None:
                    return ('ARG_SEP_SEQ', pred, arg_idx, idx + 1, arg0_name_ctx)
                return ('ARG_SEP_SEQ', pred, arg_idx, idx + 1)
            return None

        if phase == 'CLOSE_QUOTE_SEQ':
            idx = state[1]
            if idx < len(self.quote_seq) and token == self.quote_seq[idx]:
                if idx + 1 >= len(self.quote_seq):
                    return ('AFTER_STEP',)
                return ('CLOSE_QUOTE_SEQ', idx + 1)
            return None

        if phase == 'AFTER_STEP':
            if token == self.next_step_seq[0]:
                if len(self.next_step_seq) == 1:
                    return ('PREFIX', len(self.prefix_seq))  # PRED_START
                return ('NEXT_STEP_SEQ', 1)
            if token == self.close_seq[0]:
                if len(self.close_seq) == 1:
                    return ('END',)
                return ('CLOSE_SEQ', 1)
            return None

        if phase == 'NEXT_STEP_SEQ':
            idx = state[1]
            if idx < len(self.next_step_seq) and token == self.next_step_seq[idx]:
                if idx + 1 >= len(self.next_step_seq):
                    return ('PREFIX', len(self.prefix_seq))  # PRED_START
                return ('NEXT_STEP_SEQ', idx + 1)
            return None

        if phase == 'CLOSE_SEQ':
            idx = state[1]
            if idx < len(self.close_seq) and token == self.close_seq[idx]:
                if idx + 1 >= len(self.close_seq):
                    return ('END',)
                return ('CLOSE_SEQ', idx + 1)
            return None

        if phase == 'END':
            if token in self.eos_tokens:
                return ('END',)
            return None

        return None

    # =========================================================================
    # Decoder API (same interface as VHStateMachine)
    # =========================================================================

    def initial(self) -> Any:
        return self.initial_state()

    DEAD = ('DEAD',)

    def step(self, state: Any, token: int) -> Any:
        next_s = self.next_state(state, token)
        return next_s if next_s is not None else self.DEAD

    def get_phase(self, state: Any) -> str:
        if isinstance(state, tuple) and state:
            return str(state[0])
        return str(state)


def parse_sd_prompt(prompt_text: str):
    """
    Parse an SD prompt to extract objects, properties, categories, and goals.

    Args:
        prompt_text: The llm_prompt string from helm_prompt.json

    Returns:
        objects: {name: [id1, id2, ...]}
        properties: {name: {PROP1, ...}}
        categories: {name: category_string}
        initial_states: list of predicate strings
        goal_states: list of predicate strings
        action_goals: list of action predicate strings
    """
    import re

    objects = {}
    properties = {}
    categories = {}
    initial_states = []
    goal_states = []
    action_goals = []

    lines = prompt_text.strip().split('\n')

    section = None
    goal_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect sections
        if '## Relevant Objects' in line:
            section = 'objects'
            continue
        elif '## Initial States' in line:
            section = 'initial'
            continue
        elif '## Goal States' in line:
            section = 'goals'
            goal_section = None
            continue
        elif '## Necessity' in line or '## Output:' in line:
            section = None
            continue

        if section == 'objects':
            # Parse: | name.id | category | [props] |  (names may contain '-', e.g. shoe-shine_kit)
            m = re.match(r'\|\s*([\w-]+)\.(\d+)\s*\|\s*(\w+)\s*\|\s*\[(.*?)\]\s*\|', line)
            if m:
                name = m.group(1)
                obj_id = m.group(2)
                category = m.group(3)
                props_str = m.group(4).strip()

                if name not in objects:
                    objects[name] = []
                objects[name].append(obj_id)

                categories[name] = category

                if props_str:
                    prop_set = set()
                    for p in props_str.split(','):
                        p = p.strip().strip("'").strip('"')
                        if p:
                            prop_set.add(p)
                    if name not in properties:
                        properties[name] = set()
                    properties[name] |= prop_set

        elif section == 'initial':
            # Parse predicate lines like CLOSED(washing_machine.1001)
            m = re.match(r'(\w+)\(', line)
            if m:
                initial_states.append(line)

        elif section == 'goals':
            if '[States]' in line:
                goal_section = 'states'
                continue
            elif '[Actions Must Include]' in line:
                goal_section = 'actions'
                continue
            elif line == 'None':
                continue

            if goal_section == 'states':
                m = re.match(r'(\w+)\(', line)
                if m:
                    goal_states.append(line)
            elif goal_section == 'actions':
                action_goals.append(line)

    return objects, properties, categories, initial_states, goal_states, action_goals


def enumerate_gamma_valid_predicates(
    sd_sm: SDStateMachine,
    orig_id_to_canonical: Dict[str, int],
):
    """
    Enumerate all γ-valid (predicate, canonical_ids) tuples.

    This is the SD equivalent of enumerate_gamma_valid_actions for AS.
    Each entry is a (pred_name, canonical_id_tuple) pair that the γ DFA would accept.

    Args:
        sd_sm: built SDStateMachine
        orig_id_to_canonical: {raw_id_str: canonical_int}

    Returns:
        List of (pred_name: str, cids: Tuple[int, ...]) tuples
    """
    result = []

    for pred in sorted(sd_sm.viable_predicates):
        arity = SD_PREDICATE_ARITY[pred]

        if arity == 0:
            result.append((pred, ()))
            continue

        # Get allowed object names per arg
        allowed_per_arg = []
        for arg_idx in range(arity):
            allowed_names = sd_sm.allowed_objects.get((pred, arg_idx), set())
            # Expand to (name, raw_id, canonical_id) triples
            expanded = []
            for name in sorted(allowed_names):
                for raw_id in sorted(sd_sm.objects.get(name, []), key=int):
                    cid = orig_id_to_canonical.get(str(raw_id))
                    if cid is not None:
                        expanded.append((name, raw_id, cid))
            allowed_per_arg.append(expanded)

        if arity == 1:
            for name, raw_id, cid in allowed_per_arg[0]:
                result.append((pred, (cid,)))
        elif arity == 2:
            if pred in CROSS_ARG_PREDICATES:
                # For cross-arg predicates, filter arg1 based on the chosen arg0.
                for name0, rid0, cid0 in allowed_per_arg[0]:
                    allowed_arg1 = compute_sd_arg1_allowed_objects(
                        pred, name0,
                        sd_sm.objects, sd_sm.properties, sd_sm.categories,
                    )
                    for name1, rid1, cid1 in allowed_per_arg[1]:
                        if allowed_arg1 is not None and name1 not in allowed_arg1:
                            continue
                        result.append((pred, (cid0, cid1)))
            else:
                for name0, rid0, cid0 in allowed_per_arg[0]:
                    for name1, rid1, cid1 in allowed_per_arg[1]:
                        # Self-relations are not filtered here; β rejects them.
                        result.append((pred, (cid0, cid1)))

    return result


def build_sd_state_machine(tokenizer, prompt_text: str, eos_token_id: int) -> 'SDStateMachine':
    """Factory: parse prompt and build SD state machine."""
    objects, properties, categories, initial_states, goal_states, action_goals = \
        parse_sd_prompt(prompt_text)
    return SDStateMachine(tokenizer, objects, properties, categories, eos_token_id)

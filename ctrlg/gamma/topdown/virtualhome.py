"""Top-down VirtualHome Action Sequencing state machine.

This module provides a clean state machine implementation that:
- Uses hardcoded action definitions (no action parsing from prompt)
- Parses only objects from prompts
- Computes token sets directly from tokenizer
- Builds tries directly without implicit DFA
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .base import CanonicalStateMachine
from .action_defs import VH_ACTIONS, VH_ACTION_ARITY, compute_vh_affordances
from .token_utils import get_vh_token_sets
from .trie_builder import (
    TokenTrie, build_vh_tries, compute_viable_name_nodes, get_canonical_token_seqs,
    get_name_token_seqs, get_id_token_seqs,
)


class VHStateMachine(CanonicalStateMachine):
    """
    Top-down state machine for VirtualHome Action Sequencing.

    Grammar: {"ACTION": ["name", "id", ...], ...}

    States carry explicit action names (not class_id) to support:
    - Per-action meta-token lookup in the two-level decoder
    - Exact action identification at any generation step
    """

    WS_PHASES = {
        'KEY_QUOTE_CLOSE', 'EXPECT_COLON',
        'ARRAY_OPEN', 'ARRAY_BODY', 'VALUE_QUOTE_OPEN_NAME', 'VALUE_QUOTE_CLOSE_NAME',
        'NAME_ID_SEP', 'VALUE_QUOTE_OPEN_ID', 'VALUE_QUOTE_CLOSE_ID',
        'PAIRS_TAIL', 'END', 'EMPTY_ARRAY_NEED_WS'
    }

    def __init__(
        self,
        tokenizer,
        objects: Dict[str, List[str]],
        properties: Dict[str, Set[str]],
        eos_token_id: int,
        max_ws_tokens: int = 5,
    ):
        """
        Initialize VH state machine from tokenizer and parsed objects.

        Args:
            tokenizer: HuggingFace tokenizer
            objects: {name: [id1, id2, ...]} from prompt parsing
            properties: {name: {PROP1, PROP2, ...}} from prompt parsing
            eos_token_id: EOS token ID
            max_ws_tokens: Maximum consecutive whitespace tokens
        """
        import time as _t
        _t0 = _t.time()
        def _p(msg): print(f"  [VHStateMachine {_t.time()-_t0:.2f}s] {msg}", flush=True)

        # Store config
        self.max_ws_tokens = max_ws_tokens

        # Get token sets from tokenizer
        _p("get_vh_token_sets...")
        token_sets = get_vh_token_sets(tokenizer, eos_token_id)
        self.WS_TOKENS = token_sets.WS_TOKENS
        self.LBRACE_START_ALTS = token_sets.LBRACE_START_ALTS
        self.LBRACE_QUOTE_START_ALTS = token_sets.LBRACE_QUOTE_START_ALTS
        self.QUOTE_ALTS = token_sets.QUOTE_ALTS
        self.QUOTE_COLON_ALTS = token_sets.QUOTE_COLON_ALTS
        self.COLON_ALTS = token_sets.COLON_ALTS
        self.LBRACK_ALTS = token_sets.LBRACK_ALTS
        self.LBRACK_QUOTE_ALTS = token_sets.LBRACK_QUOTE_ALTS
        self.RBRACK_ALTS = token_sets.RBRACK_ALTS
        self.RBRACK_LEADING_WS_ALTS = token_sets.RBRACK_LEADING_WS_ALTS
        self.COMMA_ALTS = token_sets.COMMA_ALTS
        self.RBRACE_ALTS = token_sets.RBRACE_ALTS
        self.QUOTE_COMMA_ALTS = token_sets.QUOTE_COMMA_ALTS
        self.QUOTE_RBRACK_ALTS = token_sets.QUOTE_RBRACK_ALTS
        self.QUOTE_RBRACK_COMMA_ALTS = token_sets.QUOTE_RBRACK_COMMA_ALTS
        self.QUOTE_RBRACK_RBRACE_ALTS = token_sets.QUOTE_RBRACK_RBRACE_ALTS
        self.eos_token_ids = token_sets.eos_token_ids

        # Use hardcoded action definitions
        self.action_arity = dict(VH_ACTION_ARITY)
        self.actions = sorted(VH_ACTIONS)

        # Build tries from hardcoded actions and parsed objects
        _p("build_vh_tries...")
        self.action_trie, self.name_trie, self.id_trie_by_name = build_vh_tries(
            tokenizer, VH_ACTIONS, objects
        )

        # Get all object names
        self.object_names = sorted(objects.keys())

        # Compute affordances from hardcoded rules + parsed properties
        _p("compute_vh_affordances...")
        self.allowed_objects_per_role = compute_vh_affordances(properties)

        # Precompute viable actions (actions where all roles have ≥1 valid object)
        _p("_compute_viable_actions...")
        self.viable_actions = self._compute_viable_actions()

        # Cache viable name nodes per (action, role)
        self._viable_cache: Dict[Tuple[str, int], Optional[Set[int]]] = {}

        # Build state cache
        self._states = None

        # Store objects and scene_inventory for candidate tracking
        self.scene_inventory = objects  # {name: [id1, id2, ...]}

        # Build canonical token sequences for candidate tracking
        # These are used by VirtualHomeCandidateTracker._build_candidates()
        _p("get_canonical_token_seqs...")
        self.canonical_token_seq = get_canonical_token_seqs(tokenizer, VH_ACTIONS)
        _p("get_name_token_seqs...")
        self.name_token_seqs_by_name = get_name_token_seqs(tokenizer, objects.keys())
        _p("get_id_token_seqs...")
        self.id_token_seqs_by_name_and_id = get_id_token_seqs(tokenizer, objects)
        _p("done")

    def _compute_viable_actions(self) -> Set[str]:
        """
        Compute actions where ALL roles have at least one valid object.

        An action is NOT viable if any of its roles has an empty allowed set.
        This prevents selecting actions like PUTBACK when no SURFACES exist.
        """
        viable = set()

        for action in self.actions:
            arity = self.action_arity.get(action, 1)

            if arity == 0:
                # Arity 0 actions (STANDUP, SLEEP) are always viable
                viable.add(action)
                continue

            # Check all roles
            all_roles_ok = True
            for role in range(arity):
                key = (action, role)
                allowed = self.allowed_objects_per_role.get(key)

                # If key not in allowed_objects_per_role, no restriction (all objects allowed)
                if allowed is None:
                    continue

                # If explicitly empty set, this role has no valid objects
                if len(allowed) == 0:
                    all_roles_ok = False
                    break

            if all_roles_ok:
                viable.add(action)

        return viable

    def _get_trie_nodes(self, trie: TokenTrie) -> List[int]:
        """Get all node IDs in a trie."""
        return sorted(trie.all_nodes())

    def _viable_name_nodes(self, action: str, role: int) -> Optional[Set[int]]:
        """Get viable name trie nodes for (action, role) based on affordance."""
        key = (str(action).upper(), int(role))
        if key in self._viable_cache:
            return self._viable_cache[key]

        # Get allowed objects for this (action, role)
        allowed = self.allowed_objects_per_role.get(key)

        # Compute viable nodes
        viable = compute_viable_name_nodes(self.name_trie, self.id_trie_by_name, allowed)
        self._viable_cache[key] = viable
        return viable

    def enumerate_states(self) -> List[Any]:
        """Enumerate all canonical states."""
        if self._states is not None:
            return self._states

        states = []

        # Fixed states
        states.append(('START',))
        states.append(('BLOCK_BOUNDARY', True))   # first action
        states.append(('BLOCK_BOUNDARY', False))  # subsequent actions
        states.append(('EXPECT_ACTION', -1))      # quote-pending before trie
        states.append(('END',))

        # EXPECT_ACTION states (trie traversal)
        for node in self._get_trie_nodes(self.action_trie):
            states.append(('EXPECT_ACTION', node))

        # Action-parameterized states (explicit action name)
        for action in self.actions:
            arity = self.action_arity.get(action, 1)

            states.append(('KEY_QUOTE_CLOSE', action))
            states.append(('EXPECT_COLON', action))
            states.append(('ARRAY_OPEN', action))

            if arity == 0:
                states.append(('EMPTY_ARRAY_NEED_WS', action))
                states.append(('PAIRS_TAIL', action, 0))
            else:
                for count in range(arity + 1):
                    states.append(('ARRAY_BODY', action, count))
                    states.append(('PAIRS_TAIL', action, count))
                    states.append(('VALUE_QUOTE_CLOSE_ID', action, count))

                    if count < arity:
                        states.append(('VALUE_QUOTE_OPEN_NAME', action, count))
                        for node in self._get_trie_nodes(self.name_trie):
                            states.append(('EXPECT_NAME', action, count, node))

        # Name-parameterized states (for ID lookup)
        for name in self.object_names:
            for action in self.actions:
                arity = self.action_arity.get(action, 1)
                for count in range(arity):
                    states.append(('VALUE_QUOTE_CLOSE_NAME', name, action, count))
                    states.append(('NAME_ID_SEP', name, action, count))
                    states.append(('VALUE_QUOTE_OPEN_ID', name, action, count))

                    id_trie = self.id_trie_by_name.get(name)
                    if id_trie:
                        for id_node in self._get_trie_nodes(id_trie):
                            states.append(('EXPECT_ID', name, action, count, id_node))

        # Whitespace wrapper states
        ws_states = []
        for state in states:
            phase = state[0]
            if phase in self.WS_PHASES:
                for ws_count in range(1, self.max_ws_tokens + 1):
                    ws_states.append(('_WS', state, ws_count))
        states.extend(ws_states)

        self._states = states
        return states

    def initial_state(self) -> Any:
        return ('START',)

    def initial(self) -> Any:
        """Alias for initial_state() - for tau/candidate tracker compatibility."""
        return self.initial_state()

    def is_accept(self, state: Any) -> bool:
        if isinstance(state, tuple) and state and state[0] == '_WS':
            state = state[1]
        return isinstance(state, tuple) and state[0] == 'END'

    # DEAD state for invalid transitions
    DEAD = ('DEAD',)

    def step(self, state: Any, token: int) -> Any:
        """
        Step through DFA with token, returning next state or DEAD.

        Alias for next_state() with DEAD fallback - for tau/candidate tracker compatibility.
        """
        next_s = self.next_state(state, token)
        if next_s is None:
            return self.DEAD
        return next_s

    def _unwrap_ws(self, state: Any) -> Tuple[Any, int]:
        if isinstance(state, tuple) and state and state[0] == '_WS':
            return state[1], state[2]
        return state, 0

    def _get_ws(self, state: Any) -> Set[int]:
        base, ws_run = self._unwrap_ws(state)
        if base[0] not in self.WS_PHASES:
            return set()
        if ws_run >= self.max_ws_tokens:
            return set()
        return set(self.WS_TOKENS)

    # =========================================================================
    # TWO-LEVEL DECODER API
    # =========================================================================

    def get_phase(self, state: Any) -> str:
        """Get phase string from state - for tau/candidate tracker compatibility."""
        base, _ = self._unwrap_ws(state)
        if isinstance(base, tuple) and base:
            return str(base[0])
        return str(base)

    def get_action(self, state: Any) -> Optional[str]:
        """Alias for get_current_action - for tau/candidate tracker compatibility."""
        return self.get_current_action(state)

    def get_vh_expect_id_fields(self, state: Any) -> Tuple[Optional[str], Optional[int]]:
        """
        Get (name, node) from EXPECT_ID state.

        For tau parser ID completion tracking.
        Returns (None, None) if not in EXPECT_ID phase.
        """
        base, _ = self._unwrap_ws(state)
        if isinstance(base, tuple) and len(base) >= 5 and base[0] == 'EXPECT_ID':
            # ('EXPECT_ID', name, action, count, node)
            return (base[1], base[4])
        return (None, None)

    def get_current_action(self, state: Any) -> Optional[str]:
        """Return the committed action name, or None if still selecting."""
        base, _ = self._unwrap_ws(state)
        phase = base[0]

        if phase == 'EXPECT_ACTION':
            return None
        elif phase in ('KEY_QUOTE_CLOSE', 'EXPECT_COLON', 'ARRAY_OPEN',
                       'EMPTY_ARRAY_NEED_WS', 'ARRAY_BODY', 'VALUE_QUOTE_OPEN_NAME',
                       'PAIRS_TAIL'):
            return base[1]
        elif phase == 'EXPECT_NAME':
            return base[1]
        elif phase in ('VALUE_QUOTE_CLOSE_NAME', 'NAME_ID_SEP',
                       'VALUE_QUOTE_OPEN_ID', 'EXPECT_ID'):
            return base[2]
        elif phase == 'VALUE_QUOTE_CLOSE_ID':
            return base[1]
        elif phase in ('START', 'BLOCK_BOUNDARY', 'END'):
            return None
        return None

    def _get_reachable_actions(self, node: int) -> List[str]:
        """Get all actions reachable from a trie node (unfiltered)."""
        reachable = []

        if self.action_trie.is_leaf_node(node):
            label = self.action_trie.get_leaf_label(node)
            if label:
                reachable.append(label)

        children = self.action_trie.children.get(node, {})
        for child_node in children.values():
            reachable.extend(self._get_reachable_actions(child_node))

        return reachable

    # =========================================================================
    # ALLOWED TOKENS
    # =========================================================================

    def allowed_tokens(self, state: Any) -> Set[int]:
        """Return allowed token IDs from state."""
        base, _ = self._unwrap_ws(state)
        phase = base[0]
        ws = self._get_ws(state)

        if phase == 'START':
            return set(self.LBRACE_START_ALTS) | set(self.LBRACE_QUOTE_START_ALTS)

        if phase == 'BLOCK_BOUNDARY':
            _, is_first = base
            if is_first:
                return set(self.WS_TOKENS) | set(self.RBRACE_ALTS)
            else:
                return set(self.COMMA_ALTS) | set(self.RBRACE_ALTS)

        if phase == 'EXPECT_ACTION':
            _, node = base
            # Quote-pending state: only allow opening quote
            if node == -1:
                return set(self.QUOTE_ALTS)

            allowed = set()

            # Only allow tokens that lead to viable actions
            for tok in self.action_trie.next_tokens(node):
                child_node = self.action_trie.next_node(node, tok)
                if child_node is not None:
                    reachable = self._get_reachable_actions(child_node)
                    if any(a in self.viable_actions for a in reachable):
                        allowed.add(tok)

            # At leaf node, allow quote tokens only if action is viable
            if self.action_trie.is_leaf_node(node):
                action = self.action_trie.get_leaf_label(node)
                if action in self.viable_actions:
                    allowed |= set(self.QUOTE_ALTS) | set(self.QUOTE_COLON_ALTS)

            return allowed

        if phase == 'KEY_QUOTE_CLOSE':
            return set(self.QUOTE_ALTS) | set(self.QUOTE_COLON_ALTS) | ws

        if phase == 'EXPECT_COLON':
            return set(self.COLON_ALTS) | ws

        if phase == 'ARRAY_OPEN':
            return set(self.LBRACK_ALTS) | set(self.LBRACK_QUOTE_ALTS) | ws

        if phase == 'EMPTY_ARRAY_NEED_WS':
            return set(self.WS_TOKENS) | set(self.RBRACK_LEADING_WS_ALTS)

        if phase == 'ARRAY_BODY':
            _, action, count = base
            arity = self.action_arity.get(action, 1)
            if count >= arity or arity == 0:
                return set(self.RBRACK_ALTS) | ws
            return set(self.QUOTE_ALTS) | ws

        if phase == 'VALUE_QUOTE_OPEN_NAME':
            return set(self.QUOTE_ALTS) | ws

        if phase == 'EXPECT_NAME':
            _, action, count, node = base
            allowed = set()

            viable = self._viable_name_nodes(action, count)
            children = self.name_trie.children.get(node, {})

            if viable is not None:
                if len(viable) == 0:
                    return set()
                allowed |= {tok for tok, child in children.items() if child in viable}
            else:
                allowed |= set(self.name_trie.next_tokens(node))

            if self.name_trie.is_leaf_node(node):
                name = self.name_trie.get_leaf_label(node)
                if name is not None:
                    allowed_set = self.allowed_objects_per_role.get((action, count))
                    if allowed_set is None or name in allowed_set:
                        allowed |= set(self.QUOTE_ALTS) | set(self.QUOTE_COMMA_ALTS)

            return allowed

        if phase == 'VALUE_QUOTE_CLOSE_NAME':
            return set(self.QUOTE_ALTS) | set(self.QUOTE_COMMA_ALTS) | ws

        if phase == 'NAME_ID_SEP':
            return set(self.COMMA_ALTS) | ws

        if phase == 'VALUE_QUOTE_OPEN_ID':
            return set(self.QUOTE_ALTS) | ws

        if phase == 'EXPECT_ID':
            _, name, action, count, node = base
            arity = self.action_arity.get(action, 1)
            trie = self.id_trie_by_name.get(name)
            if not trie:
                return set()
            allowed = set(trie.next_tokens(node))
            if trie.is_leaf_node(node):
                allowed |= set(self.QUOTE_ALTS)
                count2 = count + 1
                if count2 < arity:
                    allowed |= set(self.QUOTE_COMMA_ALTS)
                else:
                    allowed |= set(self.QUOTE_RBRACK_ALTS)
            return allowed

        if phase == 'VALUE_QUOTE_CLOSE_ID':
            return (set(self.QUOTE_ALTS) | set(self.QUOTE_RBRACK_ALTS) | ws)

        if phase == 'PAIRS_TAIL':
            _, action, count = base
            arity = self.action_arity.get(action, 1)
            if arity == 0 or count >= arity:
                return set(self.RBRACK_ALTS) | ws
            return set(self.COMMA_ALTS) | ws

        if phase == 'END':
            return set(self.eos_token_ids) | ws

        return set()

    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================

    def next_state(self, state: Any, token: int) -> Optional[Any]:
        """Compute next state."""
        base, ws_run = self._unwrap_ws(state)
        phase = base[0]

        if token in self.WS_TOKENS and phase in self.WS_PHASES:
            nxt = ws_run + 1
            if nxt > self.max_ws_tokens:
                return None
            return ('_WS', base, nxt)

        if phase == 'START':
            if token in self.LBRACE_QUOTE_START_ALTS:
                return ('EXPECT_ACTION', 0)
            if token in self.LBRACE_START_ALTS:
                return ('BLOCK_BOUNDARY', True)
            return None

        if phase == 'BLOCK_BOUNDARY':
            _, is_first = base
            if is_first:
                if token in self.WS_TOKENS:
                    return ('EXPECT_ACTION', -1)
            else:
                if token in self.COMMA_ALTS:
                    return ('EXPECT_ACTION', -1)
            if token in self.RBRACE_ALTS:
                return ('END',)
            return None

        if phase == 'EXPECT_ACTION':
            _, node = base
            # Quote-pending state: consume opening quote
            if node == -1:
                if token in self.QUOTE_ALTS:
                    return ('EXPECT_ACTION', 0)
                return None

            nxt = self.action_trie.next_node(node, token)
            if nxt is not None:
                return ('EXPECT_ACTION', nxt)

            if self.action_trie.is_leaf_node(node):
                action = self.action_trie.get_leaf_label(node)
                if token in self.QUOTE_COLON_ALTS:
                    return ('ARRAY_OPEN', action)
                if token in self.QUOTE_ALTS:
                    return ('KEY_QUOTE_CLOSE', action)
            return None

        if phase == 'KEY_QUOTE_CLOSE':
            _, action = base
            if token in self.QUOTE_COLON_ALTS:
                return ('ARRAY_OPEN', action)
            if token in self.QUOTE_ALTS:
                return ('EXPECT_COLON', action)
            return None

        if phase == 'EXPECT_COLON':
            _, action = base
            if token in self.COLON_ALTS:
                return ('ARRAY_OPEN', action)
            return None

        if phase == 'ARRAY_OPEN':
            _, action = base
            arity = self.action_arity.get(action, 1)
            if token in self.LBRACK_ALTS:
                if arity == 0:
                    return ('EMPTY_ARRAY_NEED_WS', action)
                return ('ARRAY_BODY', action, 0)
            if token in self.LBRACK_QUOTE_ALTS:
                if arity == 0:
                    return None
                return ('EXPECT_NAME', action, 0, 0)
            return None

        if phase == 'EMPTY_ARRAY_NEED_WS':
            _, action = base
            if token in self.WS_TOKENS:
                return ('PAIRS_TAIL', action, 0)
            if token in self.RBRACK_LEADING_WS_ALTS:
                return ('BLOCK_BOUNDARY', False)
            return None

        if phase == 'ARRAY_BODY':
            _, action, count = base
            arity = self.action_arity.get(action, 1)
            if count >= arity or (count == 0 and arity == 0):
                if token in self.RBRACK_ALTS:
                    return ('BLOCK_BOUNDARY', False)
                if count >= arity:
                    return None
            if token in self.QUOTE_ALTS:
                return ('EXPECT_NAME', action, count, 0)
            return None

        if phase == 'VALUE_QUOTE_OPEN_NAME':
            _, action, count = base
            if token in self.QUOTE_ALTS:
                return ('EXPECT_NAME', action, count, 0)
            return None

        if phase == 'EXPECT_NAME':
            _, action, count, node = base
            nxt = self.name_trie.next_node(node, token)
            if nxt is not None:
                return ('EXPECT_NAME', action, count, nxt)

            if self.name_trie.is_leaf_node(node):
                name = self.name_trie.get_leaf_label(node)
                if name is None:
                    return None
                allowed_set = self.allowed_objects_per_role.get((action, count))
                if allowed_set is not None and name not in allowed_set:
                    return None

                if token in self.QUOTE_COMMA_ALTS:
                    return ('VALUE_QUOTE_OPEN_ID', name, action, count)
                if token in self.QUOTE_ALTS:
                    return ('NAME_ID_SEP', name, action, count)
            return None

        if phase == 'VALUE_QUOTE_CLOSE_NAME':
            _, name, action, count = base
            if token in self.QUOTE_COMMA_ALTS:
                return ('VALUE_QUOTE_OPEN_ID', name, action, count)
            if token in self.QUOTE_ALTS:
                return ('NAME_ID_SEP', name, action, count)
            return None

        if phase == 'NAME_ID_SEP':
            _, name, action, count = base
            if token in self.COMMA_ALTS:
                return ('VALUE_QUOTE_OPEN_ID', name, action, count)
            return None

        if phase == 'VALUE_QUOTE_OPEN_ID':
            _, name, action, count = base
            if token in self.QUOTE_ALTS:
                return ('EXPECT_ID', name, action, count, 0)
            return None

        if phase == 'EXPECT_ID':
            _, name, action, count, node = base
            arity = self.action_arity.get(action, 1)
            trie = self.id_trie_by_name.get(name)
            if trie is None:
                return None

            nxt = trie.next_node(node, token)
            if nxt is not None:
                return ('EXPECT_ID', name, action, count, nxt)

            if trie.is_leaf_node(node):
                count2 = count + 1
                if token in self.QUOTE_COMMA_ALTS:
                    if count2 >= arity:
                        return None
                    return ('VALUE_QUOTE_OPEN_NAME', action, count2)
                if token in self.QUOTE_RBRACK_ALTS:
                    if count2 < arity:
                        return None
                    return ('BLOCK_BOUNDARY', False)
                if token in self.QUOTE_ALTS:
                    return ('PAIRS_TAIL', action, count2)
            return None

        if phase == 'VALUE_QUOTE_CLOSE_ID':
            _, action, count = base
            arity = self.action_arity.get(action, 1)
            if token in self.QUOTE_RBRACK_ALTS:
                if count < arity:
                    return None
                return ('BLOCK_BOUNDARY', False)
            if token in self.QUOTE_ALTS:
                return ('PAIRS_TAIL', action, count)
            return None

        if phase == 'PAIRS_TAIL':
            _, action, count = base
            arity = self.action_arity.get(action, 1)
            if arity == 0 or count >= arity:
                if token in self.RBRACK_ALTS:
                    return ('BLOCK_BOUNDARY', False)
                return None
            if token in self.COMMA_ALTS:
                return ('VALUE_QUOTE_OPEN_NAME', action, count)
            return None

        if phase == 'END':
            if token in self.eos_token_ids:
                return ('END',)
            return None

        return None


# =============================================================================
# Factory Function
# =============================================================================

def build_vh_state_machine(
    tokenizer,
    prompt: str,
    eos_token_id: int,
    max_ws_tokens: int = 5,
) -> VHStateMachine:
    """
    Build VirtualHome state machine from prompt.

    This is the main entry point for creating a VH DFA.
    It parses only objects from the prompt and uses hardcoded actions.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Full VH prompt text
        eos_token_id: EOS token ID
        max_ws_tokens: Maximum consecutive whitespace tokens

    Returns:
        VHStateMachine instance
    """
    from .prompt_parser import parse_vh_objects

    objects, properties = parse_vh_objects(prompt)
    return VHStateMachine(
        tokenizer=tokenizer,
        objects=objects,
        properties=properties,
        eos_token_id=eos_token_id,
        max_ws_tokens=max_ws_tokens,
    )

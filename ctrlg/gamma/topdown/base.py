"""Abstract base class for top-down defined state machines."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np


class CanonicalStateMachine(ABC):
    """
    Abstract base for top-down defined state machines.

    Subclasses define canonical states from the grammar structure.
    `build_graph` compiles from `enumerate_states()`; `build_graph_bfs`
    instead discovers the reachable states from the initial state.
    """

    @abstractmethod
    def enumerate_states(self) -> List[Any]:
        """
        Return ALL canonical states (finite set).

        States should be hashable (tuples, dataclasses with frozen=True, etc.)
        """

    @abstractmethod
    def initial_state(self) -> Any:
        """Return initial canonical state."""

    @abstractmethod
    def is_accept(self, state: Any) -> bool:
        """Check if state is accepting."""

    @abstractmethod
    def allowed_tokens(self, state: Any) -> Set[int]:
        """Return set of allowed token IDs from state."""

    @abstractmethod
    def next_state(self, state: Any, token: int) -> Optional[Any]:
        """
        Return next state after consuming token, or None if invalid.

        This should return a canonical state (one of the enumerated states).
        """

    def build_graph(self, vocab_size: int) -> Dict[str, Any]:
        """
        Build compiled graph directly from state enumeration.

        Returns:
            Dict with:
            - edges: list of (src_id, dst_id, transition_bitset)
            - initial_state: int
            - accept_states: set of int
            - state_map: dict state -> id
            - compile_stats: dict of counts and the build method
        """
        states = self.enumerate_states()
        state_to_id = {s: i for i, s in enumerate(states)}

        # Group edges by (src, dst) to build transition arrays
        edge_tokens: Dict[tuple, Set[int]] = defaultdict(set)

        for state in states:
            src_id = state_to_id[state]
            allowed = self.allowed_tokens(state)

            for tok in allowed:
                next_s = self.next_state(state, tok)
                if next_s is not None and next_s in state_to_id:
                    dst_id = state_to_id[next_s]
                    edge_tokens[(src_id, dst_id)].add(tok)

        # Build edge list with transition arrays
        edges = []
        for (src_id, dst_id), tokens in edge_tokens.items():
            transition = np.zeros(vocab_size, dtype=bool)
            for tok in tokens:
                if 0 <= tok < vocab_size:
                    transition[tok] = True
            edges.append((src_id, dst_id, transition))

        # Identify accept states
        accept_ids = {state_to_id[s] for s in states if self.is_accept(s)}

        # Validation: check for reachable states with 0 outgoing edges
        # (excluding accept states which legitimately have EOS-only transitions)
        src_ids_with_edges = set()
        for (src_id, dst_id) in edge_tokens.keys():
            src_ids_with_edges.add(src_id)

        # Also check for transitions that lead to non-enumerated states (dropped silently)
        dropped_transitions = 0
        for state in states:
            allowed = self.allowed_tokens(state)
            for tok in allowed:
                next_s = self.next_state(state, tok)
                if next_s is not None and next_s not in state_to_id:
                    dropped_transitions += 1
                    if dropped_transitions <= 5:
                        print(f"[WARN] Dropped transition: {state} --tok={tok}--> {next_s} (not enumerated)")

        if dropped_transitions > 0:
            print(f"[WARN] Total dropped transitions (dest not enumerated): {dropped_transitions}")

        dead_end_states = []
        for state in states:
            sid = state_to_id[state]
            if sid not in src_ids_with_edges and sid not in accept_ids:
                dead_end_states.append(state)

        if dead_end_states:
            print(f"[WARN] {len(dead_end_states)} non-accept states with 0 outgoing edges (dead-end):")
            for s in dead_end_states[:10]:
                print(f"  {s}")

        return {
            'edges': edges,
            'initial_state': state_to_id[self.initial_state()],
            'accept_states': accept_ids,
            'state_map': state_to_id,
            'compile_stats': {
                'canonical_state_count': len(states),
                'edges': len(edges),
                'dead_end_states': len(dead_end_states),
                'dropped_transitions': dropped_transitions,
                'method': 'topdown',
            },
        }

    def build_graph_bfs(self, vocab_size: int) -> Dict[str, Any]:
        """
        Build compiled graph via BFS from initial state.

        Unlike build_graph() which relies on enumerate_states() being complete,
        this method discovers ALL reachable states by BFS traversal of
        allowed_tokens() and next_state(). Guarantees no dropped transitions.

        Returns same format as build_graph().
        """
        from collections import deque

        initial = self.initial_state()
        states = [initial]
        state_to_id = {initial: 0}
        edge_tokens: Dict[tuple, Set[int]] = defaultdict(set)

        queue = deque([0])

        while queue:
            src_idx = queue.popleft()
            src_state = states[src_idx]
            allowed = self.allowed_tokens(src_state)

            for tok in allowed:
                next_s = self.next_state(src_state, tok)
                if next_s is None:
                    continue

                # Discover new state
                if next_s not in state_to_id:
                    dst_idx = len(states)
                    states.append(next_s)
                    state_to_id[next_s] = dst_idx
                    queue.append(dst_idx)
                else:
                    dst_idx = state_to_id[next_s]

                edge_tokens[(src_idx, dst_idx)].add(tok)

        # Build edge list with transition arrays
        edges = []
        for (src_id, dst_id), tokens in edge_tokens.items():
            transition = np.zeros(vocab_size, dtype=bool)
            for tok in tokens:
                if 0 <= tok < vocab_size:
                    transition[tok] = True
            edges.append((src_id, dst_id, transition))

        accept_ids = {state_to_id[s] for s in states if self.is_accept(s)}

        return {
            'edges': edges,
            'initial_state': 0,
            'accept_states': accept_ids,
            'state_map': state_to_id,
            'compile_stats': {
                'canonical_state_count': len(states),
                'edges': len(edges),
                'dead_end_states': 0,
                'dropped_transitions': 0,
                'method': 'bfs',
            },
        }

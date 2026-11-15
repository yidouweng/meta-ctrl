"""Top-down DFA compilation entry point."""

from typing import Any, Dict

from .base import CanonicalStateMachine


def compile_topdown(
    state_machine: Any,
    vocab_size: int,
    *,
    debug: bool = False,
    use_bfs: bool = False,
) -> Dict[str, Any]:
    """
    Compile a state machine to a DFA model.

    Args:
        state_machine: A CanonicalStateMachine instance.
        vocab_size: Vocabulary size for transition arrays.
        debug: Print debug info.
        use_bfs: If True, discover the reachable states by BFS from the initial
                 state. If False (default), compile from enumerate_states(),
                 which must list every reachable state (it may list more).

    Returns:
        Dict with edges, initial_state, accept_states, state_map, compile_stats.
    """
    import time as _time
    t_start = _time.perf_counter()

    if isinstance(state_machine, CanonicalStateMachine):
        sm = state_machine
    else:
        cls_name = type(state_machine).__name__
        raise ValueError(
            f"compile_topdown requires a CanonicalStateMachine, got {cls_name}."
        )

    cls_name = type(sm).__name__

    t_build_start = _time.perf_counter()
    if use_bfs:
        result = sm.build_graph_bfs(vocab_size)
    else:
        sm.enumerate_states()
        result = sm.build_graph(vocab_size)
    t_build = _time.perf_counter() - t_build_start

    t_total = _time.perf_counter() - t_start

    result['compile_stats'].update({
        't_total_ms': t_total * 1000,
        't_build_ms': t_build * 1000,
        'method': 'bfs' if use_bfs else 'topdown',
        'dfa_type': cls_name,
    })

    if debug:
        print(f"[compile_topdown] {result['compile_stats']['canonical_state_count']} states, "
              f"{result['compile_stats']['edges']} edges ({result['compile_stats']['method']})")

    return result

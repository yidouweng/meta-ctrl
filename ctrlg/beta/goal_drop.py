"""Shared goal-dropping infrastructure for β state machines.

When a β-SM has too many goals (state-space blow-up) or no reachable
accept state, the relaxation is to drop a goal and retry.
This module provides the build-check-drop loop; each β-SM supplies the
build, accept-check, and drop-priority callbacks.

Used by `sd_state_machine` and `beh_sd_state_machine`. `vh_state_machine`
uses an estimate-then-drop paradigm and does not call this helper.
"""
from typing import Callable, List, Tuple, TypeVar

T = TypeVar('T')  # build result type
G = TypeVar('G')  # goal type


def build_with_dropping(
    goals: List[G],
    *,
    build_fn: Callable[[List[G]], Tuple[T, bool]],
    has_accept_fn: Callable[[T, List[G]], bool],
    pick_drop_fn: Callable[[List[G]], int],
    min_active: int = 0,
    debug: bool = False,
    label: str = 'BetaSM',
) -> Tuple[T, List[G], List[G]]:
    """Build, check accept, drop one goal at a time until feasible.

    Args:
        goals: initial active goals (copied; caller's list is not mutated).
        build_fn: takes active goals, returns (build_result, ok). `ok=False`
            means construction hit a resource cap (e.g. max_states).
        has_accept_fn: takes (build_result, active goals), returns True iff
            the result has ≥1 reachable accept state.
        pick_drop_fn: takes active goals, returns the index to drop. The
            caller's priority logic (typically: drop the goal β-steering
            helps least first).
        min_active: minimum active goal count. Loop exits once we'd drop
            below this even without accept — caller is responsible for
            deciding what to do (e.g. γ-only fallback).
        debug: print a one-line trace per drop.
        label: prefix for debug messages.

    Returns:
        (final_result, kept_goals, dropped_goals)
    """
    active = list(goals)
    dropped: List[G] = []
    while True:
        result, ok = build_fn(active)
        if ok and has_accept_fn(result, active):
            return result, active, dropped
        if len(active) <= min_active:
            return result, active, dropped
        drop_idx = pick_drop_fn(active)
        d = active.pop(drop_idx)
        dropped.append(d)
        if debug:
            reason = "over cap" if not ok else "no accept"
            print(f"[{label}] {reason} → dropping {d} (active: {len(active)})")

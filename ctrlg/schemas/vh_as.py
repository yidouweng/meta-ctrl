"""Canonical schema for VirtualHome Action Sequencing (VH AS).

This is the single source of truth for the VH AS action set, arity,
affordances, and goal-state→action mapping. The values here must be a
subset of (or equal to) the official evaluator's `valid_actions` in:

    embodied-agent-interface/src/virtualhome_eval/simulation/evolving_graph/eval_utils.py

Actions left out:
  - WAKEUP, PUTOBJBACK: not needed by any goal.
  - LOOKAT_SHORT/MEDIUM/LONG: not implemented by the VirtualHome executor.
  - WATCH: rejected by the simulator on most objects; goals that mention WATCH
    also accept LOOKAT, which is kept.
  - GREET: valid only for person objects.

Pure data; `ctrlg/gamma/topdown/action_defs.py` re-exports it.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple


# -----------------------------------------------------------------------------
# Action universe (must be ⊆ official `valid_actions`)
# -----------------------------------------------------------------------------

ACTIONS: FrozenSet[str] = frozenset({
    # Arity 0
    "STANDUP", "SLEEP",
    # Arity 1
    "DRINK", "EAT", "CUT", "TOUCH", "LOOKAT", "READ", "TYPE",
    "PUSH", "PULL", "MOVE", "SQUEEZE", "RINSE", "SCRUB", "WASH",
    "GRAB", "SWITCHOFF", "SWITCHON", "CLOSE", "FIND", "TURNTO",
    "WALK", "OPEN", "POINTAT", "RUN", "SIT", "WIPE", "DROP", "LIE",
    "PLUGIN", "PLUGOUT",
    # Arity 2
    "PUTBACK", "PUTIN", "PUTON", "PUTOFF", "POUR",
})


# -----------------------------------------------------------------------------
# Arity (number of object arguments after the action name)
# -----------------------------------------------------------------------------

ACTION_ARITY: Dict[str, int] = {
    "STANDUP": 0, "SLEEP": 0,
    "POUR": 2, "PUTBACK": 2, "PUTIN": 2,
    **{a: 1 for a in ACTIONS
       if a not in ("STANDUP", "SLEEP", "POUR", "PUTBACK", "PUTIN")},
}


# -----------------------------------------------------------------------------
# γ affordance rules — action → [(role_idx, [required_properties])]
#
# Object must have ALL listed properties for that role to be eligible.
# Mirrors the simulator's `check_*` precondition logic, distilled to the
# property-bit form γ can filter on from the prompt's object listings.
# -----------------------------------------------------------------------------

AFFORDANCES: Dict[str, List[Tuple[int, List[str]]]] = {
    "GRAB":      [(0, ["GRABBABLE"])],
    "PUTIN":     [(0, ["GRABBABLE"]), (1, ["CONTAINERS"])],
    "PUTBACK":   [(0, ["GRABBABLE"]), (1, ["SURFACES"])],
    "POUR":      [(0, ["GRABBABLE", "POURABLE"]), (1, ["RECIPIENT"])],
    "DROP":      [(0, ["GRABBABLE"])],
    "OPEN":      [(0, ["CAN_OPEN"])],
    "CLOSE":     [(0, ["CAN_OPEN"])],
    "SWITCHON":  [(0, ["HAS_SWITCH"])],
    "SWITCHOFF": [(0, ["HAS_SWITCH"])],
    "PLUGIN":    [(0, ["HAS_PLUG"])],
    "PLUGOUT":   [(0, ["HAS_PLUG"])],
    "SIT":       [(0, ["SITTABLE"])],
    "LIE":       [(0, ["LIEABLE"])],
    "EAT":       [(0, ["EATABLE"])],
    "DRINK":     [(0, ["DRINKABLE"])],
    "CUT":       [(0, ["CUTTABLE"])],
    "READ":      [(0, ["READABLE"])],
    "PUTON":     [(0, ["CLOTHES"])],
    "PUTOFF":    [(0, ["CLOTHES"])],
    # TYPE: VH's `check_typeable` allows the action iff the object is a
    # keyboard OR has HAS_SWITCH; without this rule γ emitted TYPE on
    # food/toilet paper/toothbrush.
    "TYPE":      [(0, ["HAS_SWITCH"])],
    # PUSH/PULL/MOVE: the prompt's action list requires MOVABLE; the simulator's
    # MoveExecutor.check_movable rejects MOVE/PULL on other objects (affordance error).
    "PUSH":      [(0, ["MOVABLE"])],
    "PULL":      [(0, ["MOVABLE"])],
    "MOVE":      [(0, ["MOVABLE"])],
}

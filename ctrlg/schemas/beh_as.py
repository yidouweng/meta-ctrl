"""Canonical schema for BEHAVIOR Action Sequencing (BEH AS).

Single source of truth for BEH AS action set, arity, and affordance
predicates. Mirrors the official evaluator's `EvalActions` enum in:

    embodied-agent-interface/src/behavior_eval/evolving_graph/eval_evolving_graph_env.py

Excluded vs the older AS-union vocab (and why):
  - BURN: in some EvalActions variants but not executable in the AS pipeline.
  - VACUUM: not in EvalActions; LLM hallucinated it on vacuuming prompts.
  - WIPE: BEH's `CLEAN` covers it; WIPE is VH-only.

Affordances are predicate-based (not property-based as in VH): an action
applies only to objects mentioned in init/target state blocks with the
listed state-keys (`open`, `toggled_on`, `dusty`, etc.).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Set


# -----------------------------------------------------------------------------
# Action universe
# -----------------------------------------------------------------------------

ACTIONS: FrozenSet[str] = frozenset({
    # Navigation (in EvalActions; not in prompt template)
    "NAVIGATE_TO",
    # Grasp / release
    "LEFT_GRASP", "RIGHT_GRASP",
    "LEFT_RELEASE", "RIGHT_RELEASE",
    # Placement (1-object: target/container)
    "LEFT_PLACE_ONTOP", "RIGHT_PLACE_ONTOP",
    "LEFT_PLACE_INSIDE", "RIGHT_PLACE_INSIDE",
    "LEFT_PLACE_NEXTTO", "RIGHT_PLACE_NEXTTO",
    "LEFT_PLACE_UNDER", "RIGHT_PLACE_UNDER",
    # Placement (2-object: nextto+ontop)
    "LEFT_PLACE_NEXTTO_ONTOP", "RIGHT_PLACE_NEXTTO_ONTOP",
    # Transfer contents
    "LEFT_TRANSFER_CONTENTS_INSIDE", "RIGHT_TRANSFER_CONTENTS_INSIDE",
    "LEFT_TRANSFER_CONTENTS_ONTOP", "RIGHT_TRANSFER_CONTENTS_ONTOP",
    # State actions
    "OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF",
    # Attribute actions
    "CLEAN", "SLICE", "SOAK", "DRY", "COOK", "FREEZE", "UNFREEZE",
})


# -----------------------------------------------------------------------------
# Arity
# -----------------------------------------------------------------------------

ACTION_ARITY: Dict[str, int] = {
    "LEFT_PLACE_NEXTTO_ONTOP": 2,
    "RIGHT_PLACE_NEXTTO_ONTOP": 2,
    **{a: 1 for a in ACTIONS if a not in (
        "LEFT_PLACE_NEXTTO_ONTOP", "RIGHT_PLACE_NEXTTO_ONTOP",
    )},
}


# -----------------------------------------------------------------------------
# Affordance predicates (BEHAVIOR-style: from init/target state blocks)
#
# An action requires the target object to be mentioned in init/target
# state with at least one of the listed state-key predicates. Used by
# `compute_behavior_affordances` to filter γ's per-(action, role) object
# eligibility.
# -----------------------------------------------------------------------------

AFFORDANCE_PREDICATES: Dict[str, Set[str]] = {
    "OPEN": {"open"},
    "CLOSE": {"open"},
    "TOGGLE_ON": {"toggled_on"},
    "TOGGLE_OFF": {"toggled_on"},
    "FREEZE": {"frozen"},
    "UNFREEZE": {"frozen"},
    "SLICE": {"sliced"},
    "SOAK": {"soaked"},
    "DRY": {"soaked"},
    "COOK": {"cooked"},
    "CLEAN": {"dusty", "stained"},  # either predicate suffices
}


# -----------------------------------------------------------------------------
# Patterns the simulator rejects (helps the validator surface bad LLM output)
# -----------------------------------------------------------------------------


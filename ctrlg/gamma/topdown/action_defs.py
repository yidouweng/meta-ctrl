"""γ-side action definitions for VirtualHome and BEHAVIOR — thin re-export.

The canonical data (action set, arity, affordances) lives in
`ctrlg.schemas.{vh_as,beh_as}`. This module re-exports those names under
the names the γ automata use and provides the per-prompt affordance
expansion functions (`compute_vh_affordances`, `compute_behavior_affordances`).
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ctrlg.schemas import vh_as as _vh_as
from ctrlg.schemas import beh_as as _beh_as


# =============================================================================
# BEHAVIOR Action Definitions
# =============================================================================

# Canonical sources — see ctrlg/schemas/beh_as.py
BEHAVIOR_ACTIONS: FrozenSet[str] = _beh_as.ACTIONS
BEHAVIOR_ACTION_ARITY: Dict[str, int] = _beh_as.ACTION_ARITY
BEHAVIOR_AFFORDANCE_PREDICATES: Dict[str, Set[str]] = _beh_as.AFFORDANCE_PREDICATES


def compute_behavior_affordances(
    predicates_by_object: Dict[str, Set[str]],
    all_objects: "Optional[Dict[str, list]]" = None,
    membership=None,
    container_role_bases: "Optional[Set[str]]" = None,
) -> Dict[Tuple[str, int], Set[str]]:
    """
    Compute allowed_objects_per_role for BEHAVIOR.

    Args:
        predicates_by_object: {base_name: {predicate1, predicate2, ...}}
            Parsed from init/target state blocks in prompt.
        all_objects: {base_name: [inst_names...]}. If provided, static
            capabilities (openable/toggleable/soakable-cleaner) from the
            shared taxonomy layer are ORed in for OPEN/CLOSE, TOGGLE_ON/OFF
            and SOAK/DRY, even when the prompt didn't mention the state —
            e.g. TOGGLE_ON(stove) is valid without `toggled_on(stove)`.
        membership: `ctrlg.beta.beh_affordances.BehAffordances` built from
            the prompt's {'name','category'} list; the same layer the β SM
            gates on, so γ and β share one membership source. If None while
            all_objects is given, a category-less layer is built from the
            instance names (name-derived synset fallback).
        container_role_bases: base names that appear as the container of an
            `inside` fact in the init/target state. Restricts the OPEN/CLOSE
            static fallback to these objects.

    Returns:
        {(action, role_idx): {allowed_base_names}}
    """
    if membership is None and all_objects is not None:
        from ctrlg.beta.beh_affordances import BehAffordances
        membership = BehAffordances(objects=None)

    # Taxonomy-backed static-capability fallbacks (object type, not state),
    # added on top of the prompt-mention rule below:
    #   OPEN/CLOSE     — open-gated containers that appear in a container
    #                    role (the set β's open-before-inside gating uses).
    #   TOGGLE_ON/OFF  — toggleable objects, including tap fixtures.
    #   SOAK/DRY       — cleaning implements only; other soak targets must
    #                    have a soaked state in the init/target block.
    FALLBACK_FN = {}
    if membership is not None:
        _role_set = set(container_role_bases or ())
        _open_fb = lambda b: membership.open_gated(b) and b in _role_set
        FALLBACK_FN = {
            'OPEN': _open_fb,
            'CLOSE': _open_fb,
            'TOGGLE_ON': membership.toggleable,
            'TOGGLE_OFF': membership.toggleable,
            'SOAK': membership.cleaner,
            'DRY': membership.cleaner,
        }

    allowed: Dict[Tuple[str, int], Set[str]] = {}

    for action, required_preds in BEHAVIOR_AFFORDANCE_PREDICATES.items():
        # Find objects that have ANY of the required predicates
        matching = set()
        for obj, preds in predicates_by_object.items():
            if preds & required_preds:  # Intersection
                matching.add(obj)

        # Taxonomy fallback for actions whose targets have an implicit
        # static capability (type, not state).
        fn = FALLBACK_FN.get(action)
        if fn is not None and all_objects is not None:
            for base in all_objects:
                if fn(base):
                    matching.add(base)

        # ALWAYS record the allowed set — including empty. Empty means
        # "no object in the scene has this property", so the action
        # should be blocked entirely.
        arity = BEHAVIOR_ACTION_ARITY.get(action, 1)
        for role in range(arity):
            allowed[(action, role)] = matching

    return allowed


# =============================================================================
# VirtualHome Action Definitions
# =============================================================================

# Canonical sources — see ctrlg/schemas/vh_as.py
VH_ACTIONS: FrozenSet[str] = _vh_as.ACTIONS
VH_ACTION_ARITY: Dict[str, int] = _vh_as.ACTION_ARITY
VH_AFFORDANCE_RULES: Dict[str, List[Tuple[int, List[str]]]] = _vh_as.AFFORDANCES


def compute_vh_affordances(
    properties_by_name: Dict[str, Set[str]],
) -> Dict[Tuple[str, int], Set[str]]:
    """
    Compute allowed_objects_per_role for VirtualHome.

    Args:
        properties_by_name: {object_name: {PROP1, PROP2, ...}}
            Parsed from "Objects in the scene" section in prompt.

    Returns:
        {(action, role_idx): {allowed_object_names}}
        Missing key means no restriction (all objects allowed).
    """
    allowed: Dict[Tuple[str, int], Set[str]] = {}

    for action, role_rules in VH_AFFORDANCE_RULES.items():
        for role_idx, required_props in role_rules:
            # Find objects that have ALL required properties
            matching = set()
            required_set = {p.upper() for p in required_props}

            for name, props in properties_by_name.items():
                props_upper = {p.upper() for p in props}
                if required_set <= props_upper:  # Subset check
                    matching.add(name)

            if matching:
                allowed[(action, role_idx)] = matching
            else:
                # No objects match - this action/role is blocked
                allowed[(action, role_idx)] = set()

    # DRINK special case: also allow RECIPIENT objects (cups, glasses)
    # VH semantics: DRINK means "drink from container you're holding"
    # Both DRINKABLE substances (water) and RECIPIENT containers (cup) are valid targets
    if ("DRINK", 0) in allowed:
        for name, props in properties_by_name.items():
            props_upper = {p.upper() for p in props}
            if "RECIPIENT" in props_upper and "GRABBABLE" in props_upper:
                allowed[("DRINK", 0)].add(name)

    return allowed

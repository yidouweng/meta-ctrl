"""
Predicate definitions for VirtualHome Subgoal Decomposition.

Single source of truth for:
- Predicate names (state + action predicates)
- Predicate arities
- Affordance rules (which object properties are required per argument)
"""

from typing import Dict, FrozenSet, Optional, Set, Tuple

# =============================================================================
# State Predicates
# =============================================================================

# Unary state predicates: predicate(obj.id)
SD_UNARY_STATE_PREDICATES: FrozenSet[str] = frozenset({
    "CLOSED", "OPEN",               # CAN_OPEN objects
    "ON", "OFF",                     # HAS_SWITCH objects
    "PLUGGED_IN", "PLUGGED_OUT",     # HAS_PLUG objects
    "SITTING", "LYING",              # character only
    "CLEAN", "DIRTY",               # any object
})

# Binary state predicates: predicate(obj1.id, obj2.id)
SD_BINARY_STATE_PREDICATES: FrozenSet[str] = frozenset({
    "ONTOP",       # (obj, surface)
    "INSIDE",      # (obj/character, room/container)
    "NEXT_TO",     # (character, object)
    "FACING",      # (character, object)
    "HOLDS_RH",    # (character, grabbable_obj)
    "HOLDS_LH",    # (character, grabbable_obj)
})

# =============================================================================
# Action Predicates (from subgoal_actions vocabulary)
# =============================================================================

# Unary action predicates: predicate(obj.id)
SD_UNARY_ACTION_PREDICATES: FrozenSet[str] = frozenset({
    "GRAB",         # GRABBABLE objects
    "DRINK",        # DRINKABLE or RECIPIENT
    "EAT",          # EATABLE
    "CUT",          # EATABLE or CUTTABLE
    "TOUCH",        # any object
    "LOOKAT",       # any object
    "WATCH",        # any object
    "READ",         # READABLE
    "TYPE",         # HAS_SWITCH
    "PUSH",         # MOVABLE
    "PULL",         # MOVABLE
    "MOVE",         # MOVABLE
    "SQUEEZE",      # CLOTHES
    "RINSE",        # any non-room object
    "SCRUB",        # any non-room object
    "WASH",         # any non-room object
    "WIPE",         # any non-room object
    "SWITCHOFF",    # HAS_SWITCH
})

# Nullary action predicates: predicate()
SD_NULLARY_ACTION_PREDICATES: FrozenSet[str] = frozenset({
    "SLEEP",
    "WAKEUP",
    # STANDUP is excluded: the simulator supports it, but it is not in the
    # evaluator's SD subgoal vocabulary, which scores it as an unknown
    # primitive (evaluator precondition).
})

# =============================================================================
# Combined Sets
# =============================================================================

SD_ALL_PREDICATES: FrozenSet[str] = (
    SD_UNARY_STATE_PREDICATES
    | SD_BINARY_STATE_PREDICATES
    | SD_UNARY_ACTION_PREDICATES
    | SD_NULLARY_ACTION_PREDICATES
)

SD_PREDICATE_ARITY: Dict[str, int] = {}
for p in SD_UNARY_STATE_PREDICATES:
    SD_PREDICATE_ARITY[p] = 1
for p in SD_BINARY_STATE_PREDICATES:
    SD_PREDICATE_ARITY[p] = 2
for p in SD_UNARY_ACTION_PREDICATES:
    SD_PREDICATE_ARITY[p] = 1
for p in SD_NULLARY_ACTION_PREDICATES:
    SD_PREDICATE_ARITY[p] = 0

# =============================================================================
# Affordance Rules
# =============================================================================

# For each (predicate, arg_index), which object properties are REQUIRED.
# If a predicate is not listed, any scene object is allowed for that argument.
# "character_only" is a special value meaning only the character object is valid.
#
# Format: {(predicate, arg_index): requirement}
# requirement is one of:
#   - set of property strings: object must have AT LEAST ONE of these
#   - "character_only": only the character object (category "Characters")
#   - None: no restriction (any scene object)

# =============================================================================
# Cross-argument predicates
# =============================================================================

# Predicates whose arg1 restrictions depend on the identity of arg0.
# For these predicates the state machine carries arg0_name when transitioning
# into the arg1 name trie.
CROSS_ARG_PREDICATES: FrozenSet[str] = frozenset({"ONTOP", "INSIDE"})

# Simple affordance rules (predicates where args are independent).
# ONTOP and INSIDE are handled separately in compute_sd_allowed_objects /
# compute_sd_arg1_allowed_objects because their arg1 requirements depend on
# what arg0 was chosen.
SD_AFFORDANCE_RULES: Dict[Tuple[str, int], object] = {
    # -- Unary state predicates --
    # Simulator constraints (property checked by each executor):
    ("CLOSED", 0): {"CAN_OPEN"},      # OpenExecutor checks CAN_OPEN
    ("OPEN", 0): {"CAN_OPEN"},        # OpenExecutor checks CAN_OPEN
    ("ON", 0): {"HAS_SWITCH"},        # SwitchExecutor checks HAS_SWITCH
    ("OFF", 0): {"HAS_SWITCH"},       # SwitchExecutor checks HAS_SWITCH
    ("PLUGGED_IN", 0): {"HAS_PLUG"},  # PlugExecutor checks HAS_PLUG
    ("PLUGGED_OUT", 0): {"HAS_PLUG"}, # PlugExecutor checks HAS_PLUG
    ("SITTING", 0): "character_only", # Evaluator: no-op (empty candidates)
    ("LYING", 0): "character_only",   # Evaluator: no-op (empty candidates)
    # CLEAN: evaluator generates WIPE or WASH (no property check in sim)
    # DIRTY: evaluator: no-op (empty candidates)

    # -- Binary state predicates --
    # ONTOP: arg0 rules (arg1 handled by compute_sd_arg1_allowed_objects)
    #   ONTOP(char, surface) → SIT/LIE
    #   ONTOP(obj_clothes, char) → PUTON
    #   ONTOP(obj_grabbable, any_non_room) → GRAB+PUTBACK
    # arg0: character, CLOTHES or GRABBABLE object
    # INSIDE: arg0 rules (arg1 handled by compute_sd_arg1_allowed_objects)
    #   INSIDE(char, room) → WALK room
    #   INSIDE(obj_grabbable, container) → GRAB+PUTIN
    # arg0: character or GRABBABLE object
    # NEXT_TO: evaluator only handles NEXT_TO(char, obj) via FIND
    ("NEXT_TO", 0): "character_only",
    # FACING: evaluator generates FIND+TURNTO (no property check)
    ("FACING", 0): "character_only",
    # HOLDS_RH/LH: evaluator generates FIND+GRAB (GRAB needs GRABBABLE)
    ("HOLDS_RH", 0): "character_only",
    ("HOLDS_RH", 1): {"GRABBABLE"},
    ("HOLDS_LH", 0): "character_only",
    ("HOLDS_LH", 1): {"GRABBABLE"},

    # -- Action predicates --
    # Simulator constraints (property checked by each executor):
    ("GRAB", 0): {"GRABBABLE"},             # GrabExecutor checks GRABBABLE
    ("DRINK", 0): {"DRINKABLE", "RECIPIENT"},  # DrinkExecutor checks
    ("EAT", 0): {"EATABLE"},                # EatExecutor checks EATABLE
    ("CUT", 0): {"EATABLE", "CUTTABLE"},    # CutExecutor
    ("READ", 0): {"READABLE"},              # ReadExecutor checks READABLE
    ("TYPE", 0): {"HAS_SWITCH"},            # TypeExecutor checks HAS_SWITCH
    ("PUSH", 0): {"MOVABLE"},               # PushExecutor checks MOVABLE
    ("PULL", 0): {"MOVABLE"},               # PullExecutor checks MOVABLE
    ("MOVE", 0): {"MOVABLE"},               # MoveExecutor checks MOVABLE
    ("SQUEEZE", 0): {"CLOTHES"},            # SqueezeExecutor checks CLOTHES
    ("SWITCHOFF", 0): {"HAS_SWITCH"},       # SwitchExecutor checks HAS_SWITCH
    # RINSE, SCRUB, WASH: sim has no property check, just needs CLOSE
    # WIPE: sim has no property check, just needs CLOSE + holding something
    # TOUCH, LOOKAT, WATCH: no property restriction in sim
}


def compute_sd_allowed_objects(
    objects: Dict[str, list],
    properties: Dict[str, Set[str]],
    categories: Dict[str, str],
) -> Dict[Tuple[str, int], Set[str]]:
    """
    Compute which object names are allowed for each (predicate, arg_index).

    Mirrors the property checks of the VirtualHome simulator executors and
    the evaluator's subgoal-to-action translation.

    Args:
        objects: {name: [id1, id2, ...]} — all scene objects
        properties: {name: {PROP1, PROP2, ...}} — object properties
        categories: {name: category_string} — "Rooms", "Characters", etc.

    Returns:
        {(predicate, arg_index): set of allowed object names}
    """
    all_names = set(objects.keys())
    room_names = {n for n, c in categories.items() if c == "Rooms"}
    char_names = {n for n, c in categories.items() if c == "Characters"}
    non_room_names = all_names - room_names

    def _has_any(name, required_props):
        return bool(required_props & properties.get(name, set()))

    result: Dict[Tuple[str, int], Set[str]] = {}

    for pred in SD_ALL_PREDICATES:
        arity = SD_PREDICATE_ARITY[pred]
        for arg_idx in range(arity):
            key = (pred, arg_idx)

            # ONTOP and INSIDE are cross-arg predicates whose arg1 restrictions
            # depend on which arg0 was chosen.  compute_sd_arg1_allowed_objects
            # handles the per-combo filtering; here we produce the widest
            # permissive set so that the arg0 trie is not over-restricted.
            #
            # ONTOP arg0: character, CLOTHES or GRABBABLE object
            # ONTOP arg1: permissive non-room (refined per arg0 at generation time)
            # INSIDE arg0: character OR any GRABBABLE object
            # INSIDE arg1: permissive non-room+rooms (refined per arg0 at generation time)
            if pred == "ONTOP":
                if arg_idx == 0:
                    # arg0 must be character (SIT/LIE) or CLOTHES (PUTON) or GRABBABLE (PUTBACK)
                    clothes = {n for n in all_names if _has_any(n, {"CLOTHES"})}
                    grabbable = {n for n in all_names if _has_any(n, {"GRABBABLE"})}
                    result[key] = _prune_arg0_without_arg1(
                        pred, char_names | clothes | grabbable, objects, properties, categories)
                else:
                    # arg1: permissive non-room (refined per arg0 at generation time)
                    result[key] = set(non_room_names)
                continue

            if pred == "INSIDE":
                if arg_idx == 0:
                    # character or any object with GRABBABLE
                    grabbable = {n for n in all_names if _has_any(n, {"GRABBABLE"})}
                    result[key] = _prune_arg0_without_arg1(
                        pred, char_names | grabbable, objects, properties, categories)
                else:
                    # arg1: rooms + non-room non-char containers; refined later
                    result[key] = set(all_names)
                continue

            rule = SD_AFFORDANCE_RULES.get(key)

            if rule is None:
                result[key] = set(all_names)
            elif rule == "character_only":
                result[key] = set(char_names)
            elif isinstance(rule, set):
                allowed = {n for n in all_names if _has_any(n, rule)}
                result[key] = allowed
            else:
                raise ValueError(f"Unknown affordance rule type: {rule}")

    return result



def _prune_arg0_without_arg1(
    pred: str,
    arg0_candidates: Set[str],
    objects: Dict[str, list],
    properties: Dict[str, Set[str]],
    categories: Dict[str, str],
) -> Set[str]:
    """Dead-end freeness of the γ automaton.

    For the cross-argument predicates ONTOP/INSIDE the arg1 set is refined per
    arg0 at generation time; when that refined set is EMPTY (e.g. `ONTOP(character,`
    in a scene with no SITTABLE/LIEABLE object, `INSIDE(character,` with no room
    listed) the automaton would admit the arg0 and then have no admissible token.
    Such prefixes have no completion, so removing them leaves the accepted
    language unchanged and makes hard masking alone sufficient for syntax.
    """
    kept: Set[str] = set()
    for name in arg0_candidates:
        allowed_arg1 = compute_sd_arg1_allowed_objects(pred, name, objects, properties, categories)
        if allowed_arg1 is None or len(allowed_arg1) > 0:
            kept.add(name)
    return kept

def compute_sd_arg1_allowed_objects(
    pred: str,
    arg0_name: str,
    objects: Dict[str, list],
    properties: Dict[str, Set[str]],
    categories: Dict[str, str],
) -> Optional[Set[str]]:
    """
    Compute the set of allowed arg1 object names for a cross-arg predicate
    (ONTOP or INSIDE) given that arg0 has already been chosen as arg0_name.

    Returns None if there is no restriction (any scene object is fine),
    or a set of allowed names (possibly empty, meaning no valid arg1 exists).

    Rules follow the evaluator's subgoal-to-action translation, which
    dispatches on CHARACTER IDENTITY, not properties:

    ONTOP(arg0, arg1):
      - arg0 IS character        → SIT/LIE: arg1 must have SITTABLE or LIEABLE
      - arg1 IS character        → PUTON: arg0 must have CLOTHES
      - neither is character     → GRAB+PUTBACK: arg0 must have GRABBABLE, no check on arg1
    Given arg0, arg1 can be the UNION of applicable cases:
      - If arg0 = character: arg1 = {SITTABLE ∪ LIEABLE objects}
      - If arg0 ≠ character: arg1 = {character if arg0 has CLOTHES} ∪
                                     {non-room non-char if arg0 has GRABBABLE}

    INSIDE(arg0, arg1):
      - arg0 IS character → WALK: arg1 must be a room
      - arg0 NOT character → GRAB+PUTIN: arg0 must be GRABBABLE, arg1 = non-room
    """
    all_names = set(objects.keys())
    room_names = {n for n, c in categories.items() if c == "Rooms"}
    char_names = {n for n, c in categories.items() if c == "Characters"}
    non_room_names = all_names - room_names
    non_room_non_char = all_names - room_names - char_names

    def _has_any(name: str, required_props: Set[str]) -> bool:
        return bool(required_props & properties.get(name, set()))

    arg0_is_char = arg0_name in char_names
    arg0_props = properties.get(arg0_name, set())

    if pred == "ONTOP":
        if arg0_is_char:
            # SIT/LIE: arg1 must have SITTABLE or LIEABLE
            return {n for n in all_names if _has_any(n, {"SITTABLE", "LIEABLE"})}
        else:
            # arg0 is NOT character — arg1 is the UNION of:
            #   - character (PUTON case: requires CLOTHES on arg0)
            #   - non-room non-char (PUTBACK case: requires GRABBABLE on arg0)
            allowed = set()
            if "CLOTHES" in arg0_props:
                allowed |= char_names
            if "GRABBABLE" in arg0_props:
                allowed |= non_room_non_char
            return allowed

    if pred == "INSIDE":
        if arg0_is_char:
            # WALK: arg1 must be a room
            return set(room_names)
        elif "GRABBABLE" in arg0_props:
            # GRAB+PUTIN: arg1 must be a non-room container
            return set(non_room_names)
        else:
            # No valid scenario — reject
            return set()

    # For non-cross-arg predicates, return None (no restriction from arg0)
    return None

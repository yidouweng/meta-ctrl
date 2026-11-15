"""
BEHAVIOR action-sequencing β: the semantic automaton over grounded actions.

Tracks a simplified world state:
- object locations (location type + surface/container canonical id)
- what each hand holds (canonical id or HOLD_NONE)
- property bits (open, toggled_on, soaked, stained, ...)
- goal coverage: which target predicates are satisfied

Each grounded action updates the world state if its preconditions hold; a
state accepts when every goal is satisfied.
"""
from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


HOLD_NONE = -1  # sentinel for empty hand

class BehBetaState(NamedTuple):
    """BEHAVIOR β state."""
    goal_bits: int      # bitmask of satisfied goals
    holding_L: int      # canonical ID of object in left hand, or HOLD_NONE
    holding_R: int      # canonical ID of object in right hand, or HOLD_NONE
    loc_hash: int       # hash of tracked object locations
    prop_bits: int      # bitmask of tracked property states (toggled_on, etc.)


@dataclass
class BehGoal:
    """A parsed target goal."""
    predicate: str      # 'onfloor', 'ontop', 'inside', 'nextto', etc.
    obj_cid: int        # canonical ID of the object
    target_cid: int     # canonical ID of the surface/container (-1 for property goals)
    goal_idx: int       # position in goal list


@dataclass
class BehStep:
    """A grounded action step. Same interface as VHStep/SDStep."""
    action: str
    args: Tuple[int, ...]


# Location types
LOC_ONFLOOR = 0
LOC_ONTOP = 1
LOC_INSIDE = 2
LOC_UNKNOWN = 5
LOC_NEXTTO = 6
LOC_UNDER = 7

LOCATION_PREDICATES = frozenset({'onfloor', 'ontop', 'inside', 'nextto', 'under'})

# Membership (openable / toggleable / cleaner / water-source / graspable /
# containable / slicer) comes from the shared taxonomy layer
# `ctrlg.beta.beh_affordances.BehAffordances`, built from the prompt's
# {'name','category'} object list. The γ side reads the same layer
# (action_defs.compute_behavior_affordances).
#
#   * open-gating: an `open` bit exists only for open-gated containers in
#     container role (target of an `inside` goal, or a container that
#     initially holds an object). Bits default to closed unless the
#     prompt's initial states say `open(X)`. Objects without an open state
#     get no bit and are never gated.
#   * toggleable = T:toggleable; cleaner = T:cleaningTool ∪ ISA cloth/brush;
#     water source = T:waterSource ∪ (toggleable ∧ plumbing-fixture ISA)
#     ∪ prompt-stated vessels-of-water; slicer = T:slicer.
#   * graspability / containability: ISA-subtree classes ∪
#     name-structural room/floor/wall/agent rules; containability adds the
#     prompt's own container-role evidence.


class BehBetaStateMachine:
    """β automaton for BEHAVIOR action sequencing.

    Args:
        initial_states / target_states: prompt predicate lists.
        all_instances, inst_to_cid: scene instances and their canonical ids.
        gamma_valid_actions: grounded (action, cids) pairs γ can emit.
        max_states: cap on discovered states during the forward BFS.
        max_budget: BFS depth bound (plan length).
        blocked_actions: action names that are never allowed.
        objects_meta: prompt object list [{'name', 'category'}] for the
            taxonomy membership layer. Quantified goals are expanded by the
            caller.
    """

    def __init__(
        self,
        initial_states: List[List],
        target_states: List[List],
        all_instances: List[str],
        inst_to_cid: Dict[str, int],
        gamma_valid_actions: List[Tuple[str, Tuple[int, ...]]],
        max_states: int = 150000,
        max_budget: int = 40,
        debug: bool = False,
        blocked_actions: Optional[Set[str]] = None,
        objects_meta: Optional[List[Dict]] = None,
    ):
        self.debug = debug
        self.max_states = max_states
        self.all_instances = all_instances
        self.inst_to_cid = inst_to_cid
        self.cid_to_inst = {v: k for k, v in inst_to_cid.items()}
        self.gamma_valid_actions = gamma_valid_actions
        self.blocked_actions: Set[str] = blocked_actions or set()
        self._build_max_budget = max_budget

        # Shared taxonomy membership layer (same layer as the γ side).
        from .beh_affordances import BehAffordances
        self.membership = BehAffordances(objects=objects_meta)

        # Parse goals (may include 'not_<prop>' predicates from expansion).
        self.goals: List[BehGoal] = []
        self._parse_goals(target_states)
        self.n_goals = len(self.goals)

        # Parse initial world state locations + save property predicates for later.
        # Must come BEFORE _identify_object_properties so that tracked_cids can
        # include objects with initial locations (not just goal objects).
        self._parse_initial_states(initial_states)

        # Identify non-graspable and non-containable objects; build tracked_cids.
        self._identify_object_properties()

        # Build property tracking: map (predicate, cid) → prop_bit index.
        # Needs init_locations so we can allocate 'open' bits for any container
        # that initially holds a tracked object.
        self._setup_property_tracking()

        # Now that property bits are allocated, compute the initial prop_bits
        # (re-scan saved property predicates).
        self._finalize_initial_prop_bits()

        # Precompute the set of surface cids whose identity is preserved in
        # loc_hash. Any other surface collapses to a sentinel, so β states
        # that differ only in goal-irrelevant + precondition-irrelevant
        # surface identity merge. Preserved:
        #   - goal target cids (goal satisfaction check reads surface)
        #   - grasp-open-required container cids (GRASP precondition reads
        #     surface to check container-open)
        #   - water-source cids (SOAK precondition reads surface to check
        #     "cleaner inside toggled-on sink/bathtub/teapot")
        #   - every container with an `open` bit (the accessibility check
        #     walks INSIDE chains through them)
        self._open_tracked_cids: Set[int] = {
            cid for (cid, pred) in self._prop_bit_by_cid_pred if pred == 'open'}
        self._preserved_surface_cids: Set[int] = set()
        for g in self.goals:
            if g.target_cid >= 0:
                self._preserved_surface_cids.add(g.target_cid)
        self._preserved_surface_cids |= self._grasp_open_required_cids
        self._preserved_surface_cids |= self._water_source_cids
        self._preserved_surface_cids |= self._open_tracked_cids

        # Untracked objects are not in loc_hash, so their recorded location
        # must not depend on the path: it stays the initial one (see
        # _apply_action). Recording their moves exactly multiplies the states
        # (every grasp of an object that starts in a container is a new
        # state) and pushes many automata to the state cap.
        tracked = set(self.tracked_cids)
        self._untracked_cids: Set[int] = {
            cid for cid in self.inst_to_cid.values() if cid not in tracked}

        # Build state machine
        self.states: List[BehBetaState] = []
        self.state_to_id: Dict[BehBetaState, int] = {}
        self.transitions: Dict[int, List[Tuple[BehStep, int]]] = {}
        self.accept_mask: List[bool] = []
        self._build()

    def _parse_goals(self, target_states: List[List]):
        """Parse target predicates into BehGoal objects."""
        idx = 0
        skipped = 0
        for pred_list in target_states:
            if not pred_list or len(pred_list) < 2:
                continue
            pred = pred_list[0]

            if pred in ('forall', 'exists', 'forpairs', 'forn', 'and', 'or', 'not'):
                skipped += 1
                continue

            # Simple binary predicate: ['onfloor', 'obj', 'surface']
            if pred in ('onfloor', 'ontop', 'inside', 'nextto', 'under') and len(pred_list) >= 3:
                obj_cid = self.inst_to_cid.get(pred_list[1], -1)
                target_cid = self.inst_to_cid.get(pred_list[2], -1)
                if obj_cid >= 0 and target_cid >= 0:
                    self.goals.append(BehGoal(
                        predicate=pred, obj_cid=obj_cid,
                        target_cid=target_cid, goal_idx=idx,
                    ))
                    idx += 1

            # Simple unary property predicate: ['toggled_on', 'obj']  (must be SET)
            elif pred in ('sliced', 'soaked', 'frozen', 'cooked', 'toggled_on', 'stained', 'dusty', 'open'):
                if len(pred_list) >= 2:
                    obj_cid = self.inst_to_cid.get(pred_list[1], -1)
                    if obj_cid >= 0:
                        self.goals.append(BehGoal(
                            predicate=pred, obj_cid=obj_cid,
                            target_cid=-1, goal_idx=idx,
                        ))
                        idx += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

            # Negated unary property: ['not_stained', 'obj']  (must be CLEARED)
            elif pred.startswith('not_') and pred[4:] in (
                'sliced', 'soaked', 'frozen', 'cooked', 'toggled_on',
                'stained', 'dusty', 'open',
            ):
                if len(pred_list) >= 2:
                    obj_cid = self.inst_to_cid.get(pred_list[1], -1)
                    if obj_cid >= 0:
                        self.goals.append(BehGoal(
                            predicate=pred, obj_cid=obj_cid,
                            target_cid=-1, goal_idx=idx,
                        ))
                        idx += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

        if self.debug:
            print(f"[BehBetaSM] {len(self.goals)} goals parsed, {skipped} skipped")

    # Action → property predicate mapping.
    # Each action can SET one or more properties, CLEAR others.
    # For not_<pred> goals we track the positive bit and the goal requires it cleared.
    _ACTION_SETS_PROPS: Dict[str, Tuple[str, ...]] = {
        'TOGGLE_ON': ('toggled_on',),
        'SLICE': ('sliced',),
        'SOAK': ('soaked',),
        'COOK': ('cooked',),
        'FREEZE': ('frozen',),
        'OPEN': ('open',),
    }
    _ACTION_CLEARS_PROPS: Dict[str, Tuple[str, ...]] = {
        'TOGGLE_OFF': ('toggled_on',),
        'UNFREEZE': ('frozen',),
        'DRY': ('soaked',),
        'CLOSE': ('open',),
        'CLEAN': ('stained', 'dusty'),
    }

    def _setup_property_tracking(self):
        """Build property bit indices for goal-relevant properties AND open/close tracking.

        Tracked properties:
        1. Property goals (toggled_on, sliced, not_stained, etc.) — from target_states.
           Negated goals (not_stained) track the POSITIVE bit (stained); the goal
           just requires the bit to be cleared.
        2. Open state for PLACE_INSIDE targets — containers that may need OPEN before placing
        """
        self._prop_bit_by_cid_pred: Dict[Tuple[int, str], int] = {}  # (cid, pred) → bit
        bit_idx = 0

        # Property goals (both positive and negated)
        for g in self.goals:
            if g.target_cid != -1:
                continue
            # Underlying property name (strip not_ prefix if present)
            base_pred = g.predicate[4:] if g.predicate.startswith('not_') else g.predicate
            if (g.obj_cid, base_pred) not in self._prop_bit_by_cid_pred:
                self._prop_bit_by_cid_pred[(g.obj_cid, base_pred)] = bit_idx
                bit_idx += 1

        # Open tracking for PLACE_INSIDE targets (containers). An `open`
        # bit exists only for open-gated containers; other targets (bowls,
        # trays, open-top baskets, ...) get no bit and are never gated.
        self._open_required_cids: Set[int] = set()
        for g in self.goals:
            if g.predicate == 'inside' and g.target_cid >= 0:
                cid = g.target_cid
                if cid in self._open_required_cids:
                    continue
                if not self.membership.open_gated(self.cid_to_inst.get(cid, '')):
                    continue  # nothing to open (non-openable or open-top vessel)
                self._open_required_cids.add(cid)
                if (cid, 'open') not in self._prop_bit_by_cid_pred:
                    self._prop_bit_by_cid_pred[(cid, 'open')] = bit_idx
                    bit_idx += 1

        # Open tracking for ALL containers that initially hold ANY object: the
        # simulator checks open-state regardless of whether the inside object is
        # a tracked goal (simulator constraint). We allocate an 'open' bit for every such container so
        # GRASP on any object initially inside a closed container is blocked.
        #
        # This is independent of goal tracking — a pure simulator-precondition
        # filter. Goal dropping does not weaken this filter.
        self._grasp_open_required_cids: Set[int] = set()
        # Map grasp-source object → its initial container (for GRASP precondition)
        self._init_inside_container: Dict[int, int] = {}
        for obj_cid, (loc_type, surface_cid) in self.init_locations.items():
            if loc_type != LOC_INSIDE or surface_cid < 0:
                continue
            self._init_inside_container[obj_cid] = surface_cid
            if surface_cid in self._grasp_open_required_cids:
                continue
            # Only open-gated containers gate (taxonomy-openable minus
            # open-top vessels).
            if not self.membership.open_gated(
                    self.cid_to_inst.get(surface_cid, '')):
                continue
            self._grasp_open_required_cids.add(surface_cid)
            if (surface_cid, 'open') not in self._prop_bit_by_cid_pred:
                self._prop_bit_by_cid_pred[(surface_cid, 'open')] = bit_idx
                bit_idx += 1

        # Capability tracking: allocate property bits for every (obj, prop)
        # mentioned in initial states. This extends no-op detection to
        # non-goal objects and establishes a per-object "supported properties"
        # set. Property actions on objects with NO supported property bits
        # are blocked (the object doesn't have that state).
        for obj_cid, pred in getattr(self, '_init_prop_preds', ()):
            if pred not in ('stained', 'dusty', 'soaked', 'toggled_on', 'open',
                            'sliced', 'frozen', 'cooked'):
                continue
            if (obj_cid, pred) not in self._prop_bit_by_cid_pred:
                self._prop_bit_by_cid_pred[(obj_cid, pred)] = bit_idx
                bit_idx += 1

        # Other openable objects get no `open` bit, so PLACE_INSIDE into
        # them is unconstrained; this keeps the state small.

        # Cleaner tracking: allocate `soaked` bit for every cleaner in scene.
        # Needed so SOAK can transition it 0→1 and CLEAN can require it.
        self._cleaner_cids: Set[int] = set()
        for inst, cid in self.inst_to_cid.items():
            if self.membership.cleaner(inst):
                self._cleaner_cids.add(cid)
                if (cid, 'soaked') not in self._prop_bit_by_cid_pred:
                    self._prop_bit_by_cid_pred[(cid, 'soaked')] = bit_idx
                    bit_idx += 1

        # Water-source tracking (taxonomy waterSource ∪ toggleable
        # plumbing fixtures ∪ prompt-stated vessels-of-water). Allocate
        # `toggled_on` bit so TOGGLE_ON can transition it 0→1 and SOAK can
        # require the target to be INSIDE a toggled-on source.
        from .beh_affordances import prompt_water_vessels
        _init_inside_name_pairs = [
            (self.cid_to_inst.get(o, ''), self.cid_to_inst.get(s, ''))
            for o, (t, s) in self.init_locations.items()
            if t == LOC_INSIDE and s >= 0
        ]
        _water_vessels = prompt_water_vessels(
            _init_inside_name_pairs, self.membership)
        self._water_source_cids: Set[int] = set()
        # Passive water sources (kettle/teapot/pool, prompt vessels-of-water)
        # provide water with no switch: SOAK does not require them
        # toggled_on, and they get no toggled_on bit here.
        self._passive_water_cids: Set[int] = set()
        for inst, cid in self.inst_to_cid.items():
            is_vessel = inst in _water_vessels
            if self.membership.water_source(inst) or is_vessel:
                self._water_source_cids.add(cid)
                if (self.membership.passive_water_source(inst)
                        or (is_vessel and not self.membership.toggleable(inst))):
                    self._passive_water_cids.add(cid)
                    continue
                if (cid, 'toggled_on') not in self._prop_bit_by_cid_pred:
                    self._prop_bit_by_cid_pred[(cid, 'toggled_on')] = bit_idx
                    bit_idx += 1

        # Toggleable tracking (taxonomy `toggleable`): appliances, lights,
        # electronics. Allocate `toggled_on` bit so β can detect no-op
        # TOGGLE on already-toggled targets and so property-only goals on
        # these objects work correctly.
        for inst, cid in self.inst_to_cid.items():
            if self.membership.toggleable(inst):
                if (cid, 'toggled_on') not in self._prop_bit_by_cid_pred:
                    self._prop_bit_by_cid_pred[(cid, 'toggled_on')] = bit_idx
                    bit_idx += 1

        self.n_prop_bits = bit_idx
        if self.debug and self.n_prop_bits > 0:
            print(f"[BehBetaSM] Property tracking: {self.n_prop_bits} bits")
            for (cid, pred), bit in sorted(self._prop_bit_by_cid_pred.items(), key=lambda x: x[1]):
                print(f"  bit {bit}: {pred}({self.cid_to_inst.get(cid, '?')})")

    def _apply_prop_change(self, prop_bits: int, action: str, target_cid: int) -> int:
        """Update prop_bits based on action effect."""
        for pred in self._ACTION_SETS_PROPS.get(action, ()):
            bit = self._prop_bit_by_cid_pred.get((target_cid, pred))
            if bit is not None:
                prop_bits |= (1 << bit)
        for pred in self._ACTION_CLEARS_PROPS.get(action, ()):
            bit = self._prop_bit_by_cid_pred.get((target_cid, pred))
            if bit is not None:
                prop_bits &= ~(1 << bit)
        return prop_bits

    def _identify_object_properties(self):
        """Identify non-graspable and non-containable objects.

        Membership comes from the shared taxonomy layer (ISA classes +
        name-structural room/floor/wall/agent rules). Containability adds
        the prompt's own container-role evidence: anything the prompt uses
        as the target of an `inside` relation (goal or initial state) is a
        container regardless of class membership.
        """
        self.non_graspable: Set[int] = set()
        self.non_containable: Set[int] = set()

        # Container-role names: inside-goal targets + initial-inside holders.
        container_roles: Set[str] = set()
        for g in self.goals:
            if g.predicate == 'inside' and g.target_cid >= 0:
                container_roles.add(self.cid_to_inst.get(g.target_cid, ''))
        for _obj, (loc_type, surface_cid) in self.init_locations.items():
            if loc_type == LOC_INSIDE and surface_cid >= 0:
                container_roles.add(self.cid_to_inst.get(surface_cid, ''))

        # Goal-implied graspability: a task whose goals reposition X (X is
        # arg0 of a location goal) presupposes X can be picked up and
        # carried, so the goal overrides the ISA class default (e.g.
        # furniture-rearranging tasks move lamps and chairs).
        # Floors/walls/agent stay non-graspable regardless.
        repositioned = {g.obj_cid for g in self.goals if g.predicate in LOCATION_PREDICATES}

        def _structurally_fixed(inst: str) -> bool:
            b = inst.split('.', 1)[0]
            return (inst.startswith('room_floor') or inst.startswith('room_')
                    or b in ('floor', 'wall', 'ceiling') or 'agent' in inst)

        for inst, cid in self.inst_to_cid.items():
            if self.membership.non_graspable(inst):
                if cid in repositioned and not _structurally_fixed(inst):
                    pass  # goal-implied graspable
                else:
                    self.non_graspable.add(cid)
            if self.membership.non_containable(inst, container_role_names=container_roles):
                self.non_containable.add(cid)

        # Goal-driven scope: track location only for objects whose location
        # matters for goal satisfaction or for a relevant precondition.
        # Property-only goals (e.g. not_stained) don't need location
        # tracking for goal bits, but SOAK's precondition "cleaner inside
        # toggled-on water source" is loc-based, so we keep cleaners in
        # tracked_cids. Surface/container targets are fixed-location and
        # excluded.
        loc_goal_obj_cids = repositioned

        # Cleaner objects (taxonomy cleaningTool ∪ cloth/brush ISA) — their
        # location is read by SOAK's precondition ("inside toggled-on water
        # source"). Keep them tracked so β enforces the SOAK→CLEAN chain.
        cleaner_cids = {cid for inst, cid in self.inst_to_cid.items()
                        if self.membership.cleaner(inst)}

        candidate_cids = (loc_goal_obj_cids | cleaner_cids) - self.non_graspable
        self.tracked_cids: List[int] = sorted(candidate_cids)
        self.n_tracked = len(self.tracked_cids)

        if self.debug:
            tracked_names = [self.cid_to_inst.get(c, '?') for c in self.tracked_cids]
            ng_names = [self.cid_to_inst.get(c, '?') for c in sorted(self.non_graspable)]
            print(f"[BehBetaSM] Tracked: {self.n_tracked}: {tracked_names}")
            print(f"[BehBetaSM] Non-graspable: {ng_names}")

    def _parse_initial_states(self, initial_states: List[List]):
        """Parse initial world state locations and save property predicates.

        Only locations are materialized here; prop_bits is computed later in
        `_finalize_initial_prop_bits` once property bit indices are allocated.
        """
        self.init_locations: Dict[int, Tuple[int, int]] = {}
        self.init_prop_bits: int = 0
        self._init_prop_preds: List[Tuple[int, str]] = []  # (obj_cid, pred)

        for pred_list in initial_states:
            if not pred_list or len(pred_list) < 2:
                continue
            pred = pred_list[0]

            if pred in ('onfloor', 'ontop', 'inside') and len(pred_list) >= 3:
                obj_name = pred_list[1]
                surface_name = pred_list[2]

                if 'agent' in obj_name.lower():
                    continue  # the agent's own location is not tracked

                obj_cid = self.inst_to_cid.get(obj_name, -1)
                surface_cid = self.inst_to_cid.get(surface_name, -1)
                if obj_cid >= 0 and surface_cid >= 0:
                    loc_type = {'onfloor': LOC_ONFLOOR, 'ontop': LOC_ONTOP, 'inside': LOC_INSIDE}[pred]
                    self.init_locations[obj_cid] = (loc_type, surface_cid)

            # Property predicates: save for later finalization.
            elif pred in ('stained', 'dusty', 'soaked', 'toggled_on', 'open',
                           'sliced', 'frozen', 'cooked') and len(pred_list) >= 2:
                obj_name = pred_list[1]
                if isinstance(obj_name, str) and 'agent' not in obj_name.lower():
                    obj_cid = self.inst_to_cid.get(obj_name, -1)
                    if obj_cid >= 0:
                        self._init_prop_preds.append((obj_cid, pred))

        if self.debug:
            print(f"[BehBetaSM] Initial: {len(self.init_locations)} object locations, "
                  f"{len(self._init_prop_preds)} property predicates")

    def _finalize_initial_prop_bits(self):
        """Compute init_prop_bits now that property bit indices are allocated.

        Open bits default to 0 (closed) unless the prompt's initial states
        say `open(X)`. Objects without an open state have no bit and are
        never gated.
        """
        self.init_prop_bits = 0
        for obj_cid, pred in self._init_prop_preds:
            bit = self._prop_bit_by_cid_pred.get((obj_cid, pred))
            if bit is not None:
                self.init_prop_bits |= (1 << bit)

        if self.debug:
            print(f"[BehBetaSM] init_prop_bits={bin(self.init_prop_bits)} "
                  f"(n_prop_bits={self.n_prop_bits})")

    def _encode_locations(self, locations: Dict[int, Tuple[int, int]]) -> int:
        # Canonicalize surface_cid to the minimal identity needed:
        # keep it only when it affects goal satisfaction (goal target)
        # or a precondition (e.g. GRASP open-check for initial-inside
        # containers). All other surfaces collapse to the sentinel -1.
        # This keeps `loc_hash` stable across states that differ only in
        # goal-irrelevant/precondition-irrelevant surface identity,
        # dramatically shrinking reachable β state count.
        preserved = self._preserved_surface_cids
        parts = []
        for cid in self.tracked_cids:
            loc_type, surface = locations.get(cid, (LOC_UNKNOWN, -1))
            canon = surface if surface in preserved else -1
            parts.append((loc_type, canon + 1))
        return hash(tuple(parts))

    def _get_initial_state(self) -> BehBetaState:
        loc_hash = self._encode_locations(self.init_locations)
        # Evaluate initial goal_bits from starting prop_bits + locations.
        init_goal_bits = 0
        for g in self.goals:
            if self._check_goal(self.init_locations, g, self.init_prop_bits):
                init_goal_bits |= (1 << g.goal_idx)
        return BehBetaState(
            goal_bits=init_goal_bits, holding_L=HOLD_NONE, holding_R=HOLD_NONE,
            loc_hash=loc_hash, prop_bits=self.init_prop_bits,
        )

    def _check_goal(self, locations: Dict[int, Tuple[int, int]], goal: BehGoal,
                    prop_bits: int = 0) -> bool:
        # Property goal: check prop_bits (handle negated predicates too).
        if goal.target_cid == -1:
            negated = goal.predicate.startswith('not_')
            base_pred = goal.predicate[4:] if negated else goal.predicate
            bit = self._prop_bit_by_cid_pred.get((goal.obj_cid, base_pred))
            if bit is None:
                return False
            is_set = bool(prop_bits & (1 << bit))
            return (not is_set) if negated else is_set
        # Location goal: check locations
        loc = locations.get(goal.obj_cid)
        if loc is None:
            return False
        loc_type, surface_cid = loc
        if goal.predicate == 'onfloor':
            return loc_type == LOC_ONFLOOR and surface_cid == goal.target_cid
        elif goal.predicate == 'ontop':
            return loc_type == LOC_ONTOP and surface_cid == goal.target_cid
        elif goal.predicate == 'nextto':
            return loc_type == LOC_NEXTTO and surface_cid == goal.target_cid
        elif goal.predicate == 'under':
            return loc_type == LOC_UNDER and surface_cid == goal.target_cid
        elif goal.predicate == 'inside':
            return loc_type == LOC_INSIDE and surface_cid == goal.target_cid
        return False

    def _check_open(self, prop_bits: int, target_cid: int) -> bool:
        """Check if target container is open (for PLACE_INSIDE precondition).

        Applies to any container with an allocated 'open' bit — includes:
          - inside-goal targets
          - initial-state-inside-source containers
          - containers mentioned with `open` / `not open` in initial or target
        """
        bit = self._prop_bit_by_cid_pred.get((target_cid, 'open'))
        if bit is None:
            return True  # not tracked → no constraint (naturally-open or unknown)
        return bool(prop_bits & (1 << bit))

    def _grasp_container_open(self, locations: Dict[int, Tuple[int, int]],
                              target_cid: int, prop_bits: int,
                              held: Tuple[int, int] = ()) -> bool:
        """Walk up the containment chain; reject if any ancestor with a
        tracked `open` bit is currently closed.

        Mirrors the simulator's `interactivity_precond`, which blocks any
        action targeting an object with a closed INSIDE-ancestor
        (simulator constraint).

        Applied to GRASP, PLACE_* and property actions. A held object (and
        whatever is inside it) is in the agent's hand, hence accessible.
        Other objects absent from `locations` fall back to their initial
        container in `_init_inside_container`.
        """
        seen: Set[int] = set()
        cur = target_cid
        while cur not in seen:
            seen.add(cur)
            if cur in held:
                return True
            # Find the immediate container of `cur`. Prefer the dynamic
            # `locations` entry (post-PLACE updates); fall back to the
            # static `_init_inside_container` for objects not in the
            # dynamic map.
            container: Optional[int] = None
            loc = locations.get(cur)
            if loc is not None:
                loc_type, surf = loc
                if loc_type == LOC_INSIDE and surf >= 0:
                    container = surf
            else:
                container = self._init_inside_container.get(cur)
            if container is None:
                return True  # no (more) ancestors — target is accessible
            bit = self._prop_bit_by_cid_pred.get((container, 'open'))
            if bit is not None and not (prop_bits & (1 << bit)):
                return False  # ancestor is closed → simulator rejects
            cur = container
        return True

    def _is_transitively_inside(self, locations: Dict[int, Tuple[int, int]],
                                obj_cid: int, ancestor_cid: int) -> bool:
        """True if `obj_cid` is (transitively) INSIDE `ancestor_cid`.

        Used to reject CIRCULAR containment: placing a held object INSIDE/ONTOP a
        `target` that is already inside the held object (e.g. plate_0 inside
        plate_4, then GRASP plate_4 and PLACE_INSIDE plate_0 → plate_4 inside
        plate_0 inside plate_4). Such placements are physically impossible and
        the evaluator rejects them (physical constraint). Legitimate moves (A from
        B to C) are unaffected; only target-inside-held is forbidden.
        """
        seen: Set[int] = set()
        cur = obj_cid
        while cur not in seen:
            seen.add(cur)
            loc = locations.get(cur)
            container = loc[1] if (loc is not None and loc[0] == LOC_INSIDE and loc[1] >= 0) else None
            if container is None:
                return False
            if container == ancestor_cid:
                return True
            cur = container
        return False


    def _apply_action(self, locations, holding_L, holding_R, action, cids,
                      prop_bits: int = 0):
        """Apply action to world state. Returns (new_locations, new_L, new_R) or None.

        Untracked objects keep their recorded (initial) location, so that the
        locations stored with a state are a function of the state.
        """
        result = self._apply_action_raw(locations, holding_L, holding_R, action, cids,
                                        prop_bits=prop_bits)
        if result is None:
            return None
        new_locs, new_L, new_R = result
        moved = [c for c in (holding_L, holding_R, *cids)
                 if c in self._untracked_cids and new_locs.get(c) != locations.get(c)]
        if moved:
            new_locs = dict(new_locs)
            for c in moved:
                if c in locations:
                    new_locs[c] = locations[c]
                else:
                    new_locs.pop(c, None)
        return new_locs, new_L, new_R

    def _apply_action_raw(self, locations, holding_L, holding_R, action, cids,
                          prop_bits: int = 0):
        """The action's effect on (locations, left hand, right hand), or None."""
        if not cids:
            return None

        # Per-prompt blocked actions
        if action in self.blocked_actions:
            return None

        target_cid = cids[0]

        # --- NAVIGATE_TO: no state change ---
        if action == 'NAVIGATE_TO':
            return (locations, holding_L, holding_R)  # no loc change → share ref (never mutated in place)

        # --- GRASP: hand empty, target graspable, not already held,
        # and, if target is inside a tracked container, container must be open ---
        if action == 'LEFT_GRASP':
            if holding_L != HOLD_NONE or target_cid == holding_R:
                return None
            if target_cid in self.non_graspable:
                return None
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            new_locs = dict(locations)
            new_locs.pop(target_cid, None)
            return (new_locs, target_cid, holding_R)

        if action == 'RIGHT_GRASP':
            if holding_R != HOLD_NONE or target_cid == holding_L:
                return None
            if target_cid in self.non_graspable:
                return None
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            new_locs = dict(locations)
            new_locs.pop(target_cid, None)
            return (new_locs, holding_L, target_cid)

        # --- PLACE variants: must hold, target not held ---
        # Each variant produces a different location type matching the evaluator's semantics
        def _place_loc(act, tgt_cid):
            is_floor = self.cid_to_inst.get(tgt_cid, '').startswith('room_floor')
            if 'NEXTTO' in act and 'ONTOP' not in act:
                return LOC_NEXTTO
            elif 'UNDER' in act:
                return LOC_UNDER
            elif is_floor:
                return LOC_ONFLOOR
            else:
                return LOC_ONTOP

        if action in ('LEFT_PLACE_ONTOP', 'LEFT_PLACE_NEXTTO', 'LEFT_PLACE_UNDER',
                       'LEFT_PLACE_NEXTTO_ONTOP'):
            if holding_L == HOLD_NONE:
                return None
            if target_cid == holding_L or target_cid == holding_R:
                return None
            if self._is_transitively_inside(locations, target_cid, holding_L):
                return None  # circular: target already inside held object
            # Target must not sit inside a closed container (simulator constraint).
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            new_locs = dict(locations)
            new_locs[holding_L] = (_place_loc(action, target_cid), target_cid)
            return (new_locs, HOLD_NONE, holding_R)

        if action in ('RIGHT_PLACE_ONTOP', 'RIGHT_PLACE_NEXTTO', 'RIGHT_PLACE_UNDER',
                       'RIGHT_PLACE_NEXTTO_ONTOP'):
            if holding_R == HOLD_NONE:
                return None
            if target_cid == holding_L or target_cid == holding_R:
                return None
            if self._is_transitively_inside(locations, target_cid, holding_R):
                return None  # circular: target already inside held object
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            new_locs = dict(locations)
            new_locs[holding_R] = (_place_loc(action, target_cid), target_cid)
            return (new_locs, holding_L, HOLD_NONE)

        # --- PLACE_INSIDE: must hold, target not held, target containable AND open ---
        if action == 'LEFT_PLACE_INSIDE':
            if holding_L == HOLD_NONE:
                return None
            if target_cid == holding_L or target_cid == holding_R:
                return None
            if target_cid in self.non_containable:
                return None
            if not self._check_open(prop_bits, target_cid):
                return None  # container is closed
            # Transitive: target itself can't be inside a closed parent.
            if self._is_transitively_inside(locations, target_cid, holding_L):
                return None  # circular: target already inside held object
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            new_locs = dict(locations)
            new_locs[holding_L] = (LOC_INSIDE, target_cid)
            return (new_locs, HOLD_NONE, holding_R)

        if action == 'RIGHT_PLACE_INSIDE':
            if holding_R == HOLD_NONE:
                return None
            if target_cid == holding_L or target_cid == holding_R:
                return None
            if target_cid in self.non_containable:
                return None
            if not self._check_open(prop_bits, target_cid):
                return None  # container is closed
            if self._is_transitively_inside(locations, target_cid, holding_R):
                return None  # circular: target already inside held object
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            new_locs = dict(locations)
            new_locs[holding_R] = (LOC_INSIDE, target_cid)
            return (new_locs, holding_L, HOLD_NONE)

        # --- RELEASE: must hold, target must be the held object ---
        if action == 'LEFT_RELEASE':
            if holding_L == HOLD_NONE:
                return None
            if target_cid != holding_L:
                return None  # can only release what you're holding
            new_locs = dict(locations)
            new_locs[holding_L] = (LOC_ONFLOOR, -1)  # dropped at current location
            return (new_locs, HOLD_NONE, holding_R)

        if action == 'RIGHT_RELEASE':
            if holding_R == HOLD_NONE:
                return None
            if target_cid != holding_R:
                return None  # can only release what you're holding
            new_locs = dict(locations)
            new_locs[holding_R] = (LOC_ONFLOOR, -1)  # dropped at current location
            return (new_locs, holding_L, HOLD_NONE)

        # --- OPEN/CLOSE: needs free hand; block no-op (already in target state) ---
        if action in ('OPEN', 'CLOSE'):
            if holding_L != HOLD_NONE and holding_R != HOLD_NONE:
                return None
            bit = self._prop_bit_by_cid_pred.get((target_cid, 'open'))
            if bit is not None:
                is_open = bool(prop_bits & (1 << bit))
                if action == 'OPEN' and is_open:
                    return None  # already open → no-op
                if action == 'CLOSE' and not is_open:
                    return None  # already closed → no-op
            # OPEN requires the target not toggled on (simulator
            # constraint: `open_close_precond` rejects opening a running
            # appliance).
            if action == 'OPEN':
                t_bit = self._prop_bit_by_cid_pred.get((target_cid, 'toggled_on'))
                if t_bit is not None and (prop_bits & (1 << t_bit)):
                    return None  # cannot open while toggled on
            return (locations, holding_L, holding_R)  # no loc change → share ref (never mutated in place)

        # --- Property actions: no location change ---
        if action in ('TOGGLE_ON', 'TOGGLE_OFF', 'CLEAN', 'SLICE', 'SOAK',
                       'DRY', 'COOK', 'FREEZE', 'UNFREEZE'):
            # Target must have no closed INSIDE-ancestor (simulator
            # constraint, `interactivity_precond`).
            if not self._grasp_container_open(locations, target_cid, prop_bits,
                                              (holding_L, holding_R)):
                return None
            # TOGGLE_ON requires the target closed (simulator constraint,
            # `toggle_on_off_precond`).
            if action == 'TOGGLE_ON':
                o_bit = self._prop_bit_by_cid_pred.get((target_cid, 'open'))
                if o_bit is not None and (prop_bits & (1 << o_bit)):
                    return None  # must close first
            # SLICE needs a slicer in either hand (simulator constraint).
            # No free hand is required: the slicee is typically held in the
            # other hand.
            if action == 'SLICE':
                def _is_knife(cid):
                    if cid == HOLD_NONE:
                        return False
                    # Taxonomy `slicer` ability (knife/cleaver family).
                    return self.membership.slicer(
                        self.cid_to_inst.get(cid, ''))
                if not (_is_knife(holding_L) or _is_knife(holding_R)):
                    return None
            elif holding_L != HOLD_NONE and holding_R != HOLD_NONE:
                # Other property actions (TOGGLE, CLEAN, SOAK, ...) still
                # need a free hand to interact with the target.
                return None
            # CLEAN (simulator constraint): the simulator dispatches on the
            # target's current state.
            #   - dusty target: any cleaning tool in hand.
            #   - stained target: a soaked cleaning tool in hand.
            if action == 'CLEAN':
                def _is_cleaner(cid):
                    if cid == HOLD_NONE or cid not in self._cleaner_cids:
                        return False
                    return True
                def _is_soaked_cleaner(cid):
                    if not _is_cleaner(cid):
                        return False
                    bit = self._prop_bit_by_cid_pred.get((cid, 'soaked'))
                    return bit is not None and bool(prop_bits & (1 << bit))
                stained_bit = self._prop_bit_by_cid_pred.get((target_cid, 'stained'))
                target_stained = (stained_bit is not None
                                  and bool(prop_bits & (1 << stained_bit)))
                if target_stained:
                    if not (_is_soaked_cleaner(holding_L)
                            or _is_soaked_cleaner(holding_R)):
                        return None  # stained → need soaked cleaner
                else:
                    if not (_is_cleaner(holding_L) or _is_cleaner(holding_R)):
                        return None  # dusty (or untracked) → need any cleaner
            # SOAK requires the target INSIDE a water source; tap-equipped
            # sources (sink/bathtub) must be toggled on, passive vessels
            # (kettle/teapot/pool, prompt vessels-of-water) provide water
            # without a switch.
            if action == 'SOAK':
                loc = locations.get(target_cid)
                if loc is None:
                    return None
                loc_type, surface = loc
                if loc_type != LOC_INSIDE or surface < 0:
                    return None
                if surface not in self._water_source_cids:
                    return None
                if surface not in self._passive_water_cids:
                    bit = self._prop_bit_by_cid_pred.get((surface, 'toggled_on'))
                    if bit is None or not (prop_bits & (1 << bit)):
                        return None
            # Capability/affordance check is handled by γ DFA (action_defs.
            # compute_behavior_affordances blocks actions on objects with no
            # matching property mention in initial/target). β here only does
            # simulator preconditions that depend on dynamic state.
            sets = self._ACTION_SETS_PROPS.get(action, ())
            clears = self._ACTION_CLEARS_PROPS.get(action, ())
            # Block no-op property actions: if action's effect is already
            # reflected in prop_bits (over the subset of allocated bits),
            # executing it would be a no-op. Skip properties whose bit is
            # not allocated (we don't track them, so can't say).
            if sets:
                bits = [self._prop_bit_by_cid_pred.get((target_cid, p)) for p in sets]
                allocated = [b for b in bits if b is not None]
                if allocated and all(prop_bits & (1 << b) for b in allocated):
                    return None  # all tracked SET props already set → no-op
            if clears:
                bits = [self._prop_bit_by_cid_pred.get((target_cid, p)) for p in clears]
                allocated = [b for b in bits if b is not None]
                if allocated and all(not (prop_bits & (1 << b)) for b in allocated):
                    return None  # all tracked CLEAR props already clear → no-op
            return (locations, holding_L, holding_R)  # no loc change → share ref (never mutated in place)

        # --- TRANSFER_CONTENTS: must hold, target not held ---
        if action in ('LEFT_TRANSFER_CONTENTS_ONTOP', 'LEFT_TRANSFER_CONTENTS_INSIDE'):
            if holding_L == HOLD_NONE:
                return None
            if target_cid == holding_L or target_cid == holding_R:
                return None
            return (locations, holding_L, holding_R)  # no loc change → share ref (never mutated in place)

        if action in ('RIGHT_TRANSFER_CONTENTS_ONTOP', 'RIGHT_TRANSFER_CONTENTS_INSIDE'):
            if holding_R == HOLD_NONE:
                return None
            if target_cid == holding_L or target_cid == holding_R:
                return None
            return (locations, holding_L, holding_R)  # no loc change → share ref (never mutated in place)

        # Unknown action — block
        return None

    def _is_accept(self, state: BehBetaState) -> bool:
        # Accept iff all goals are satisfied.
        all_goals = (1 << self.n_goals) - 1
        return (state.goal_bits & all_goals) == all_goals

    def _build(self):
        """Depth-bounded forward BFS + dead-end pruning.

        1. Forward BFS from initial state, bounded by max_budget depth
        2. Prune states with no path to any accept state (dead-end removal)

        This avoids the state explosion problem: only states on a path
        from initial to accept (within budget) are kept.
        """
        import time as _t
        t0 = _t.time()

        init_locs = dict(self.init_locations)
        initial = self._get_initial_state()
        self.states.append(initial)
        self.state_to_id[initial] = 0
        self._state_locations: Dict[int, Dict[int, Tuple[int, int]]] = {0: init_locs}

        steps = [BehStep(action=act, args=cids) for act, cids in self.gamma_valid_actions]

        # Phase 1: depth-bounded forward BFS (depth tracked per state).
        depth: Dict[int, int] = {0: 0}
        queue = deque([0])

        # Depth bound: max_budget.
        budget_bound = self._build_max_budget

        while queue:
            src_idx = queue.popleft()
            src_depth = depth[src_idx]

            if src_depth >= budget_bound:
                continue  # don't explore beyond budget depth

            src_state = self.states[src_idx]
            src_locs = self._state_locations[src_idx]
            trans_list = []

            for step in steps:
                result = self._apply_action(
                    src_locs, src_state.holding_L, src_state.holding_R,
                    step.action, step.args, prop_bits=src_state.prop_bits,
                )
                if result is None:
                    continue

                new_locs, new_hold_L, new_hold_R = result

                new_prop = self._apply_prop_change(
                    src_state.prop_bits, step.action,
                    step.args[0] if step.args else -1,
                )

                goal_bits = 0
                for g in self.goals:
                    if self._check_goal(new_locs, g, new_prop):
                        goal_bits |= (1 << g.goal_idx)

                loc_hash = self._encode_locations(new_locs)
                dst_state = BehBetaState(
                    goal_bits=goal_bits, holding_L=new_hold_L,
                    holding_R=new_hold_R, loc_hash=loc_hash,
                    prop_bits=new_prop,
                )

                if dst_state not in self.state_to_id:
                    if len(self.states) >= self.max_states:
                        continue
                    dst_idx = len(self.states)
                    self.states.append(dst_state)
                    self.state_to_id[dst_state] = dst_idx
                    self._state_locations[dst_idx] = new_locs
                    depth[dst_idx] = src_depth + 1
                    queue.append(dst_idx)
                else:
                    dst_idx = self.state_to_id[dst_state]
                    # Update depth if shorter path found
                    if src_depth + 1 < depth.get(dst_idx, budget_bound + 1):
                        depth[dst_idx] = src_depth + 1

                trans_list.append((step, dst_idx))

            self.transitions[src_idx] = trans_list

        n_before = len(self.states)
        n_accept_before = sum(1 for s in self.states if self._is_accept(s))

        # Make accept states absorbing: drop their outgoing transitions so β-DP
        # cannot exit accept and cannot self-loop through identity actions.
        # Paired with the no-carry DP in backward_dp.py, this ensures the DP
        # value strictly discriminates by depth-to-accept (C[B, accept]=0,
        # C[B, non_accept_d] decreases with d), keeping β's gradient sharp at
        # any budget. Once γ is at accept, the processor's BLOCK ABORT path
        # (no β-valid action) drives a graceful list close.
        for i, s in enumerate(self.states):
            if self._is_accept(s):
                self.transitions[i] = []

        if self.debug:
            print(f"[BehBetaSM] Forward BFS: {n_before} states, "
                  f"{n_accept_before} accept, depth≤{budget_bound} ({_t.time()-t0:.2f}s)")

        # Phase 2: Dead-end pruning via backward reachability from accept
        t1 = _t.time()
        accept_ids = {i for i, s in enumerate(self.states) if self._is_accept(s)}

        if not accept_ids:
            # No accept states found — keep everything (let DP handle -inf)
            self.accept_mask = [self._is_accept(s) for s in self.states]
            if self.debug:
                print(f"[BehBetaSM] WARNING: 0 accept states, skipping pruning")
            return

        # Build reverse edges
        rev: Dict[int, Set[int]] = {}
        for src, trans_list in self.transitions.items():
            for _, dst in trans_list:
                if dst not in rev:
                    rev[dst] = set()
                rev[dst].add(src)

        # Backward BFS from accept
        useful: Set[int] = set(accept_ids)
        bqueue = deque(accept_ids)
        while bqueue:
            nid = bqueue.popleft()
            for src in rev.get(nid, ()):
                if src not in useful:
                    useful.add(src)
                    bqueue.append(src)

        # Also keep initial state (even if not on path to accept)
        useful.add(0)

        if self.debug:
            print(f"[BehBetaSM] Backward reach: {len(useful)}/{n_before} useful ({_t.time()-t1:.2f}s)")

        # Phase 3: Rebuild with only useful states
        if len(useful) < n_before:
            t2 = _t.time()
            old_to_new: Dict[int, int] = {}
            new_states: List[BehBetaState] = []
            new_locs: Dict[int, Dict] = {}
            new_transitions: Dict[int, List] = {}

            for old_idx in sorted(useful):
                new_idx = len(new_states)
                old_to_new[old_idx] = new_idx
                new_states.append(self.states[old_idx])
                new_locs[new_idx] = self._state_locations[old_idx]

            for old_idx in sorted(useful):
                new_idx = old_to_new[old_idx]
                new_trans = []
                for step, old_dst in self.transitions.get(old_idx, []):
                    if old_dst in old_to_new:
                        new_trans.append((step, old_to_new[old_dst]))
                new_transitions[new_idx] = new_trans

            self.states = new_states
            self.state_to_id = {s: i for i, s in enumerate(self.states)}
            self._state_locations = new_locs
            self.transitions = new_transitions

            if self.debug:
                print(f"[BehBetaSM] Pruned: {n_before} → {len(self.states)} states ({_t.time()-t2:.2f}s)")

        self.accept_mask = [self._is_accept(s) for s in self.states]

        if self.debug:
            print(f"[BehBetaSM] Final: {len(self.states)} states, {sum(self.accept_mask)} accept, "
                  f"Transitions: {sum(len(t) for t in self.transitions.values())} "
                  f"({_t.time()-t0:.2f}s total)")

    # =========================================================================
    # Interface for BetaBackwardDP
    # =========================================================================

    def get_initial_state_id(self) -> int:
        return 0

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_accept(self) -> int:
        return sum(self.accept_mask)

    def is_accept(self, state_id: int) -> bool:
        if 0 <= state_id < len(self.accept_mask):
            return self.accept_mask[state_id]
        return False

    def apply(self, state_id: int, step) -> Optional[int]:
        if state_id < 0 or state_id >= len(self.states):
            return None
        src_state = self.states[state_id]
        src_locs = self._state_locations.get(state_id, {})

        result = self._apply_action(
            src_locs, src_state.holding_L, src_state.holding_R,
            step.action, step.args, prop_bits=src_state.prop_bits,
        )
        if result is None:
            return None

        new_locs, new_hold_L, new_hold_R = result
        new_prop = self._apply_prop_change(
            src_state.prop_bits, step.action,
            step.args[0] if hasattr(step, 'args') and step.args else -1,
        )
        goal_bits = 0
        for g in self.goals:
            if self._check_goal(new_locs, g, new_prop):
                goal_bits |= (1 << g.goal_idx)

        loc_hash = self._encode_locations(new_locs)
        dst_state = BehBetaState(
            goal_bits=goal_bits, holding_L=new_hold_L,
            holding_R=new_hold_R, loc_hash=loc_hash,
            prop_bits=new_prop,
        )
        return self.state_to_id.get(dst_state)


def parse_beh_states(prompt: str) -> Tuple[List[List], List[List]]:
    """Parse initial and target states from BEH prompt."""
    task = prompt[prompt.find('Your task:'):]

    init_states = []
    target_states = []

    init_m = re.search(r'initial environment state:\n(.*?)(?:\n\ntarget|\ntarget)', task, re.DOTALL)
    if init_m:
        for line in init_m.group(1).strip().split('\n'):
            line = line.strip()
            if line.startswith('['):
                try:
                    init_states.append(ast.literal_eval(line))
                except:
                    pass

    target_m = re.search(r'target environment state:\n(.*?)(?:\n\ninteractable|\ninteractable)', task, re.DOTALL)
    if target_m:
        for line in target_m.group(1).strip().split('\n'):
            line = line.strip()
            if line.startswith('['):
                try:
                    target_states.append(ast.literal_eval(line))
                except:
                    pass

    return init_states, target_states

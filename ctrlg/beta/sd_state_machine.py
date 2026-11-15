"""
VirtualHome Subgoal Decomposition β state machine.

β is the semantic automaton over grounded subgoal predicates. It tracks a
simplified world state to enforce:
- Goal coverage (all kept task goals must be satisfied at accept)
- Temporal ordering via preconditions (e.g., ON requires PLUGGED_IN)

State dimensions (SDBetaState):
- goal_bits: bitmask of emission-tracked goals (action goals, other relations)
- plugged_bits / on_bits / open_bits: per HAS_PLUG / HAS_SWITCH / CAN_OPEN object
- posture: STANDING / SITTING / LYING
- holding_L / holding_R: object held by each hand (explicit HOLDS_* goals)
- dirty_bits: per object declared CLEAN/DIRTY in the initial state
- relation_bits: per placement goal, whether the latest placement matches it

Main preconditions (see _check_precondition):
- ON(obj) requires obj plugged in and not open
- OPEN(obj) requires obj not switched on
- CLOSED / PLUGGED_IN / PLUGGED_OUT require the opposite state
- targets must not be inside a closed container; INSIDE needs an open container
- proximity/grab predicates require STANDING and a non-room, non-body-part target
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


POSTURE_STANDING = 0
POSTURE_SITTING = 1
POSTURE_LYING = 2

HOLD_NONE = -1  # sentinel for an empty hand

# State-property predicates whose goal satisfaction is derived from the
# tracked state bits (open/on/plugged/posture) at accept-check time, not
# accumulated in goal_bits on emission. A goal such as CLOSED(X) that holds
# initially therefore becomes unsatisfied again if a later OPEN(X) flips
# open_bits[X]. Action goals (args=None) and the remaining relation goals
# stay emission-tracked via goal_bits.
STATE_PROP_PREDS = frozenset({
    "OPEN", "CLOSED",
    "ON", "OFF",
    "PLUGGED_IN", "PLUGGED_OUT",
    "SITTING", "LYING", "STANDING",
})

# Relation goals with "latest-wins" semantics in the evaluator's realizer
# (evaluator precondition):
# - ONTOP(plate, dishwasher) then ONTOP(plate, dish_soap) → plate ends up on
#   dish_soap (each placement re-grabs and moves the object), so the goal
#   ONTOP(plate, dishwasher) is no longer satisfied.
# - HOLDS_RH(char, glass) then HOLDS_RH(char, fork) → char ends up holding fork.
# These predicates are therefore derived from the current placement / holding
# state (relation_bits / holding_L/R), like STATE_PROP_PREDS, rather than
# accumulated in goal_bits.
LOCATION_PREDS = frozenset({"ONTOP", "INSIDE"})
HOLD_PREDS = frozenset({"HOLDS_RH", "HOLDS_LH"})

# FACING is not durable either. The evaluator's realizer
# (state_action_translator.map_primitive_to_action_set) starts every subgoal
# that takes an object with FIND or WALK, and both delete the character's FACING
# edges in the simulator (TURNTO replaces them). So a FACING(character, X) goal
# holds at the end only if FACING(character, X), LOOKAT(X) or WATCH(X) (realized
# as FIND, TURNTO X, ...) comes after the last other such subgoal. Subgoals the
# realizer maps to no movement keep FACING:
FACING_PRESERVING = frozenset({"SITTING", "LYING", "STANDING", "DIRTY"})


class SDBetaState(NamedTuple):
    """Subgoal decomposition β state."""
    goal_bits: int       # bitmask of covered goals (state + action goals)
    plugged_bits: int    # per-object: 1=plugged_in, 0=plugged_out
    on_bits: int         # per-object: 1=on, 0=off
    open_bits: int       # per-object: 1=open, 0=closed
    posture: int         # 0=STANDING, 1=SITTING, 2=LYING
    holding_L: int = HOLD_NONE  # cid held in left hand (explicit HOLDS_LH goals), or HOLD_NONE
    holding_R: int = HOLD_NONE  # cid held in right hand (explicit HOLDS_RH goals), or HOLD_NONE
    dirty_bits: int = 0  # per dirty-tracked object: 1=dirty(stained), 0=clean
    # Per placement-goal bit (ONTOP/INSIDE, non-character subject): 1 iff the
    # latest ONTOP/INSIDE of that subject targets the goal's target. See
    # LOCATION_PREDS / _setup_placement_tracking.
    relation_bits: int = 0


@dataclass(frozen=True)
class SDGoal:
    """A parsed task goal."""
    predicate: str       # e.g. "ON", "CLOSED", "ONTOP"
    args: Tuple[int, ...]  # canonical IDs
    goal_idx: int        # position in goal list (for bitmask)


@dataclass
class SDStep:
    """A grounded predicate step with canonical IDs. Same interface as VHStep."""
    action: str          # predicate name (e.g. "ON", "NEXT_TO")
    args: Tuple[int, ...]  # canonical IDs


class SDBetaStateMachine:
    """
    β state machine for VirtualHome Subgoal Decomposition.

    Interface compatible with BetaBackwardDP:
    - transitions: Dict[int, List[Tuple[SDStep, int]]]
    - accept_mask: List[bool]
    - num_states, num_accept
    - apply(state_id, step) -> Optional[int]
    - is_accept(state_id) -> bool
    - get_initial_state_id() -> int
    """

    def __init__(
        self,
        initial_states: List[str],
        goal_states: List[str],
        action_goals: List[str],
        objects: Dict[str, List[str]],
        properties: Dict[str, Set[str]],
        categories: Dict[str, str],
        gamma_valid_predicates: List[Tuple[str, Tuple[int, ...]]],
        orig_id_to_canonical: Dict[str, int],
        canonical_to_info: Dict[int, Tuple[str, str]],
        max_states: int = 50000,
        debug: bool = False,
    ):
        self.debug = debug
        self.max_states = max_states
        self.objects = objects
        self.properties = properties
        self.categories = categories
        self.gamma_valid_predicates = gamma_valid_predicates
        self.orig_id_to_canonical = orig_id_to_canonical
        self.canonical_to_info = canonical_to_info

        # Parse which objects have which trackable properties
        self._setup_tracking()

        # Parse goals
        self.goals: List[SDGoal] = []
        self._parse_goals(goal_states, action_goals)
        # n_goals = number of unique goal indices (not entries, since OR alternatives share an index)
        self.n_goals = max((g.goal_idx for g in self.goals), default=-1) + 1

        # Parse initial world state
        self._parse_initial_states(initial_states)

        # Build the state machine via BFS, dropping goals by priority if BFS
        # exceeds max_states or finds no accept state.
        self.dropped_goal_indices: List[int] = []
        self._build_with_dropping()

    def _pick_drop_goal(self, active_groups: List[Tuple['SDGoal', ...]]) -> int:
        """Pick which goal to drop FIRST. Higher rank → dropped earlier.
        Priority (largest rank tuple wins):
          1. action-goal style (args=None) — LM satisfies these solo more easily
          2. tiebreak: highest goal_idx (deterministic, last-added first)
        """
        def rank(group: Tuple['SDGoal', ...]) -> Tuple[int, int]:
            is_action = 1 if group[0].args is None else 0
            return (is_action, group[0].goal_idx)
        return max(range(len(active_groups)),
                   key=lambda i: rank(active_groups[i]))

    def _build_with_dropping(self) -> None:
        """BFS-build with priority goal dropping on overflow / 0-accept.

        Goals are dropped and renumbered as groups: the alternatives of an
        "A or B" action goal share one goal_idx (one bit), and any one of them
        satisfies the goal. Uses the shared `build_with_dropping` helper.
        """
        from .goal_drop import build_with_dropping

        groups: Dict[int, List['SDGoal']] = {}
        for g in self.goals:
            groups.setdefault(g.goal_idx, []).append(g)

        def _build(active_groups: List[Tuple['SDGoal', ...]]) -> Tuple[int, bool]:
            self.goals = [
                SDGoal(predicate=g.predicate, args=g.args, goal_idx=new_bit)
                for new_bit, group in enumerate(active_groups) for g in group
            ]
            self.n_goals = len(active_groups)
            self.states = []
            self.state_to_id = {}
            self.transitions = {}
            self.accept_mask = []
            self._build()
            n_states = len(self.states)
            cap_hit = n_states >= self.max_states
            return n_states, (not cap_hit)

        def _has_accept(_n_states: int, _active) -> bool:
            return sum(self.accept_mask) > 0

        _, kept, dropped = build_with_dropping(
            [tuple(groups[i]) for i in sorted(groups)],
            build_fn=_build,
            has_accept_fn=_has_accept,
            pick_drop_fn=self._pick_drop_goal,
            min_active=1,
            debug=self.debug,
            label='SDBetaSM',
        )
        self.dropped_goal_indices = [group[0].goal_idx for group in dropped]

    def _setup_tracking(self):
        """Identify which objects need state tracking."""
        self.pluggable_objs: List[int] = []  # canonical IDs of HAS_PLUG objects
        self.switchable_objs: List[int] = []  # canonical IDs of HAS_SWITCH objects
        self.openable_objs: List[int] = []   # canonical IDs of CAN_OPEN objects

        self._plug_idx: Dict[int, int] = {}  # cid -> bit index in plugged_bits
        self._switch_idx: Dict[int, int] = {}
        self._open_idx: Dict[int, int] = {}

        for name, props in self.properties.items():
            for raw_id in self.objects.get(name, []):
                cid = self.orig_id_to_canonical.get(str(raw_id))
                if cid is None:
                    continue
                if "HAS_PLUG" in props:
                    self._plug_idx[cid] = len(self.pluggable_objs)
                    self.pluggable_objs.append(cid)
                if "HAS_SWITCH" in props:
                    self._switch_idx[cid] = len(self.switchable_objs)
                    self.switchable_objs.append(cid)
                if "CAN_OPEN" in props:
                    self._open_idx[cid] = len(self.openable_objs)
                    self.openable_objs.append(cid)

        if self.debug:
            print(f"[SDBetaSM] Tracking: {len(self.pluggable_objs)} pluggable, "
                  f"{len(self.switchable_objs)} switchable, {len(self.openable_objs)} openable")

    def _parse_pred_string(self, s: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
        """Parse 'PRED(name.id, name.id)' → (pred, (cid1, cid2))."""
        m = re.match(r'(\w+)\((.*)\)', s.strip())
        if not m:
            return None
        pred = m.group(1)
        args_str = m.group(2).strip()
        if not args_str:
            return (pred, ())
        cids = []
        for arg in args_str.split(','):
            arg = arg.strip()
            parts = arg.split('.')
            if len(parts) != 2:
                return None
            raw_id = parts[1]
            cid = self.orig_id_to_canonical.get(str(raw_id))
            if cid is None:
                return None
            cids.append(cid)
        return (pred, tuple(cids))

    def _parse_goals(self, goal_states: List[str], action_goals: List[str]):
        """Parse goal strings into SDGoal objects.

        State goals: exact match (predicate + args).
        Action goals: bare predicate names (e.g., "RINSE", "TYPE") or "A or B" format.
          Each line is one goal. "or" alternatives share one goal_idx.
          args=None means any instance of the predicate covers the goal.
        """
        idx = 0
        n_state = 0
        n_action = 0

        # State goals: PRED(args) format
        for gs in goal_states:
            parsed = self._parse_pred_string(gs)
            if parsed:
                pred, cids = parsed
                self.goals.append(SDGoal(predicate=pred, args=cids, goal_idx=idx))
                idx += 1
                n_state += 1

        # Action goals: bare names like "RINSE" or "WASH or RINSE or SCRUB"
        for ag in action_goals:
            if not ag or ag.strip() == 'None':
                continue
            alternatives = [part.strip().upper() for part in ag.split(' or ') if part.strip()]
            if not alternatives:
                continue
            goal_idx = idx
            for alt in alternatives:
                # Action goals are bare predicate names (no parentheses)
                # Any instance of that predicate (with any args) satisfies the goal
                self.goals.append(SDGoal(predicate=alt, args=None, goal_idx=goal_idx))
            idx += 1
            n_action += 1

        if self.debug:
            print(f"[SDBetaSM] {len(self.goals)} goals parsed "
                  f"({n_state} state, {n_action} action)")

    def _parse_initial_states(self, initial_states: List[str]):
        """Parse initial states to set starting world state bits and posture."""
        self._raw_initial_states = list(initial_states)  # read by _init_relation_bits
        self.init_plugged = 0
        self.init_on = 0
        self.init_open = 0
        self.init_posture = POSTURE_STANDING
        # Dirty/clean tracking: only objects declared CLEAN or DIRTY in the
        # initial state are tracked; an undeclared object is treated as clean.
        # CLEAN(x) is valid only while x is dirty and makes it clean, so a
        # repeated no-op CLEAN(x) is rejected.
        self._dirty_idx: Dict[int, int] = {}
        self.init_dirty = 0
        for s in initial_states:
            parsed = self._parse_pred_string(s)
            if not parsed:
                continue
            pred, cids = parsed
            if pred in ("CLEAN", "DIRTY") and len(cids) == 1:
                c = cids[0]
                if c not in self._dirty_idx:
                    self._dirty_idx[c] = len(self._dirty_idx)
                if pred == "DIRTY":
                    self.init_dirty |= (1 << self._dirty_idx[c])
        # Map obj_cid → container_cid for objects initially INSIDE an
        # openable container. `_target_accessible` walks this parent chain,
        # as the evaluator walks the relation tree (evaluator precondition).
        self.init_inside_container: Dict[int, int] = {}

        for s in initial_states:
            parsed = self._parse_pred_string(s)
            if not parsed:
                continue
            pred, cids = parsed

            if pred == "PLUGGED_IN" and len(cids) == 1:
                bit = self._plug_idx.get(cids[0])
                if bit is not None:
                    self.init_plugged |= (1 << bit)
            elif pred == "ON" and len(cids) == 1:
                bit = self._switch_idx.get(cids[0])
                if bit is not None:
                    self.init_on |= (1 << bit)
            elif pred == "OPEN" and len(cids) == 1:
                bit = self._open_idx.get(cids[0])
                if bit is not None:
                    self.init_open |= (1 << bit)
            elif pred == "INSIDE" and len(cids) == 2:
                # INSIDE(obj, container) — record if container is openable.
                obj_cid, container_cid = cids
                if container_cid in self._open_idx:
                    self.init_inside_container[obj_cid] = container_cid
            elif pred == "SITTING":
                self.init_posture = POSTURE_SITTING
            elif pred == "LYING":
                self.init_posture = POSTURE_LYING

        if self.debug:
            posture_name = {0: 'STANDING', 1: 'SITTING', 2: 'LYING'}[self.init_posture]
            print(f"[SDBetaSM] Initial: plugged=0b{self.init_plugged:b}, "
                  f"on=0b{self.init_on:b}, open=0b{self.init_open:b}, "
                  f"posture={posture_name}")

    def _goal_satisfied_at_init(self, g: 'SDGoal') -> bool:
        """True iff the goal predicate is already satisfied by the parsed
        initial state. Used to pre-flip goal_bits so the LLM doesn't need
        to redundantly re-emit goals that init already covers (e.g.
        PLUGGED_IN(freezer) when init has PLUGGED_IN(freezer))."""
        if g.args is None:
            return False  # action-goal — cannot be satisfied without an action
        pred, cids = g.predicate, g.args
        if pred == "PLUGGED_IN" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            return bit is not None and bool(self.init_plugged & (1 << bit))
        if pred == "PLUGGED_OUT" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            return bit is not None and not (self.init_plugged & (1 << bit))
        if pred == "ON" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            return bit is not None and bool(self.init_on & (1 << bit))
        if pred == "OFF" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            return bit is not None and not (self.init_on & (1 << bit))
        if pred == "OPEN" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            return bit is not None and bool(self.init_open & (1 << bit))
        if pred == "CLOSED" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            return bit is not None and not (self.init_open & (1 << bit))
        if pred == "SITTING":
            return self.init_posture == POSTURE_SITTING
        if pred == "LYING":
            return self.init_posture == POSTURE_LYING
        if pred == "STANDING":
            return self.init_posture == POSTURE_STANDING
        if pred == "INSIDE" and len(cids) == 2:
            obj_cid, container_cid = cids
            return self.init_inside_container.get(obj_cid) == container_cid
        # ONTOP at init isn't separately tracked here — but placement goals
        # are derived from relation_bits at accept time (init_relation_bits
        # is built from the same parsed init), so returning False here is
        # safe: _compute_init_goal_bits already skips placement goals.
        return False

    def _compute_init_goal_bits(self) -> int:
        """Bitmask of emission-tracked goals already satisfied by init,
        pre-set so the LLM doesn't need to re-emit them. Derived goals
        (state-prop, placement, holding) are skipped: their satisfaction is
        computed from the tracked state at accept time, so it follows later
        changes to that state.
        """
        bits = 0
        for g in self.goals:
            if g.args is not None and g.predicate in STATE_PROP_PREDS:
                continue  # state-prop — derived from tracked bits
            # Placement goals (ONTOP/INSIDE non-char subject) and holding goals
            # (HOLDS_RH/LH char-subject) are also derived (from relation_bits /
            # holding_L/R) — see _goal_currently_satisfied. Skip pre-flip.
            if (g.args is not None and g.goal_idx in
                    getattr(self, '_goal_idx_to_rel_bit', {})):
                continue
            if (g.args is not None and g.predicate in HOLD_PREDS
                    and len(g.args) == 2 and self._is_character(g.args[0])):
                continue
            if self._goal_satisfied_at_init(g):
                bits |= (1 << g.goal_idx)
        return bits

    def _get_initial_state(self) -> SDBetaState:
        return SDBetaState(
            goal_bits=self._compute_init_goal_bits(),
            plugged_bits=self.init_plugged,
            on_bits=self.init_on,
            open_bits=self.init_open,
            posture=self.init_posture,
            holding_L=HOLD_NONE,
            holding_R=HOLD_NONE,
            dirty_bits=self.init_dirty,
            relation_bits=self._init_relation_bits(),
        )

    def _is_body_part(self, cid: int) -> bool:
        """True iff `cid` is a body part (carries the BODY_PART property).
        Body parts can't be FIND/navigation/grab targets (see _check_precondition)."""
        info = self.canonical_to_info.get(cid)
        if not info:
            return False
        return "BODY_PART" in self.properties.get(info[0], set())

    def _is_character(self, cid: int) -> bool:
        info = self.canonical_to_info.get(cid)
        return bool(info and self.categories.get(info[0]) == "Characters")

    def _has_prop(self, cid: int, prop: str) -> bool:
        info = self.canonical_to_info.get(cid)
        return bool(info and prop in self.properties.get(info[0], set()))

    def _ontop_posture(self, cids: Tuple[int, ...]) -> Optional[int]:
        """If ONTOP(character, X) means the character sits/lies on a surface,
        return the resulting posture (SITTING for SITTABLE, LYING for LIEABLE),
        else None. The evaluator realizes ONTOP(character, X) as [SIT]<X>,
        which sets sitting/lying."""
        if len(cids) == 2 and self._is_character(cids[0]):
            tgt = cids[1]
            if self._has_prop(tgt, "SITTABLE"):
                return POSTURE_SITTING
            if self._has_prop(tgt, "LIEABLE"):
                return POSTURE_LYING
        return None

    def _target_accessible(self, state: SDBetaState, target_cid: int) -> bool:
        """Walk up `target_cid`'s containment chain; reject if any openable
        ancestor is currently closed.

        Evaluator precondition: any action on an object inside a closed
        container fails. E.g. `ONTOP(plate, dishwasher)` is realized with a
        GRAB of the plate, which fails while the plate is in a closed cabinet.

        Only the initial INSIDE chain is walked (β doesn't track dynamic
        location); once an OPEN(container) subgoal sets the open bit, the
        walk passes through.
        """
        seen = set()
        cur = target_cid
        while cur not in seen:
            seen.add(cur)
            container = self.init_inside_container.get(cur)
            if container is None:
                return True  # no more ancestors — accessible
            bit = self._open_idx.get(container)
            if bit is not None and not (state.open_bits & (1 << bit)):
                return False  # ancestor closed
            cur = container
        return True

    def _check_precondition(self, state: SDBetaState, pred: str, cids: Tuple[int, ...]) -> bool:
        """
        Check if predicate is valid from current state.

        A subgoal predicate is valid if:
        - The desired state is ALREADY TRUE → no-op, always valid (BFS does nothing)
        - The desired state needs changing → check the underlying action's preconditions

        Example: ON(wm) when wm is unplugged AND off → blocked (SWITCHON needs plugged_in).
                 ON(wm) when wm is already on → allowed (no-op).
                 ON(wm) when wm is plugged_in but off → allowed (SWITCHON valid).
        """
        # Container-open precondition: for any subgoal that the evaluator
        # realizes via an action on the target object, the target must be
        # accessible (not inside a transitively-closed container). The
        # exception is OPEN/CLOSED themselves, which act ON a container —
        # those don't need their own contents to be accessible.
        if pred not in ("OPEN", "CLOSED") and cids:
            # Check accessibility of the first arg (the subject of the predicate).
            # For relations like INSIDE(obj, container) or ONTOP(obj, surface),
            # the evaluator needs to GRAB obj, so obj must be reachable.
            if not self._target_accessible(state, cids[0]):
                return False
        # INSIDE(obj, container) is realized as PUTIN<obj><container>, which
        # requires an openable container to be open (evaluator precondition).
        # An INSIDE goal stays reachable by emitting OPEN first.
        if pred == "INSIDE" and len(cids) == 2:
            container = cids[1]
            obit = self._open_idx.get(container)
            if obit is not None and not (state.open_bits & (1 << obit)):
                return False

        # PLUGGED_IN requires the target to be plugged out: PLUGIN on a
        # plugged-in object fails (evaluator precondition).
        if pred == "PLUGGED_IN" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            if bit is not None and (state.plugged_bits & (1 << bit)):
                return False  # already plugged → PLUGIN would fail

        # Symmetric: PLUGGED_OUT requires currently plugged_in.
        if pred == "PLUGGED_OUT" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            if bit is not None and not (state.plugged_bits & (1 << bit)):
                return False  # not plugged → PLUGOUT would fail

        if pred == "ON" and len(cids) == 1:
            bit_sw = self._switch_idx.get(cids[0])
            if bit_sw is not None and (state.on_bits & (1 << bit_sw)):
                return True  # already on → no-op
            # Need to switch on → requires plugged_in
            bit_pl = self._plug_idx.get(cids[0])
            if bit_pl is not None and not (state.plugged_bits & (1 << bit_pl)):
                return False
            # SWITCHON requires the object to be closed (evaluator precondition).
            bit_op = self._open_idx.get(cids[0])
            if bit_op is not None and (state.open_bits & (1 << bit_op)):
                return False  # open → must CLOSE first

        if pred == "OFF" and len(cids) == 1:
            bit_sw = self._switch_idx.get(cids[0])
            if bit_sw is not None and not (state.on_bits & (1 << bit_sw)):
                return True  # already off → no-op
            # Need to switch off → always valid if currently on

        if pred == "OPEN" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            if bit is not None and (state.open_bits & (1 << bit)):
                return True  # already open → no-op
            # OPEN requires the target to be switched off: opening a switched-on
            # appliance fails (evaluator precondition).
            sw_bit = self._switch_idx.get(cids[0])
            if sw_bit is not None and (state.on_bits & (1 << sw_bit)):
                return False  # on → must SWITCHOFF first

        if pred == "CLOSED" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            # CLOSE fails unless the object is currently open (evaluator
            # precondition), so CLOSED is not a no-op on a closed object.
            if bit is not None and not (state.open_bits & (1 << bit)):
                return False  # not open → CLOSE would fail at runtime

        # CLEAN(x) requires x to be dirty-tracked and currently dirty; cleaning
        # a clean object is a no-op. If no object is declared DIRTY, every CLEAN
        # is rejected.
        if pred == "CLEAN" and len(cids) == 1:
            bit = self._dirty_idx.get(cids[0])
            if bit is None or not (state.dirty_bits & (1 << bit)):
                return False  # already clean (or untracked → clean) → no-op
        # DIRTY(x) only undoes progress; reject it.
        if pred == "DIRTY":
            return False

        # SWITCHOFF requires object to be ON (can't switch off what's already off)
        if pred == "SWITCHOFF" and len(cids) == 1:
            bit_sw = self._switch_idx.get(cids[0])
            if bit_sw is not None and not (state.on_bits & (1 << bit_sw)):
                return False  # object is OFF → can't switch off

        # Hand occupancy for explicit HOLDS_RH/HOLDS_LH(character, X): these are
        # realized as a GRAB of X with the named hand, and a hand holds one
        # object, so a second GRAB into an occupied hand fails (evaluator
        # precondition). Placement (ONTOP/INSIDE) is not gated on holding: the
        # evaluator grabs the object itself when realizing a placement.
        if pred in ("HOLDS_RH", "HOLDS_LH") and len(cids) >= 2:
            obj = cids[1]
            # The grabbed object is cids[1]; the generic accessibility check
            # above only tested cids[0] (the character).
            if not self._target_accessible(state, obj):
                return False
            if pred == "HOLDS_RH":
                # reject if RH is full, or the object is already held in LH
                # (grabbing an object already held fails).
                if state.holding_R != HOLD_NONE or state.holding_L == obj:
                    return False
            else:  # HOLDS_LH
                if state.holding_L != HOLD_NONE or state.holding_R == obj:
                    return False

        # Room affordance: TOUCH/NEXT_TO a room fails (simulator constraint; the
        # agent is inside the room). Rooms have category 'Rooms'.
        if pred in ("TOUCH", "NEXT_TO"):
            for c in cids:
                info = self.canonical_to_info.get(c)
                if info and self.categories.get(info[0]) == "Rooms":
                    return False

        # Body parts (face/hair/teeth/hands_both/..., the BODY_PART property)
        # are never CLOSE to the agent in the simulator, so predicates realized
        # via FIND<bodypart> cannot target them. The rule also covers
        # INSIDE/ONTOP, so a goal such as INSIDE(water, hands_both) is out of
        # reach for β and its task is decoded with the syntax mask alone.
        if pred in ("TOUCH", "NEXT_TO", "FACING", "HOLDS_RH", "HOLDS_LH",
                    "LOOKAT", "WATCH", "POINTAT", "FIND", "INSIDE", "ONTOP"):
            for c in cids:
                if self._is_body_part(c):
                    return False

        # Self-reference: a relation between an object and itself
        # (INSIDE(X,X)/ONTOP(X,X)/NEXT_TO(X,X)/FACING(X,X)) is not realizable
        # (physical constraint).
        if pred in ("INSIDE", "ONTOP", "NEXT_TO", "FACING") and len(cids) == 2 \
                and cids[0] == cids[1]:
            return False

        # Grabbability: HOLDS_RH/LH(char, X) is realized as GRAB<X>, which
        # requires X to be GRABBABLE (evaluator precondition).
        if pred in ("HOLDS_RH", "HOLDS_LH") and len(cids) == 2:
            if not self._has_prop(cids[1], "GRABBABLE"):
                return False

        # Posture ordering: NEXT_TO/FACING/HOLDS/TOUCH are realized via
        # FIND<target>, which fails while the character is sitting or lying
        # (evaluator precondition). STANDUP is not in the SD subgoal vocabulary,
        # so navigation/grab predicates must all come before any SIT/LIE.
        if pred in ("NEXT_TO", "FACING", "HOLDS_RH", "HOLDS_LH", "TOUCH"):
            if state.posture != POSTURE_STANDING:
                return False

        # SLEEP/WAKEUP require the character to be SITTING or LYING
        # (evaluator precondition).
        if pred in ("SLEEP", "WAKEUP") and len(cids) == 0:
            if state.posture == POSTURE_STANDING:
                return False  # can't sleep/wakeup while standing

        # SITTING/LYING when already in that posture: allowed as a no-op.
        if pred == "SITTING" and state.posture == POSTURE_SITTING:
            return True  # already sitting → no-op

        if pred == "LYING" and state.posture == POSTURE_LYING:
            return True  # already lying → no-op

        # STANDUP is not modeled: it is not in the evaluator's SD subgoal
        # vocabulary.

        return True

    def _apply_effect(self, state: SDBetaState, pred: str, cids: Tuple[int, ...]) -> SDBetaState:
        """Apply predicate effect to state, return new state."""
        goal_bits = state.goal_bits
        plugged_bits = state.plugged_bits
        on_bits = state.on_bits
        open_bits = state.open_bits
        posture = state.posture
        holding_L = state.holding_L
        holding_R = state.holding_R
        dirty_bits = state.dirty_bits
        relation_bits = state.relation_bits

        # Placement-goal bit update for ONTOP(X, Y) / INSIDE(X, Y) emissions.
        # For each placement goal with subject X, set its bit iff Y matches the
        # goal target; else clear it ("latest-wins", see LOCATION_PREDS).
        if pred in LOCATION_PREDS and len(cids) == 2:
            for rel_bit, goal_target in self._loc_goals_by_subject.get(
                    (pred, cids[0]), []):
                if cids[1] == goal_target:
                    relation_bits |= (1 << rel_bit)
                else:
                    relation_bits &= ~(1 << rel_bit)

        # CLEAN(x) flips a dirty object to clean (precondition guarantees it was
        # dirty-tracked + dirty).
        if pred == "CLEAN" and len(cids) == 1:
            bit = self._dirty_idx.get(cids[0])
            if bit is not None:
                dirty_bits &= ~(1 << bit)

        # Hand-occupancy effect: an explicit HOLDS_RH/LH fills that hand with the
        # held object (cids[1]). Placement predicates are not modeled here: the
        # evaluator grabs and releases for ONTOP/INSIDE, so they don't durably
        # occupy a hand.
        if pred == "HOLDS_RH" and len(cids) >= 2:
            holding_R = cids[1]
        elif pred == "HOLDS_LH" and len(cids) >= 2:
            holding_L = cids[1]

        # Update world state
        if pred == "PLUGGED_IN" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            if bit is not None:
                plugged_bits |= (1 << bit)
        elif pred == "PLUGGED_OUT" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            if bit is not None:
                plugged_bits &= ~(1 << bit)
        elif pred == "ON" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            if bit is not None:
                on_bits |= (1 << bit)
        elif pred == "OFF" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            if bit is not None:
                on_bits &= ~(1 << bit)
        elif pred == "OPEN" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            if bit is not None:
                open_bits |= (1 << bit)
        elif pred == "CLOSED" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            if bit is not None:
                open_bits &= ~(1 << bit)
        elif pred == "SWITCHOFF" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            if bit is not None:
                on_bits &= ~(1 << bit)

        # Posture effects
        if pred == "SITTING":
            posture = POSTURE_SITTING
        elif pred == "LYING":
            posture = POSTURE_LYING
        elif pred == "ONTOP":
            # ONTOP(character, sittable/lieable) is realized as SIT/LIE.
            seated = self._ontop_posture(cids)
            if seated is not None:
                posture = seated
        # STANDUP is not in the SD subgoal vocabulary.

        # Update goal coverage. Goals that are derived from tracked state are
        # NOT accumulated in goal_bits — their satisfaction is computed from
        # the relevant state bits / fields at accept-check time (see
        # `_goal_currently_satisfied`). Only "true emission" goals (action
        # goals + non-derived relation goals) flip a goal_bit.
        if cids and pred not in FACING_PRESERVING:
            # The subgoal is realized starting with FIND/WALK, which clears FACING.
            goal_bits &= ~self._facing_goal_mask
            if pred in ("LOOKAT", "WATCH"):  # realized as FIND, TURNTO <target>, ...
                goal_bits |= self._facing_goal_bit_by_target.get(cids[-1], 0)
        for g in self.goals:
            if g.args is None:
                # Action goal: match by predicate name only (any args)
                if g.predicate == pred:
                    goal_bits |= (1 << g.goal_idx)
                continue
            if g.predicate in STATE_PROP_PREDS:
                continue  # derived from plugged/on/open/posture/dirty bits
            if g.goal_idx in self._goal_idx_to_rel_bit:
                continue  # placement goal — derived from relation_bits
            if (g.predicate in HOLD_PREDS and len(g.args) == 2
                    and self._is_character(g.args[0])):
                continue  # holding goal — derived from holding_L / holding_R
            # Other relation goals (e.g., NEXT_TO/FACING/BETWEEN, or
            # ONTOP/INSIDE with character subject) — emission-tracked.
            if g.predicate == pred and g.args == cids:
                goal_bits |= (1 << g.goal_idx)

        return SDBetaState(
            goal_bits=goal_bits,
            plugged_bits=plugged_bits,
            on_bits=on_bits,
            open_bits=open_bits,
            posture=posture,
            holding_L=holding_L,
            holding_R=holding_R,
            dirty_bits=dirty_bits,
            relation_bits=relation_bits,
        )

    def _goal_currently_satisfied(self, g: 'SDGoal', state: SDBetaState) -> bool:
        """True iff the goal is currently satisfied by `state`.

        - State-prop goals (OPEN/CLOSED/ON/OFF/PLUGGED_IN/PLUGGED_OUT/
          SITTING/LYING/STANDING) check the underlying state bits.
        - Placement goals check relation_bits; character HOLDS_* goals check
          the hand slots.
        - Action goals (g.args is None) and other relation goals check the
          accumulated goal_bits.
        """
        if g.args is None:
            return bool(state.goal_bits & (1 << g.goal_idx))
        pred, cids = g.predicate, g.args
        if pred == "PLUGGED_IN" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            return bit is not None and bool(state.plugged_bits & (1 << bit))
        if pred == "PLUGGED_OUT" and len(cids) == 1:
            bit = self._plug_idx.get(cids[0])
            return bit is not None and not (state.plugged_bits & (1 << bit))
        if pred == "ON" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            return bit is not None and bool(state.on_bits & (1 << bit))
        if pred == "OFF" and len(cids) == 1:
            bit = self._switch_idx.get(cids[0])
            return bit is not None and not (state.on_bits & (1 << bit))
        if pred == "OPEN" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            return bit is not None and bool(state.open_bits & (1 << bit))
        if pred == "CLOSED" and len(cids) == 1:
            bit = self._open_idx.get(cids[0])
            return bit is not None and not (state.open_bits & (1 << bit))
        if pred == "SITTING":
            return state.posture == POSTURE_SITTING
        if pred == "LYING":
            return state.posture == POSTURE_LYING
        if pred == "STANDING":
            return state.posture == POSTURE_STANDING
        # Placement goal (ONTOP/INSIDE, non-character subject) — derived from
        # relation_bits, which tracks the latest placement target per subject.
        rel_bit = self._goal_idx_to_rel_bit.get(g.goal_idx)
        if rel_bit is not None:
            return bool(state.relation_bits & (1 << rel_bit))
        # Holding goal (HOLDS_RH/HOLDS_LH with character subject) — derived
        # from the corresponding hand-slot, which already tracks the LATEST
        # grabbed object.
        if (pred in HOLD_PREDS and len(cids) == 2 and self._is_character(cids[0])):
            held = state.holding_R if pred == "HOLDS_RH" else state.holding_L
            return held == cids[1]
        # Other relation goals (NEXT_TO/FACING/BETWEEN/etc.) — emission-tracked.
        return bool(state.goal_bits & (1 << g.goal_idx))

    def _is_accept(self, state: SDBetaState) -> bool:
        """Accept when every goal is currently satisfied (state-prop goals
        derived from state bits, others from goal_bits)."""
        return all(self._goal_currently_satisfied(g, state) for g in self.goals)

    def _setup_placement_tracking(self):
        """Identify placement goals (ONTOP/INSIDE with non-character subject)
        and allocate a relation_bits bit per such goal. The bit is set iff the
        current placement matches the goal target — derived from the latest
        ONTOP/INSIDE emission for that subject. Called per build (after goal
        renumbering) since the set of kept goals depends on goal-dropping."""
        self._goal_idx_to_rel_bit: Dict[int, int] = {}
        # Map (pred, subject_cid) → [(rel_bit, goal_target_cid)] for fast update
        # in _apply_effect.
        self._loc_goals_by_subject: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
        bit = 0
        for g in self.goals:
            if (g.args is None or g.predicate not in LOCATION_PREDS
                    or len(g.args) != 2):
                continue
            subj = g.args[0]
            # Skip character-subject ONTOP/INSIDE: ONTOP(char, sittable) is
            # posture-handled via _ontop_posture (sets sitting/lying), and
            # INSIDE(char, room) is char-location which β doesn't track for SD.
            if self._is_character(subj):
                continue
            self._goal_idx_to_rel_bit[g.goal_idx] = bit
            self._loc_goals_by_subject.setdefault(
                (g.predicate, subj), []
            ).append((bit, g.args[1]))
            bit += 1

    def _setup_facing_tracking(self):
        """Goal bits of the FACING(character, X) goals, all together and by X
        (see FACING_PRESERVING). Called per build, after goal renumbering."""
        self._facing_goal_mask = 0
        self._facing_goal_bit_by_target: Dict[int, int] = {}
        for g in self.goals:
            if (g.predicate == "FACING" and g.args is not None and len(g.args) == 2
                    and self._is_character(g.args[0])):
                self._facing_goal_mask |= 1 << g.goal_idx
                self._facing_goal_bit_by_target[g.args[1]] = (
                    self._facing_goal_bit_by_target.get(g.args[1], 0) | 1 << g.goal_idx)

    def _init_relation_bits(self) -> int:
        """Init relation_bits from any ONTOP/INSIDE relations in the prompt's
        initial states that already match a placement goal target."""
        rb = 0
        for s in getattr(self, '_raw_initial_states', []):
            parsed = self._parse_pred_string(s)
            if not parsed:
                continue
            pred, cids = parsed
            if pred not in LOCATION_PREDS or len(cids) != 2:
                continue
            for rel_bit, goal_target in self._loc_goals_by_subject.get(
                    (pred, cids[0]), []):
                if goal_target == cids[1]:
                    rb |= (1 << rel_bit)
        return rb

    def _build(self):
        """BFS from initial state to enumerate all reachable states and transitions."""
        # Set up placement tracking BEFORE _get_initial_state (which reads
        # relation_bits via _init_relation_bits).
        self._setup_placement_tracking()
        self._setup_facing_tracking()
        initial = self._get_initial_state()
        self.states.append(initial)
        self.state_to_id[initial] = 0

        queue = deque([0])
        steps = [SDStep(action=pred, args=cids) for pred, cids in self.gamma_valid_predicates]

        while queue:
            src_idx = queue.popleft()
            src_state = self.states[src_idx]
            trans_list = []

            for step in steps:
                # Check precondition
                if not self._check_precondition(src_state, step.action, step.args):
                    continue

                # Apply effect
                dst_state = self._apply_effect(src_state, step.action, step.args)

                # Get or create destination state
                if dst_state not in self.state_to_id:
                    if len(self.states) >= self.max_states:
                        continue  # skip if state space exhausted
                    dst_idx = len(self.states)
                    self.states.append(dst_state)
                    self.state_to_id[dst_state] = dst_idx
                    queue.append(dst_idx)
                else:
                    dst_idx = self.state_to_id[dst_state]

                trans_list.append((step, dst_idx))

            self.transitions[src_idx] = trans_list

        # Compute accept mask
        self.accept_mask = [self._is_accept(s) for s in self.states]

        if self.debug:
            print(f"[SDBetaSM] States: {len(self.states)}, "
                  f"Accept: {sum(self.accept_mask)}, "
                  f"Transitions: {sum(len(t) for t in self.transitions.values())}")

    # =========================================================================
    # Interface for BetaBackwardDP
    # =========================================================================

    def get_initial_state_id(self) -> int:
        initial = self._get_initial_state()
        return self.state_to_id.get(initial, 0)

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
        """Apply step. Return values:
          - int != state_id : state-changing transition
          - state_id        : identity (no abstract effect, or destination
                              pruned/unenumerated — goal-dropped atom case)
          - None            : hard reject (precondition fail)
        """
        if state_id < 0 or state_id >= len(self.states):
            return None
        src_state = self.states[state_id]
        pred = step.action
        cids = step.args

        if not self._check_precondition(src_state, pred, cids):
            return None  # hard reject — DP gets NEG_INF

        dst_state = self._apply_effect(src_state, pred, cids)
        next_id = self.state_to_id.get(dst_state)
        # Destination not in the BFS-explored set: the step only affects
        # dropped goals, so it is an identity in the coarse β.
        return state_id if next_id is None else next_id

"""
VirtualHome β State Machine: Top-Down DFA Construction with Canonical IDs.

State = (agent_loc, holding_L, holding_R, posture, facing, open_bits, on_bits, plugged_bits, edge_bits, action_goal_bits)

Key design decisions:
- State is explicitly enumerated (top-down), not discovered via BFS
- Accept states are precomputed at construction time
- Character edge goals are derived from state (no extra bits)
- Goal dropping if the state space exceeds max_states (dropped goals are no longer enforced)

CANONICAL IDs:
- Objects are assigned canonical IDs 0, 1, 2, ..., N-1 based on sorted (name, raw_id) order
- N (number of objects) is used as NONE/OTHER sentinel
- All internal state (VHBetaState) and actions (VHStep) use canonical IDs
- Sorting: by (name, int(raw_id)) if raw_id is numeric, else (name, raw_id)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, NamedTuple
from collections import defaultdict

from ctrlg.schemas.vh_as import ACTIONS as _VH_VOCAB_ACTIONS


# =============================================================================
# Constants
# =============================================================================

MAX_STATES = 150_000

# Sentinel cid for the character, which has no canonical ID (parse_objects
# skips it). Used as target_cid for goals like "X is ON to character".
CHARACTER_CID = -1


# =============================================================================
# State Representation
# =============================================================================

class VHBetaState(NamedTuple):
    """
    VirtualHome β state using canonical IDs.

    Multi-categorical (using canonical IDs 0..N-1, with N as sentinel):
        agent_loc: Where is the agent (canonical ID, or N=OTHER/unknown)
        holding_L: What's in left hand (canonical ID, or N=NONE)
        holding_R: What's in right hand (canonical ID, or N=NONE)
        facing: What is agent facing (canonical ID, or N=FACING_NONE)

    Categorical:
        posture: Agent posture (POSTURE_STANDING=0, POSTURE_SITTING=1, POSTURE_LYING=2)

    Binary:
        open_bits: Bitmask for openable objects (1=OPEN, 0=CLOSED)
        on_bits: Bitmask for switchable objects (1=ON, 0=OFF)
        plugged_bits: Bitmask for pluggable objects (1=PLUGGED_IN, 0=OUT)
        edge_bits: Bitmask for object edge goals satisfied
    """
    agent_loc: int
    holding_L: int
    holding_R: int
    posture: int  # 0=standing, 1=sitting, 2=lying
    facing: int   # canonical ID of faced object, or sentinel for none
    open_bits: int
    on_bits: int
    plugged_bits: int
    edge_bits: int
    action_goal_bits: int  # bitmask for action goals satisfied (in order)


# Posture constants
POSTURE_STANDING = 0
POSTURE_SITTING = 1
POSTURE_LYING = 2


@dataclass
class VHStep:
    """A grounded action step with canonical object IDs (0..N-1)."""
    action: str
    args: Tuple[int, ...]  # Canonical object IDs (integers 0..N-1)

    def __hash__(self):
        return hash((self.action, self.args))

    def __eq__(self, other):
        return self.action == other.action and self.args == other.args


# =============================================================================
# Goal Structures
# =============================================================================

@dataclass
class NodeGoal:
    """Object state goal: obj should be in state."""
    obj_cid: int  # Canonical object ID (0..N-1)
    state: str  # 'OPEN', 'CLOSED', 'ON', 'OFF', 'PLUGGED_IN', 'PLUGGED_OUT'


@dataclass
class ObjectEdgeGoal:
    """Object placement goal: obj should be relation to target."""
    obj_cid: int      # Canonical object ID (0..N-1)
    relation: str     # 'ON', 'INSIDE'
    target_cid: int   # Canonical target ID (0..N-1)


@dataclass
class CharacterEdgeGoal:
    """Character relation goal: character should be relation to obj."""
    relation: str  # 'HOLDS_LH', 'HOLDS_RH', 'INSIDE', 'FACING', 'CLOSE', 'ON'
    obj_cid: int   # Canonical object ID (0..N-1)


@dataclass
class CharacterStateGoal:
    """Character state goal: character should be in a specific posture."""
    state: str  # 'SITTING' or 'LYING'


@dataclass
class ActionGoal:
    """Action that must be performed. Any alternative satisfies the goal."""
    alternatives: FrozenSet[str]  # e.g., frozenset({'WASH', 'RINSE', 'SCRUB', 'WIPE'})


@dataclass
class ParsedGoals:
    """All goals parsed from prompt."""
    node_goals: List[NodeGoal]
    object_edge_goals: List[ObjectEdgeGoal]
    character_edge_goals: List[CharacterEdgeGoal]
    character_state_goals: List['CharacterStateGoal']  # character SITTING/LYING goals
    action_goals: List[ActionGoal]  # ordered list of required actions


# =============================================================================
# Object Info
# =============================================================================

@dataclass
class ObjectInfo:
    """
    Information about objects in the scene using CANONICAL IDs.

    All IDs in this class are canonical IDs (0, 1, 2, ..., N-1).
    N (n_objects) is used as the NONE/OTHER sentinel.
    """
    # Core info
    n_objects: int  # Total number of objects; also used as NONE/OTHER sentinel

    # Object data indexed by canonical ID
    cid_to_name: Dict[int, str]      # canonical ID -> object name
    cid_to_raw: Dict[int, int]       # canonical ID -> raw ID (for debug/output)
    name_to_cids: Dict[str, List[int]]  # name -> list of canonical IDs

    # Capability lists (canonical IDs)
    graspable: List[int]   # Canonical IDs that are GRABBABLE
    openable: List[int]    # Canonical IDs that have CAN_OPEN
    switchable: List[int]  # Canonical IDs that have HAS_SWITCH
    pluggable: List[int]   # Canonical IDs that have HAS_PLUG
    sittable: List[int]    # Canonical IDs that are SITTABLE or LIEABLE
    rooms: List[int]       # Canonical IDs that are rooms

    # Capability sets for quick lookup (canonical IDs)
    graspable_set: FrozenSet[int]
    openable_set: FrozenSet[int]
    switchable_set: FrozenSet[int]
    pluggable_set: FrozenSet[int]
    sittable_set: FrozenSet[int]
    rooms_set: FrozenSet[int]

    # Room containment: room canonical ID -> set of canonical IDs inside that room
    room_contents: Dict[int, Set[int]] = None

    def raw_to_cid(self, raw_id: int) -> Optional[int]:
        """Convert raw object ID to canonical ID, or None if not found."""
        for cid, rid in self.cid_to_raw.items():
            if rid == raw_id:
                return cid
        return None


# =============================================================================
# Prompt Parsing
# =============================================================================

ROOM_KEYWORDS = {
    'bathroom', 'bedroom', 'kitchen', 'livingroom', 'living_room',
    'dining_room', 'home_office', 'office', 'hallway', 'entrance',
    'garage', 'laundry', 'basement', 'attic', 'closet',
}


def _canonical_sort_key(pair: Tuple[str, int]) -> Tuple[str, int]:
    """
    Sort key for canonical ID assignment.

    Matches the training code's _sorted_pairs function:
    - Sort by (name, int(raw_id)) if raw_id is numeric
    - Sort by (name, str(raw_id)) otherwise
    """
    name, raw_id = pair
    return (str(name), int(raw_id) if str(raw_id).isdigit() else str(raw_id))


def parse_objects(prompt: str) -> ObjectInfo:
    """
    Parse objects and their capabilities from prompt.

    Returns ObjectInfo with CANONICAL IDs (0, 1, 2, ..., N-1).
    Objects are sorted by (name, raw_id) to match the training code's canonical ID scheme.
    """
    # Find objects section
    match = re.search(r'Objects in the scene:\n(.*?)\n-{3,}', prompt, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'Objects in the scene' section")

    objects_text = match.group(1).strip()

    # First pass: collect all (name, raw_id, properties)
    raw_objects: List[Tuple[str, int, Set[str]]] = []

    for line in objects_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Parse: name, id: X, properties: [...]
        # Note: [\w-]+ to handle hyphenated names like shoe-shine_kit
        name_match = re.match(r'([\w-]+),\s*id:\s*(\d+)', line)
        if not name_match:
            continue

        name = name_match.group(1).lower()
        raw_id = int(name_match.group(2))

        # Skip character
        if name == 'character':
            continue

        # Parse properties
        props_match = re.search(r'properties:\s*\[(.*?)\]', line)
        props = set()
        if props_match:
            for p in props_match.group(1).split(','):
                p = p.strip().strip("'\"").upper()
                if p:
                    props.add(p)

        raw_objects.append((name, raw_id, props))

    # Sort by (name, raw_id) to get canonical ordering
    sorted_objects = sorted(raw_objects, key=lambda x: _canonical_sort_key((x[0], x[1])))

    # Build canonical ID mappings
    n_objects = len(sorted_objects)
    cid_to_name: Dict[int, str] = {}
    cid_to_raw: Dict[int, int] = {}
    raw_to_cid: Dict[int, int] = {}  # For internal use during parsing
    name_to_cids: Dict[str, List[int]] = defaultdict(list)

    graspable: List[int] = []
    openable: List[int] = []
    switchable: List[int] = []
    pluggable: List[int] = []
    sittable: List[int] = []
    rooms: List[int] = []

    for cid, (name, raw_id, props) in enumerate(sorted_objects):
        cid_to_name[cid] = name
        cid_to_raw[cid] = raw_id
        raw_to_cid[raw_id] = cid
        name_to_cids[name].append(cid)

        # Categorize by capability (using canonical ID)
        if 'GRABBABLE' in props:
            graspable.append(cid)
        if 'CAN_OPEN' in props:
            openable.append(cid)
        if 'HAS_SWITCH' in props:
            switchable.append(cid)
        if 'HAS_PLUG' in props:
            pluggable.append(cid)
        if 'SITTABLE' in props or 'LIEABLE' in props:
            sittable.append(cid)

        # Rooms: objects without properties, or named exactly like a room.
        if not props or name in ROOM_KEYWORDS:
            rooms.append(cid)

    return ObjectInfo(
        n_objects=n_objects,
        cid_to_name=cid_to_name,
        cid_to_raw=cid_to_raw,
        name_to_cids=dict(name_to_cids),
        graspable=graspable,
        openable=openable,
        switchable=switchable,
        pluggable=pluggable,
        sittable=sittable,
        rooms=rooms,
        graspable_set=frozenset(graspable),
        openable_set=frozenset(openable),
        switchable_set=frozenset(switchable),
        pluggable_set=frozenset(pluggable),
        sittable_set=frozenset(sittable),
        rooms_set=frozenset(rooms),
        room_contents={},  # Will be populated by parse_room_containment
    )


def parse_initial_states(prompt: str, objects: ObjectInfo) -> Dict[str, Dict[int, bool]]:
    """
    Parse initial object states from Nodes section.

    Returns dict with canonical IDs as keys.
    """
    initial = {
        'open': {},    # canonical ID -> True if OPEN
        'on': {},      # canonical ID -> True if ON
        'plugged': {}, # canonical ID -> True if PLUGGED_IN
    }

    # Find Nodes section
    match = re.search(r'Nodes:\n(.*?)\n\nEdges:', prompt, re.DOTALL)
    if not match:
        return initial

    nodes_text = match.group(1).strip()

    for line in nodes_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Parse: name, states: [...], properties: [...]
        name_match = re.match(r'([\w-]+),', line)
        if not name_match:
            continue

        name = name_match.group(1).lower()
        cids = objects.name_to_cids.get(name, [])

        if not cids:
            continue

        # Parse states
        states_match = re.search(r'states:\s*\[(.*?)\]', line)
        if not states_match:
            continue

        states = set()
        for s in states_match.group(1).split(','):
            s = s.strip().strip("'\"").upper()
            if s:
                states.add(s)

        for cid in cids:
            if cid in objects.openable_set:
                initial['open'][cid] = 'OPEN' in states
            if cid in objects.switchable_set:
                initial['on'][cid] = 'ON' in states
            if cid in objects.pluggable_set:
                initial['plugged'][cid] = 'PLUGGED_IN' in states

    return initial


def parse_initial_character_posture(prompt: str) -> int:
    """
    Parse character's initial posture from Nodes section.

    Looks for line: character, states: ['SITTING'], ... or ['LYING'], ...
    Returns:
        POSTURE_SITTING (1) if SITTING in states
        POSTURE_LYING (2) if LYING in states
        POSTURE_STANDING (0) otherwise
    """
    match = re.search(r'Nodes:\n(.*?)\n\nEdges:', prompt, re.DOTALL)
    if not match:
        return POSTURE_STANDING

    nodes_text = match.group(1).strip()

    for line in nodes_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        name_match = re.match(r'([\w-]+),', line)
        if not name_match:
            continue

        if name_match.group(1).lower() != 'character':
            continue

        states_match = re.search(r'states:\s*\[(.*?)\]', line)
        if not states_match:
            return POSTURE_STANDING

        for s in states_match.group(1).split(','):
            s = s.strip().strip("'\"").upper()
            if s == 'SITTING':
                return POSTURE_SITTING
            if s == 'LYING':
                return POSTURE_LYING

        return POSTURE_STANDING

    return POSTURE_STANDING


def parse_initial_agent_location(prompt: str, objects: ObjectInfo) -> Optional[int]:
    """
    Parse initial agent location from Edges section.

    Returns canonical ID of the room/location, or None if not found.
    """
    match = re.search(r'Edges:\n(.*?)\n-{3,}', prompt, re.DOTALL)
    if not match:
        return None

    edges_text = match.group(1).strip()

    for line in edges_text.split('\n'):
        line = line.strip().lower()
        if 'character' in line and 'inside' in line:
            # Format: <character> (65) is INSIDE to <room> (id)
            room_match = re.search(r'inside\s+to\s+<([\w-]+)>\s*\((\d+)\)', line)
            if room_match:
                raw_id = int(room_match.group(2))
                return objects.raw_to_cid(raw_id)

    return None


def parse_initial_character_holding(
    prompt: str, objects: ObjectInfo
) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse initial character holding state from Edges section.

    Looks for lines like:
        <character> (65) is HOLDS_LH to <obj_name> (obj_id)
        <character> (65) is HOLDS_RH to <obj_name> (obj_id)

    Returns (left_cid, right_cid) — canonical IDs, or None if not holding.
    """
    left_cid: Optional[int] = None
    right_cid: Optional[int] = None

    match = re.search(r'Edges:\n(.*?)\n-{3,}', prompt, re.DOTALL)
    if not match:
        return left_cid, right_cid

    edges_text = match.group(1).strip()

    for line in edges_text.split('\n'):
        line_lower = line.strip().lower()
        if 'character' not in line_lower:
            continue

        if 'holds_lh' in line_lower:
            # Format: <character> (65) is HOLDS_LH to <obj> (id)
            obj_match = re.search(
                r'holds_lh\s+to\s+<([\w-]+)>\s*\((\d+)\)', line_lower
            )
            if obj_match:
                raw_id = int(obj_match.group(2))
                left_cid = objects.raw_to_cid(raw_id)

        elif 'holds_rh' in line_lower:
            obj_match = re.search(
                r'holds_rh\s+to\s+<([\w-]+)>\s*\((\d+)\)', line_lower
            )
            if obj_match:
                raw_id = int(obj_match.group(2))
                right_cid = objects.raw_to_cid(raw_id)

    return left_cid, right_cid


def parse_room_containment(prompt: str, objects: ObjectInfo) -> Dict[int, Set[int]]:
    """
    Parse which objects are inside which rooms from Edges section.

    Returns:
        Dict mapping room canonical ID -> set of canonical IDs inside that room
    """
    room_contents: Dict[int, Set[int]] = defaultdict(set)

    match = re.search(r'Edges:\n(.*?)\n-{3,}', prompt, re.DOTALL)
    if not match:
        return dict(room_contents)

    edges_text = match.group(1).strip()

    for line in edges_text.split('\n'):
        line = line.strip()
        if not line or 'INSIDE' not in line.upper():
            continue

        # Format: <obj_name> (obj_id) is INSIDE to <room_name> (room_id)
        # Match pattern: <name> (id) is INSIDE to <name> (id)
        m = re.search(r'<([\w-]+)>\s*\((\d+)\)\s+is\s+INSIDE\s+to\s+<([\w-]+)>\s*\((\d+)\)', line, re.IGNORECASE)
        if m:
            obj_raw_id = int(m.group(2))
            room_raw_id = int(m.group(4))

            # Convert raw IDs to canonical IDs
            obj_cid = objects.raw_to_cid(obj_raw_id)
            room_cid = objects.raw_to_cid(room_raw_id)

            if obj_cid is None or room_cid is None:
                continue

            # Only track if room_cid is actually a room
            if room_cid in objects.rooms_set:
                room_contents[room_cid].add(obj_cid)
                # Room is also inside itself (for location matching)
                room_contents[room_cid].add(room_cid)

    return dict(room_contents)


def parse_container_containment(
    prompt: str, objects: ObjectInfo,
) -> Dict[int, int]:
    """Return {obj_cid: container_cid} for objects starting INSIDE an
    openable container.

    Used by the open-before-access preconditions: an object inside a closed
    container cannot be found, grabbed or otherwise interacted with until the
    container is opened. Only openable (CAN_OPEN) containers qualify; rooms
    and surfaces are excluded.
    """
    obj_to_container: Dict[int, int] = {}
    openable: Set[int] = set(objects.openable or [])

    match = re.search(r'Edges:\n(.*?)\n-{3,}', prompt, re.DOTALL)
    if not match:
        return obj_to_container
    edges_text = match.group(1).strip()

    for line in edges_text.split('\n'):
        line = line.strip()
        if not line or 'INSIDE' not in line.upper():
            continue
        m = re.search(
            r'<([\w-]+)>\s*\((\d+)\)\s+is\s+INSIDE\s+to\s+<([\w-]+)>\s*\((\d+)\)',
            line, re.IGNORECASE,
        )
        if not m:
            continue
        obj_raw_id = int(m.group(2))
        container_raw_id = int(m.group(4))
        obj_cid = objects.raw_to_cid(obj_raw_id)
        container_cid = objects.raw_to_cid(container_raw_id)
        if obj_cid is None or container_cid is None:
            continue
        if container_cid in objects.rooms_set:
            continue  # Rooms tracked separately by parse_room_containment
        if container_cid not in openable:
            continue  # Non-openable container — no open precondition
        obj_to_container[obj_cid] = container_cid

    return obj_to_container


def _unique(items: list) -> list:
    """Items in first-occurrence order without repeats (by equality)."""
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def parse_goals(prompt: str, objects: ObjectInfo) -> ParsedGoals:
    """
    Parse all goals from prompt.

    All goals use canonical IDs (0, 1, 2, ..., N-1).
    """
    node_goals = []
    object_edge_goals = []
    character_edge_goals = []
    character_state_goals = []

    # Parse Node goals. Each goal section is bounded by the next section
    # header, not by the dashes line: in an empty section the dashes line
    # directly follows the header, and a `.*?\n-{3,}` match would run into the
    # next section (e.g. reading an edge goal as a node goal). The captured
    # dashes line does not match the goal pattern and is skipped.
    node_match = re.search(r'Node goals are:\n(.*?)\nEdge goals are:', prompt, re.DOTALL)
    if node_match:
        node_text = node_match.group(1).strip()
        for line in node_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Format: object_name is STATE
            m = re.match(r'([\w-]+)\s+is\s+(\w+)', line)
            if m:
                obj_name = m.group(1).lower()
                state = m.group(2).upper()

                # Special case: character posture goals (SITTING/LYING)
                if obj_name == 'character' and state in ('SITTING', 'LYING'):
                    character_state_goals.append(CharacterStateGoal(state=state))
                    continue

                cids = objects.name_to_cids.get(obj_name, [])
                for cid in cids:
                    node_goals.append(NodeGoal(obj_cid=cid, state=state))

    # Parse Edge goals (bounded by the "Action goals are:" header, as above).
    edge_match = re.search(r'Edge goals are:\n(.*?)\nAction goals are:', prompt, re.DOTALL)
    if edge_match:
        edge_text = edge_match.group(1).strip()
        for line in edge_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Format: obj_A is RELATION to obj_B
            m = re.match(r'([\w-]+)\s+is\s+(\w+)\s+to\s+([\w-]+)', line)
            if m:
                subj_name = m.group(1).lower()
                relation = m.group(2).upper()
                obj_name = m.group(3).lower()

                if subj_name == 'character':
                    # Character edge goal
                    cids = objects.name_to_cids.get(obj_name, [])
                    for cid in cids:
                        character_edge_goals.append(
                            CharacterEdgeGoal(relation=relation, obj_cid=cid)
                        )
                else:
                    # Object edge goal
                    subj_cids = objects.name_to_cids.get(subj_name, [])
                    # "X is ON to character" (X is worn). The character has
                    # no canonical ID, so the goal targets the CHARACTER_CID
                    # sentinel; arity-1 PUTON in _apply_action_effect sets the
                    # edge bit against the same sentinel.
                    if obj_name == 'character':
                        target_cids = [CHARACTER_CID]
                    else:
                        target_cids = objects.name_to_cids.get(obj_name, [])
                    for subj_cid in subj_cids:
                        for target_cid in target_cids:
                            object_edge_goals.append(
                                ObjectEdgeGoal(obj_cid=subj_cid, relation=relation, target_cid=target_cid)
                            )

    # Parse Action goals
    # Format: pairs of lines (human-readable + pipe-separated machine-readable)
    # e.g., "GRAB\nGRAB\nWASH or RINSE\nWASH|RINSE" → [{'GRAB'}, {'WASH','RINSE'}]
    # Duplicate line is formatting, not "do twice". Parse pipe-separated line only.
    action_goals = []
    action_match = re.search(r'Action goals are:\n(.*?)(?:\n-{3,}|\nPlease output)', prompt, re.DOTALL)
    if action_match:
        section = action_match.group(1).strip()
        if 'no action requirement' not in section.lower() and 'should be included' in section.lower():
            lines = section.split('\n')
            goal_lines = lines[1:]  # skip "The following action(s) should be included:"
            # Parse pairs: human-readable line, then pipe-separated machine-readable line.
            # Alternatives outside the γ vocabulary (`_VH_VOCAB_ACTIONS`) are
            # dropped, since β could never accept a goal γ cannot emit. A goal
            # with no remaining alternative is skipped as untrackable.
            for i in range(0, len(goal_lines), 2):
                if i + 1 < len(goal_lines):
                    machine_line = goal_lines[i + 1].strip()
                    if machine_line:
                        raw_alts = machine_line.split('|')
                        kept = frozenset(a for a in raw_alts if a in _VH_VOCAB_ACTIONS)
                        if not kept:
                            continue  # untrackable: no γ-emittable alternative
                        action_goals.append(ActionGoal(alternatives=kept))

    # A goal line is repeated once per goal instance, and each line is
    # expanded to every instance of its object names, so the same grounded
    # goal can appear several times (e.g. two "clothes_shirt is ON to
    # washing_machine" lines with two shirts). Keep one copy: a duplicate
    # edge goal would get a bit that is never set and make accept unreachable.
    node_goals = _unique(node_goals)
    object_edge_goals = _unique(object_edge_goals)
    character_edge_goals = _unique(character_edge_goals)

    return ParsedGoals(
        node_goals=node_goals,
        object_edge_goals=object_edge_goals,
        character_edge_goals=character_edge_goals,
        character_state_goals=character_state_goals,
        action_goals=action_goals,
    )


# =============================================================================
# State Machine Builder
# =============================================================================

class VHBetaStateMachine:
    """
    VirtualHome β State Machine.

    Constructs a top-down DFA for β constraints:
    - Enumerates all states explicitly
    - Precomputes accept mask
    - Builds transition graph from γ-valid actions

    Key design:
    - All edges come from gamma_valid_actions (actions with canonical IDs)
    - For each (state, action) pair, action falls into one of three buckets:
      1. Progress: transitions to different state
      2. Self-loop: stays in same state
      3. Disallowed: precondition fails (not in transitions)
    """

    def __init__(
        self,
        prompt: str,
        gamma_valid_actions: Optional[List[VHStep]] = None,
        max_states: int = MAX_STATES,
        debug: bool = False,
    ):
        """
        Build β state machine.

        Args:
            prompt: VirtualHome prompt with objects, initial states, goals
            gamma_valid_actions: List of γ-valid actions with canonical IDs.
                Required: these are the transition candidates.
            max_states: Maximum number of states before dropping goals
            debug: Print debug info
        """
        self.debug = debug
        self.max_states = max_states
        self.gamma_valid_actions = gamma_valid_actions

        # Parse prompt
        self.objects = parse_objects(prompt)
        self.objects.room_contents = parse_room_containment(prompt, self.objects)
        self.obj_to_container = parse_container_containment(prompt, self.objects)
        self.initial_obj_states = parse_initial_states(prompt, self.objects)
        self.initial_agent_loc = parse_initial_agent_location(prompt, self.objects)
        self.initial_agent_posture = parse_initial_character_posture(prompt)
        self.initial_holding_L_cid, self.initial_holding_R_cid = \
            parse_initial_character_holding(prompt, self.objects)
        self.goals = parse_goals(prompt, self.objects)

        # Objects with a HOLDS_RH goal are grabbed with the right hand first.
        self._holds_rh_goal_cids: Set[int] = {
            g.obj_cid for g in self.goals.character_edge_goals
            if g.relation == 'HOLDS_RH'
        }

        # Resolve contradictory goals before building
        self.dropped_contradictory_goals: List[CharacterEdgeGoal] = []
        self._fix_contradictory_goals()

        # Determine what to track based on goals
        self._setup_tracking()

        # Goal dropping if needed
        self._estimate_and_drop_goals()

        # Build index mappings
        self._build_indices()

        # Enumerate states
        self.states: List[VHBetaState] = []
        self.state_to_id: Dict[VHBetaState, int] = {}
        self._enumerate_states()

        # Compute accept mask
        self.accept_mask: List[bool] = []
        self._compute_accept_mask()

        # Sanity check: must have at least one accept state
        if self.num_accept == 0:
            goal_summary = []
            for g in self.goals.node_goals:
                obj_name = self.objects.cid_to_name.get(g.obj_cid, str(g.obj_cid))
                goal_summary.append(f"{obj_name} is {g.state}")
            for g in self.goals.object_edge_goals:
                obj_name = self.objects.cid_to_name.get(g.obj_cid, str(g.obj_cid))
                tgt_name = self.objects.cid_to_name.get(g.target_cid, str(g.target_cid))
                goal_summary.append(f"{obj_name} {g.relation} {tgt_name}")
            for g in self.goals.character_edge_goals:
                obj_name = self.objects.cid_to_name.get(g.obj_cid, str(g.obj_cid))
                goal_summary.append(f"character {g.relation} {obj_name}")
            raise ValueError(
                f"VHBetaStateMachine has no accept states! "
                f"Goals may be unsatisfiable: {goal_summary[:5]}..."
                if len(goal_summary) > 5 else f"Goals: {goal_summary}"
            )

        # Build transitions
        self.transitions: Dict[int, List[Tuple[VHStep, int]]] = {}
        self._build_transitions()

        if debug:
            self._print_summary()

    def _fix_contradictory_goals(self):
        """
        Detect and resolve goals that cannot hold together.

        Contradiction types detected:
        1. Same object in HOLDS_LH and HOLDS_RH with only one instance
           -> Drop one of the HOLDS goals (prefer keeping HOLDS_RH)

        2. Mutually exclusive node states (OPEN+CLOSED, ON+OFF)
           -> Keep the first goal, drop the second
        """
        # Count object instances by name
        obj_name_counts: Dict[str, int] = defaultdict(int)
        for cid in range(self.objects.n_objects):
            name = self.objects.cid_to_name.get(cid, '')
            obj_name_counts[name] += 1

        # Check 1: Same object needs to be in both hands
        holds_lh_objs = {g.obj_cid for g in self.goals.character_edge_goals
                        if g.relation == 'HOLDS_LH'}
        holds_rh_objs = {g.obj_cid for g in self.goals.character_edge_goals
                        if g.relation == 'HOLDS_RH'}

        # Find objects that appear in both HOLDS_LH and HOLDS_RH
        both_hands_objs = holds_lh_objs & holds_rh_objs

        for cid in both_hands_objs:
            obj_name = self.objects.cid_to_name.get(cid, '')
            # Check if there's only one instance of this object
            if obj_name_counts.get(obj_name, 1) <= 1:
                # Contradiction: same single object in both hands
                # Keep HOLDS_RH, drop HOLDS_LH
                goals_to_drop = [g for g in self.goals.character_edge_goals
                                 if g.relation == 'HOLDS_LH' and g.obj_cid == cid]
                for g in goals_to_drop:
                    self.goals.character_edge_goals.remove(g)
                    self.dropped_contradictory_goals.append(g)
                    if self.debug:
                        print(f"[VHBetaSM] Dropped contradictory goal: HOLDS_LH {obj_name} "
                              f"(can't hold same object in both hands)")

        # Check 2: Mutually exclusive node states
        node_goals_by_obj: Dict[int, List[NodeGoal]] = defaultdict(list)
        for g in self.goals.node_goals:
            node_goals_by_obj[g.obj_cid].append(g)

        exclusive_pairs = [
            ('OPEN', 'CLOSED'),
            ('ON', 'OFF'),
            ('PLUGGED_IN', 'PLUGGED_OUT'),
        ]

        for cid, obj_goals in node_goals_by_obj.items():
            states = {g.state for g in obj_goals}
            for s1, s2 in exclusive_pairs:
                if s1 in states and s2 in states:
                    # Contradiction: both mutually exclusive states required
                    # Drop s2, keep s1
                    goals_to_drop = [g for g in obj_goals if g.state == s2]
                    for g in goals_to_drop:
                        self.goals.node_goals.remove(g)
                        if self.debug:
                            obj_name = self.objects.cid_to_name.get(cid, str(cid))
                            print(f"[VHBetaSM] Dropped contradictory goal: {obj_name} is {s2} "
                                  f"(conflicts with {s1})")

    def _setup_tracking(self):
        """
        Determine which state bits to track.

        All IDs are canonical IDs (0..N-1).
        """
        # Track ALL openable objects (not just goal-relevant)
        # Reason: PUTIN requires target to be open (if openable)
        # This affects preconditions for non-goal objects too.
        self.all_openable = list(self.objects.openable)  # canonical IDs

        # Track ALL switchable objects (not just goal-relevant)
        # Reason: OPEN requires obj not-on, PLUGOUT requires not-on
        # ON/OFF state affects preconditions for non-goal objects too.
        self.all_switchable = list(self.objects.switchable)  # canonical IDs

        # Track ALL pluggable objects (not just goal-relevant)
        # Reason: SWITCHON requires obj plugged_in
        # Plugged state affects preconditions for non-goal objects too.
        self.all_pluggable = list(self.objects.pluggable)  # canonical IDs

        # Posture (WALK/SIT/LIE/STANDUP preconditions) and facing (LOOKAT/POINTAT
        # require FACING the target) are always tracked.

        # Object edge goals to track
        self.tracked_edge_goals = list(self.goals.object_edge_goals)
        self.dropped_edge_goals: List[ObjectEdgeGoal] = []

        # Action goals to track (ordered — must satisfy in sequence)
        self.action_goals: List[ActionGoal] = list(self.goals.action_goals)

        # Track ALL graspable objects (not just goal-relevant)
        # Reason: grabbing ANY object changes hand state, which affects:
        # - Whether we can grab something else (hand full)
        # - Whether we can do PUT actions (need to hold something)
        # This is a state transition, not a self-loop.
        self.all_graspable = list(self.objects.graspable)  # canonical IDs

    def _estimate_state_count(self) -> int:
        """Estimate state space size."""
        # agent_loc: 0..N-1 for objects, N for OTHER
        n_locs = self.objects.n_objects + 1
        # holding: 0..G-1 for graspable objects, NONE uses n_graspable as sentinel
        n_hold = len(self.all_graspable) + 1  # +1 for NONE sentinel
        # posture: 3 values (standing, sitting, lying)
        n_posture = 3
        # facing: N+1 values (0..N-1 for objects, N for FACING_NONE)
        n_facing = self.objects.n_objects + 1
        n_open = 2 ** len(self.all_openable)
        n_on = 2 ** len(self.all_switchable)
        n_plug = 2 ** len(self.all_pluggable)
        n_edge = 2 ** len(self.tracked_edge_goals)
        # Action goals: only prefix masks valid (ordering constraint), so N+1 values not 2^N
        n_action_goals = len(self.action_goals) + 1

        return n_locs * (n_hold ** 2) * n_posture * n_facing * n_open * n_on * n_plug * n_edge * n_action_goals

    def _estimate_and_drop_goals(self):
        """Drop goals if state space exceeds max_states."""

        estimate = self._estimate_state_count()

        if self.debug:
            print(f"[VHBetaSM] Initial state estimate: {estimate:,}")

        if estimate <= self.max_states:
            return

        # First drop object edge goals, those with the most-shared target first.
        target_counts: Dict[int, int] = defaultdict(int)
        for g in self.tracked_edge_goals:
            target_counts[g.target_cid] += 1

        # Sort by target frequency (drop high-frequency first)
        sorted_goals = sorted(
            self.tracked_edge_goals,
            key=lambda g: -target_counts[g.target_cid]
        )

        while self._estimate_state_count() > self.max_states and self.tracked_edge_goals:
            # Drop the goal with most shared target
            to_drop = sorted_goals.pop(0)
            self.tracked_edge_goals.remove(to_drop)
            self.dropped_edge_goals.append(to_drop)

            if self.debug:
                print(f"[VHBetaSM] Dropped edge goal: {to_drop}")

        if self.debug and self.dropped_edge_goals:
            print(f"[VHBetaSM] After P1 (edge goals): {self._estimate_state_count():,} states, "
                  f"dropped {len(self.dropped_edge_goals)} edge goals")

        # Then stop tracking open bits of objects without an OPEN/CLOSED goal.
        if self._estimate_state_count() > self.max_states:
            goal_open_cids = {g.obj_cid for g in self.goals.node_goals
                              if g.state in ('OPEN', 'CLOSED')}
            old_count = len(self.all_openable)
            self.all_openable = [c for c in self.all_openable if c in goal_open_cids]
            if self.debug:
                print(f"[VHBetaSM] P4: openable {old_count} → {len(self.all_openable)}, "
                      f"estimate {self._estimate_state_count():,}")

        # Then on bits of objects without an ON/OFF goal.
        if self._estimate_state_count() > self.max_states:
            goal_on_cids = {g.obj_cid for g in self.goals.node_goals
                            if g.state in ('ON', 'OFF')}
            old_count = len(self.all_switchable)
            self.all_switchable = [c for c in self.all_switchable if c in goal_on_cids]
            if self.debug:
                print(f"[VHBetaSM] P5: switchable {old_count} → {len(self.all_switchable)}, "
                      f"estimate {self._estimate_state_count():,}")

        # Then plugged bits of objects without a PLUGGED_IN/PLUGGED_OUT goal.
        if self._estimate_state_count() > self.max_states:
            goal_plug_cids = {g.obj_cid for g in self.goals.node_goals
                              if g.state in ('PLUGGED_IN', 'PLUGGED_OUT')}
            old_count = len(self.all_pluggable)
            self.all_pluggable = [c for c in self.all_pluggable if c in goal_plug_cids]
            if self.debug:
                print(f"[VHBetaSM] P6: pluggable {old_count} → {len(self.all_pluggable)}, "
                      f"estimate {self._estimate_state_count():,}")

        final_estimate = self._estimate_state_count()
        if self.debug:
            print(f"[VHBetaSM] Final state estimate: {final_estimate:,}")

        if final_estimate > self.max_states:
            raise ValueError(
                f"State space ({final_estimate:,}) exceeds max_states ({self.max_states:,}) "
                f"even after dropping through priority 6. "
                f"Scene has too many objects."
            )

    def _build_indices(self):
        """
        Build index mappings for state encoding.

        With canonical IDs:
        - agent_loc uses canonical ID directly (0..N-1), with N as OTHER
        - holding uses grasp_idx (index into all_graspable, 0..G-1), with G as NONE
        - Bitmasks use index into the respective capability lists
        """
        # Sentinel values
        self.LOC_OTHER = self.objects.n_objects  # N = unknown/other location
        self.HOLD_NONE = len(self.all_graspable)  # G = not holding anything
        self.FACING_NONE = self.objects.n_objects  # N = not facing anything

        # Graspable: map canonical ID -> grasp index (0..G-1)
        # grasp_idx -> canonical ID is just all_graspable[grasp_idx]
        self.cid_to_grasp_idx = {cid: idx for idx, cid in enumerate(self.all_graspable)}

        # Openable: canonical ID -> bit index for bitmask
        self.open_to_bit = {cid: i for i, cid in enumerate(self.all_openable)}

        # Switchable: canonical ID -> bit index for bitmask
        self.on_to_bit = {cid: i for i, cid in enumerate(self.all_switchable)}

        # Pluggable: canonical ID -> bit index for bitmask
        self.plug_to_bit = {cid: i for i, cid in enumerate(self.all_pluggable)}

        # Edge goal index for bitmask (using canonical IDs)
        self.edge_to_bit = {
            (g.obj_cid, g.relation, g.target_cid): i
            for i, g in enumerate(self.tracked_edge_goals)
        }

    def _enumerate_states(self):
        """
        Enumerate all valid states.

        State encoding:
        - agent_loc: canonical ID (0..N-1) or N (OTHER)
        - holding_L/R: grasp_idx (0..G-1) or G (NONE)
        - posture: 0 (standing), 1 (sitting), 2 (lying)
        - facing: canonical ID (0..N-1) or N (FACING_NONE)
        """
        n_locs = self.objects.n_objects + 1  # 0..N-1 + OTHER
        n_grasp = len(self.all_graspable) + 1  # 0..G-1 + NONE
        posture_vals = [POSTURE_STANDING, POSTURE_SITTING, POSTURE_LYING]
        # Facing: object cids plus the FACING_NONE sentinel n_objects.
        facing_vals = list(range(self.objects.n_objects + 1))
        n_open = 2 ** len(self.all_openable)
        n_on = 2 ** len(self.all_switchable)
        n_plug = 2 ** len(self.all_pluggable)
        n_edge = 2 ** len(self.tracked_edge_goals)
        # Action goals: only prefix masks valid (ordering: goal 0 before goal 1)
        # Valid values: 0b000, 0b001, 0b011, 0b111, ... (i.e., (1<<i)-1 for i=0..N)
        valid_action_goal_bits = [(1 << i) - 1 for i in range(len(self.action_goals) + 1)]

        for agent_loc in range(n_locs):
            for holding_L in range(n_grasp):
                for holding_R in range(n_grasp):
                    # Can't hold same object in both hands
                    # (grasp_idx values are equal and not NONE)
                    if holding_L != self.HOLD_NONE and holding_L == holding_R:
                        continue

                    for posture in posture_vals:
                        for facing in facing_vals:
                            for open_bits in range(n_open):
                                for on_bits in range(n_on):
                                    for plugged_bits in range(n_plug):
                                        for edge_bits in range(n_edge):
                                            for action_goal_bits in valid_action_goal_bits:
                                                state = VHBetaState(
                                                    agent_loc=agent_loc,
                                                    holding_L=holding_L,
                                                    holding_R=holding_R,
                                                    posture=posture,
                                                    facing=facing,
                                                    open_bits=open_bits,
                                                    on_bits=on_bits,
                                                    plugged_bits=plugged_bits,
                                                    edge_bits=edge_bits,
                                                    action_goal_bits=action_goal_bits,
                                                )
                                                state_id = len(self.states)
                                                self.states.append(state)
                                                self.state_to_id[state] = state_id

    def _compute_accept_mask(self):
        """Precompute which states are accepting."""

        all_edge_mask = (1 << len(self.tracked_edge_goals)) - 1
        all_action_goal_mask = (1 << len(self.action_goals)) - 1

        for state in self.states:
            accept = self._is_accept(state, all_edge_mask, all_action_goal_mask)
            self.accept_mask.append(accept)

    def _is_accept(self, state: VHBetaState, all_edge_mask: int, all_action_goal_mask: int) -> bool:
        """Check if state satisfies all goals (using canonical IDs)."""

        # 1. Check node goals (object states)
        for goal in self.goals.node_goals:
            cid = goal.obj_cid
            if goal.state == 'OPEN':
                if cid in self.open_to_bit:
                    if not (state.open_bits & (1 << self.open_to_bit[cid])):
                        return False
            elif goal.state == 'CLOSED':
                if cid in self.open_to_bit:
                    if state.open_bits & (1 << self.open_to_bit[cid]):
                        return False
            elif goal.state == 'ON':
                if cid in self.on_to_bit:
                    if not (state.on_bits & (1 << self.on_to_bit[cid])):
                        return False
            elif goal.state == 'OFF':
                if cid in self.on_to_bit:
                    if state.on_bits & (1 << self.on_to_bit[cid]):
                        return False
            elif goal.state == 'PLUGGED_IN':
                if cid in self.plug_to_bit:
                    if not (state.plugged_bits & (1 << self.plug_to_bit[cid])):
                        return False
            elif goal.state == 'PLUGGED_OUT':
                if cid in self.plug_to_bit:
                    if state.plugged_bits & (1 << self.plug_to_bit[cid]):
                        return False

        # 2. Check character edge goals (derived from state)
        #
        # Build set of objects that have FACING/CLOSE goals (stricter than INSIDE).
        # If we satisfy a FACING/CLOSE goal, we're implicitly "near" that object,
        # which should satisfy any INSIDE room goal (data may omit object-in-room edges).
        facing_close_objs = {g.obj_cid for g in self.goals.character_edge_goals
                            if g.relation in ('FACING', 'CLOSE')}

        for goal in self.goals.character_edge_goals:
            cid = goal.obj_cid
            if goal.relation == 'HOLDS_LH':
                grasp_idx = self.cid_to_grasp_idx.get(cid, -1)
                if grasp_idx == -1 or state.holding_L != grasp_idx:
                    return False
            elif goal.relation == 'HOLDS_RH':
                grasp_idx = self.cid_to_grasp_idx.get(cid, -1)
                if grasp_idx == -1 or state.holding_R != grasp_idx:
                    return False
            elif goal.relation == 'INSIDE':
                # INSIDE room: agent can be at any location within the room
                if cid in self.objects.rooms_set:
                    # Get all locations (canonical IDs) inside this room
                    room_locs = self.objects.room_contents.get(cid, set())
                    # Include the room itself as a valid location
                    room_locs = room_locs | {cid}

                    # Also include objects with FACING/CLOSE goals (they're implicitly in the room)
                    room_locs = room_locs | facing_close_objs

                    # Check if agent is at any of these locations
                    # state.agent_loc is a canonical ID (or LOC_OTHER)
                    if state.agent_loc not in room_locs:
                        return False
                else:
                    # Not a room, treat as exact location match
                    # cid is the canonical ID, state.agent_loc is also canonical ID
                    if state.agent_loc != cid:
                        return False
            elif goal.relation == 'CLOSE':
                # CLOSE requires exact location match (canonical ID)
                if state.agent_loc != cid:
                    return False
            elif goal.relation == 'FACING':
                # FACING requires state.facing == cid (set by TURNTO)
                if state.facing != cid:
                    return False
            elif goal.relation == 'ON':
                # Character ON furniture requires sitting/lying + location (canonical ID)
                if state.agent_loc != cid or state.posture == POSTURE_STANDING:
                    return False

        # 3. Check character state goals (SITTING/LYING)
        for goal in self.goals.character_state_goals:
            if goal.state == 'SITTING' and state.posture != POSTURE_SITTING:
                return False
            if goal.state == 'LYING' and state.posture != POSTURE_LYING:
                return False

        # 4. Check object edge goals (tracked in edge_bits)
        if state.edge_bits != all_edge_mask:
            return False

        # 5. Check action goals (tracked in action_goal_bits)
        if state.action_goal_bits != all_action_goal_mask:
            return False

        # 6. Dropped goals are assumed satisfied

        return True

    def _build_transitions(self):
        """
        Build transitions with the gamma-valid actions as edge candidates.

        Per state, an action whose preconditions fail is left out; otherwise
        its effect gives the next state (possibly the same state).
        """
        assert self.gamma_valid_actions is not None, \
            "gamma_valid_actions must be provided to build transitions"
        if self.debug:
            print(f"[VHBetaSM] Building transitions from {len(self.gamma_valid_actions)} γ-valid actions")

        # Stats
        total_progress = 0
        total_selfloop = 0
        total_disallowed = 0

        for state_id, state in enumerate(self.states):
            self.transitions[state_id] = []

            for step in self.gamma_valid_actions:
                # Check preconditions
                if not self.check_preconditions(state, step.action, step.args):
                    total_disallowed += 1
                    continue  # Disallowed - precondition fails

                # Compute effect (next state)
                next_state = self._apply_action_effect(state, step)
                if next_state is not None:
                    next_state = self._maybe_advance_action_goal(next_state, step.action)

                if next_state is None:
                    # Action has no effect model - skip
                    total_disallowed += 1
                    continue

                # Check if next_state exists in our state space
                if next_state not in self.state_to_id:
                    # State not enumerated - skip (shouldn't happen if states are complete)
                    total_disallowed += 1
                    continue

                next_id = self.state_to_id[next_state]

                # Add to transitions (either progress or self-loop)
                self.transitions[state_id].append((step, next_id))

                if next_id == state_id:
                    total_selfloop += 1
                else:
                    total_progress += 1

        if self.debug:
            print(f"[VHBetaSM] Transitions built: {total_progress} progress, "
                  f"{total_selfloop} self-loops, {total_disallowed} disallowed")


    # Toggle effects: action -> (bit map, state field, new bit value).
    _TOGGLE_EFFECTS = {
        'OPEN': ('open_to_bit', 'open_bits', True),
        'CLOSE': ('open_to_bit', 'open_bits', False),
        'SWITCHON': ('on_to_bit', 'on_bits', True),
        'SWITCHOFF': ('on_to_bit', 'on_bits', False),
        'PLUGIN': ('plug_to_bit', 'plugged_bits', True),
        'PLUGOUT': ('plug_to_bit', 'plugged_bits', False),
    }

    def _apply_action_effect(self, state: VHBetaState, step: VHStep) -> Optional[VHBetaState]:
        """
        Apply action effect to state and return next state.

        This computes the state transition WITHOUT checking preconditions.
        Preconditions should be checked separately via check_preconditions().

        Args:
            state: Current VHBetaState
            step: VHStep with canonical IDs

        Returns:
            Next VHBetaState, or None if action has no state effect
        """
        action = step.action.upper()
        cids = step.args

        # ===== WALK: change agent location, reset facing =====
        # Navigation clears FACING (simulator constraint); TURNTO is needed again.
        if action in ('WALK', 'RUN', 'FIND'):
            if cids:
                new_loc = cids[0]  # Canonical ID of destination
                return state._replace(agent_loc=new_loc, facing=self.FACING_NONE)
            return state._replace(facing=self.FACING_NONE)

        # ===== GRAB: put object in free hand (goal-aware preference) =====
        if action == 'GRAB':
            if cids:
                cid = cids[0]
                grasp_idx = self.cid_to_grasp_idx.get(cid)
                if grasp_idx is not None:
                    # Clear edge bits for grabbed object
                    new_edge_bits = self._clear_edge_bits_for_obj(state.edge_bits, cid)

                    # Goal-aware hand preference:
                    # If object has HOLDS_RH goal, fill R first
                    # Otherwise, fill L first (default, also handles HOLDS_LH)
                    if cid in self._holds_rh_goal_cids:
                        # Object has HOLDS_RH goal - prefer right hand
                        if state.holding_R == self.HOLD_NONE:
                            return state._replace(holding_R=grasp_idx, edge_bits=new_edge_bits)
                        elif state.holding_L == self.HOLD_NONE:
                            return state._replace(holding_L=grasp_idx, edge_bits=new_edge_bits)
                    else:
                        # Default: fill left hand first (handles HOLDS_LH and no-goal cases)
                        if state.holding_L == self.HOLD_NONE:
                            return state._replace(holding_L=grasp_idx, edge_bits=new_edge_bits)
                        elif state.holding_R == self.HOLD_NONE:
                            return state._replace(holding_R=grasp_idx, edge_bits=new_edge_bits)
            return None  # Can't grab - hands full or not graspable

        # ===== PUTBACK/PUTON/DROP: release from hand, possibly set edge goal =====
        if action in ('PUTBACK', 'PUTON', 'DROP'):
            if len(cids) >= 2:  # PUTBACK
                held_cid = cids[0]
                target_cid = cids[1]
                grasp_idx = self.cid_to_grasp_idx.get(held_cid)
                if grasp_idx is not None:
                    # Set edge bit if this satisfies a goal
                    new_edge_bits = self._set_edge_bit_if_goal(state.edge_bits, held_cid, 'ON', target_cid)

                    # Release from hand
                    if state.holding_L == grasp_idx:
                        return state._replace(holding_L=self.HOLD_NONE, edge_bits=new_edge_bits)
                    elif state.holding_R == grasp_idx:
                        return state._replace(holding_R=self.HOLD_NONE, edge_bits=new_edge_bits)
            elif len(cids) == 1:
                held_cid = cids[0]
                grasp_idx = self.cid_to_grasp_idx.get(held_cid)
                if grasp_idx is not None:
                    # Arity-1 PUTON puts on held clothing; the implicit target
                    # is the character, so it satisfies "X is ON to character"
                    # goals (CHARACTER_CID, see parse_goals).
                    new_edge_bits = state.edge_bits
                    if action == 'PUTON':
                        new_edge_bits = self._set_edge_bit_if_goal(
                            state.edge_bits, held_cid, 'ON', CHARACTER_CID
                        )
                    if state.holding_L == grasp_idx:
                        return state._replace(holding_L=self.HOLD_NONE, edge_bits=new_edge_bits)
                    elif state.holding_R == grasp_idx:
                        return state._replace(holding_R=self.HOLD_NONE, edge_bits=new_edge_bits)
            return state  # No effect

        # ===== PUTIN: release into container, set INSIDE edge goal =====
        if action == 'PUTIN':
            if len(cids) >= 2:
                held_cid = cids[0]
                target_cid = cids[1]
                grasp_idx = self.cid_to_grasp_idx.get(held_cid)
                if grasp_idx is not None:
                    # Set edge bits for both INSIDE and ON (VH uses both)
                    new_edge_bits = self._set_edge_bit_if_goal(state.edge_bits, held_cid, 'INSIDE', target_cid)
                    new_edge_bits = self._set_edge_bit_if_goal(new_edge_bits, held_cid, 'ON', target_cid)

                    # Release from hand
                    if state.holding_L == grasp_idx:
                        return state._replace(holding_L=self.HOLD_NONE, edge_bits=new_edge_bits)
                    elif state.holding_R == grasp_idx:
                        return state._replace(holding_R=self.HOLD_NONE, edge_bits=new_edge_bits)
            return state

        # ===== OPEN/CLOSE, SWITCHON/SWITCHOFF, PLUGIN/PLUGOUT: set or clear a bit =====
        if action in self._TOGGLE_EFFECTS:
            bit_map, field, value = self._TOGGLE_EFFECTS[action]
            bit = getattr(self, bit_map).get(cids[0]) if cids else None
            if bit is None:
                return state
            bits = getattr(state, field)
            return state._replace(**{field: bits | (1 << bit) if value else bits & ~(1 << bit)})

        # ===== SIT: set posture to SITTING, change location =====
        if action == 'SIT':
            if cids:
                cid = cids[0]
                return state._replace(posture=POSTURE_SITTING, agent_loc=cid)
            return state._replace(posture=POSTURE_SITTING)

        # ===== LIE: set posture to LYING, change location =====
        if action == 'LIE':
            if cids:
                cid = cids[0]
                return state._replace(posture=POSTURE_LYING, agent_loc=cid)
            return state._replace(posture=POSTURE_LYING)

        # ===== STANDUP: set posture to STANDING =====
        if action == 'STANDUP':
            return state._replace(posture=POSTURE_STANDING)

        # ===== TURNTO: set facing to target object =====
        if action == 'TURNTO':
            if cids:
                cid = cids[0]
                return state._replace(facing=cid)
            return state

        # ===== All other actions: no state effect (self-loop) =====
        # LOOKAT, TYPE, WASH, SQUEEZE, CUT, EAT, TOUCH, MOVE,
        # DRINK, READ, POUR, WIPE, SLEEP, etc.
        return state


    def _clear_edge_bits_for_obj(self, edge_bits: int, obj_cid: int) -> int:
        """Clear edge goal bits for an object (when grabbed). Uses canonical IDs."""
        new_bits = edge_bits
        for (o, rel, t), bit in self.edge_to_bit.items():
            if o == obj_cid:
                new_bits &= ~(1 << bit)
        return new_bits

    def _set_edge_bit_if_goal(self, edge_bits: int, obj_cid: int, relation: str, target_cid: int) -> int:
        """Set edge goal bit if this placement satisfies a goal. Uses canonical IDs."""
        key = (obj_cid, relation, target_cid)
        if key in self.edge_to_bit:
            return edge_bits | (1 << self.edge_to_bit[key])
        return edge_bits

    def _maybe_advance_action_goal(self, state: VHBetaState, action: str) -> VHBetaState:
        """If action matches the next unsatisfied action goal (in order), set its bit."""
        if not self.action_goals:
            return state
        bits = state.action_goal_bits
        for i in range(len(self.action_goals)):
            if not (bits & (1 << i)):
                # This is the next goal to satisfy (ordering: must be sequential)
                if action.upper() in self.action_goals[i].alternatives:
                    return state._replace(action_goal_bits=bits | (1 << i))
                return state  # Action doesn't match next goal
        return state  # All goals already satisfied

    def get_initial_state(self) -> VHBetaState:
        """Get the initial state from prompt. Uses canonical IDs."""

        # Agent location (canonical ID or LOC_OTHER)
        agent_loc = self.LOC_OTHER
        if self.initial_agent_loc is not None:
            agent_loc = self.initial_agent_loc  # Already a canonical ID

        # Initial object states (keyed by canonical ID)
        open_bits = 0
        for cid, is_open in self.initial_obj_states['open'].items():
            if cid in self.open_to_bit and is_open:
                open_bits |= (1 << self.open_to_bit[cid])

        on_bits = 0
        for cid, is_on in self.initial_obj_states['on'].items():
            if cid in self.on_to_bit and is_on:
                on_bits |= (1 << self.on_to_bit[cid])

        plugged_bits = 0
        for cid, is_plugged in self.initial_obj_states['plugged'].items():
            if cid in self.plug_to_bit and is_plugged:
                plugged_bits |= (1 << self.plug_to_bit[cid])

        # Initial holding state (canonical ID → grasp index)
        holding_L = self.HOLD_NONE
        if self.initial_holding_L_cid is not None:
            grasp_idx = self.cid_to_grasp_idx.get(self.initial_holding_L_cid)
            if grasp_idx is not None:
                holding_L = grasp_idx

        holding_R = self.HOLD_NONE
        if self.initial_holding_R_cid is not None:
            grasp_idx = self.cid_to_grasp_idx.get(self.initial_holding_R_cid)
            if grasp_idx is not None:
                holding_R = grasp_idx

        # Initial facing: not facing anything
        facing = self.FACING_NONE

        return VHBetaState(
            agent_loc=agent_loc,
            holding_L=holding_L,
            holding_R=holding_R,
            posture=self.initial_agent_posture,
            facing=facing,
            open_bits=open_bits,
            on_bits=on_bits,
            plugged_bits=plugged_bits,
            edge_bits=0,  # No edge goals satisfied initially
            action_goal_bits=0,  # No action goals satisfied initially
        )

    def get_initial_state_id(self) -> int:
        """Get the initial state ID."""
        initial = self.get_initial_state()
        return self.state_to_id.get(initial, 0)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_accept(self) -> int:
        return sum(self.accept_mask)

    def is_accept(self, state_id: int) -> bool:
        """Check if a state is an accept state by ID."""
        if 0 <= state_id < len(self.accept_mask):
            return self.accept_mask[state_id]
        return False

    def apply(self, state_id: int, step: VHStep) -> Optional[int]:
        """
        Apply a step to a state and return the next state ID.

        Args:
            state_id: Current state ID
            step: VHStep to apply

        Returns:
            Next state ID, or None if transition not found
        """
        if state_id not in self.transitions:
            return None

        for trans_step, next_id in self.transitions[state_id]:
            if trans_step.action == step.action and trans_step.args == step.args:
                return next_id

        return None

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------

    # Actions that need the agent CLOSE to arg[0] (interaction target), and
    # those that need it CLOSE to arg[1] (destination of PUT*/POUR).
    # PLUGIN/PLUGOUT also need proximity (simulator constraint).
    _CLOSE_ARG0 = frozenset({
        'GRAB', 'OPEN', 'CLOSE', 'SWITCHON', 'SWITCHOFF', 'SIT', 'LIE',
        'TOUCH', 'WIPE', 'TYPE', 'MOVE', 'PUSH', 'PULL', 'WASH', 'RINSE',
        'SCRUB', 'SQUEEZE', 'READ', 'CUT', 'EAT', 'PLUGIN', 'PLUGOUT',
    })
    _CLOSE_ARG1 = frozenset({'PUTBACK', 'PUTIN', 'POUR'})

    # Body parts (hands_both, face, hair, ...) are never CLOSE to the agent
    # (simulator constraint), so actions that need proximity cannot target
    # them, and they are not navigable. β then grounds WASH/RINSE/etc. onto
    # real objects such as the sink or faucet. Matched by exact name, since
    # a substring test would catch 'hair' in 'chair' or 'face' in 'surface'.
    _BODY_PARTS = frozenset({'hands_both', 'face', 'hair', 'teeth', 'eyes',
                             'arms_both', 'legs_both', 'feet_both'})

    # Toggle-state preconditions: action -> ((bit map, state field, required value), ...).
    # A check applies only when the object's bit is tracked.
    _TOGGLE_PRECONDITIONS = {
        'OPEN': (('open_to_bit', 'open_bits', False),    # closed ...
                 ('on_to_bit', 'on_bits', False)),       # ... and not on (PDDL)
        'CLOSE': (('open_to_bit', 'open_bits', True),),
        'SWITCHON': (('on_to_bit', 'on_bits', False),
                     ('plug_to_bit', 'plugged_bits', True)),  # plugged in (PDDL)
        'SWITCHOFF': (('on_to_bit', 'on_bits', True),),
        'PLUGIN': (('plug_to_bit', 'plugged_bits', False),),
        'PLUGOUT': (('plug_to_bit', 'plugged_bits', True),
                    ('on_to_bit', 'on_bits', False)),    # not on (PDDL)
    }

    def _is_bodypart(self, cid: int) -> bool:
        return self.objects.cid_to_name.get(cid, '') in self._BODY_PARTS

    def _inside_closed(self, state: VHBetaState, cid: int) -> bool:
        """True if `cid` is INSIDE a tracked openable container that is closed."""
        cont = self.obj_to_container.get(cid)
        if cont is None or cont not in self.open_to_bit:
            return False
        return not (state.open_bits & (1 << self.open_to_bit[cont]))

    def _holds(self, state: VHBetaState, cid: int) -> bool:
        grasp_idx = self.cid_to_grasp_idx.get(cid)
        return grasp_idx is not None and grasp_idx in (state.holding_L, state.holding_R)

    def _proximity_ok(self, state: VHBetaState, action: str, cids: Tuple[int, ...]) -> bool:
        """The agent must be CLOSE to the interaction target."""
        if action in self._CLOSE_ARG0 and cids:
            target = cids[0]
            if self._is_bodypart(target):
                return False
            # Rooms are entered (INSIDE), never CLOSE.
            if target in self.objects.rooms_set:
                return False
            if state.agent_loc != target:
                return False
            # Open-before-access (evaluator precondition): an object INSIDE a
            # closed container cannot be interacted with. No constraint if the
            # container's open bit was dropped to fit max_states.
            if self._inside_closed(state, target):
                return False
        elif action in self._CLOSE_ARG1 and len(cids) >= 2:
            if self._is_bodypart(cids[1]) or state.agent_loc != cids[1]:
                return False
        return True

    def _pre_locomotion(self, state, action, cids) -> bool:
        if cids and self._is_bodypart(cids[0]):
            return False
        # An object inside a closed container cannot be found (evaluator precondition).
        if action == 'FIND' and cids and self._inside_closed(state, cids[0]):
            return False
        return state.posture == POSTURE_STANDING

    def _pre_standing(self, state, action, cids) -> bool:
        return state.posture == POSTURE_STANDING

    def _pre_not_standing(self, state, action, cids) -> bool:
        return state.posture != POSTURE_STANDING

    def _pre_facing(self, state, action, cids) -> bool:
        # LOOKAT/POINTAT require FACING the target, which TURNTO sets.
        return state.facing == cids[0] if cids else True

    def _pre_grab(self, state, action, cids) -> bool:
        if state.holding_L != self.HOLD_NONE and state.holding_R != self.HOLD_NONE:
            return False  # both hands full
        # Not already holding the target. (A target inside a closed container
        # is rejected by the proximity check.)
        return not (cids and self._holds(state, cids[0]))

    def _pre_put(self, state, action, cids) -> bool:
        # The named object must be in a hand.
        if not (cids and self._holds(state, cids[0])):
            return False
        # Putting into an openable destination (PUTIN, or PUTBACK into e.g. an
        # oven) requires it to be open (simulator constraint).
        if action in ('PUTIN', 'PUTBACK') and len(cids) >= 2 and cids[1] in self.open_to_bit:
            return bool(state.open_bits & (1 << self.open_to_bit[cids[1]]))
        return True

    def _pre_toggle(self, state, action, cids) -> bool:
        if not cids:
            return True
        for bit_map, field, required in self._TOGGLE_PRECONDITIONS[action]:
            bit = getattr(self, bit_map).get(cids[0])
            if bit is not None and bool(getattr(state, field) & (1 << bit)) != required:
                return False
        return True

    def _pre_drink(self, state, action, cids) -> bool:
        # The simulator executes DRINK only on a held object (simulator
        # constraint), so non-graspable targets such as water are rejected.
        # γ also admits RECIPIENT+GRABBABLE targets (cups, glasses), so a
        # valid DRINK target exists whenever such a container does.
        if state.holding_L == self.HOLD_NONE and state.holding_R == self.HOLD_NONE:
            return False
        return self._holds(state, cids[0]) if cids else True

    def _pre_hold_if_graspable(self, state, action, cids) -> bool:
        # READ/POUR: a graspable target must be held.
        if cids and cids[0] in self.cid_to_grasp_idx:
            return self._holds(state, cids[0])
        return True

    _PRECONDITIONS = {
        'WALK': _pre_locomotion, 'RUN': _pre_locomotion, 'FIND': _pre_locomotion,
        'SIT': _pre_standing, 'LIE': _pre_standing,
        'STANDUP': _pre_not_standing, 'SLEEP': _pre_not_standing,
        'LOOKAT': _pre_facing, 'POINTAT': _pre_facing,
        'GRAB': _pre_grab,
        'PUTBACK': _pre_put, 'PUTIN': _pre_put, 'PUTON': _pre_put, 'DROP': _pre_put,
        'OPEN': _pre_toggle, 'CLOSE': _pre_toggle, 'SWITCHON': _pre_toggle,
        'SWITCHOFF': _pre_toggle, 'PLUGIN': _pre_toggle, 'PLUGOUT': _pre_toggle,
        'DRINK': _pre_drink,
        'READ': _pre_hold_if_graspable, 'POUR': _pre_hold_if_graspable,
    }

    def check_preconditions(self, state: VHBetaState, action: str, cids: Tuple[int, ...]) -> bool:
        """True if `action(cids)` may be executed from `state`.

        Checks, in order: no object in both argument slots; the agent is CLOSE
        to the interaction target; the action's own dynamic preconditions
        (posture, facing, hands, open/on/plugged bits), following the
        VirtualHome PDDL and the evaluator. Actions without an entry in
        `_PRECONDITIONS` (TURNTO, TYPE, WASH, CUT, EAT, TOUCH, ...) have no
        dynamic precondition. Static affordances are enforced by γ.

        Args:
            state: current VHBetaState
            action: action name (case-insensitive)
            cids: canonical object IDs
        """
        action = action.upper()
        if len(cids) >= 2 and cids[0] == cids[1]:
            return False
        if not self._proximity_ok(state, action, cids):
            return False
        check = self._PRECONDITIONS.get(action)
        return check is None or check(self, state, action, cids)

    def _print_summary(self):
        """Print summary of the state machine."""
        print(f"\n[VHBetaStateMachine Summary]")
        print(f"  Objects: {self.objects.n_objects} (canonical IDs 0..{self.objects.n_objects-1})")
        print(f"    Graspable: {len(self.objects.graspable)}")
        print(f"    Openable: {len(self.objects.openable)}")
        print(f"    Switchable: {len(self.objects.switchable)}")
        print(f"    Pluggable: {len(self.objects.pluggable)}")
        print(f"  Goals:")
        print(f"    Node goals: {len(self.goals.node_goals)}")
        print(f"    Object edge goals: {len(self.goals.object_edge_goals)} (tracked: {len(self.tracked_edge_goals)}, dropped: {len(self.dropped_edge_goals)})")
        print(f"    Character edge goals: {len(self.goals.character_edge_goals)}")
        print(f"  State tracking:")
        print(f"    All graspable: {len(self.all_graspable)}")
        print(f"    All openable: {len(self.all_openable)}")
        print(f"    All switchable: {len(self.all_switchable)}")
        print(f"    All pluggable: {len(self.all_pluggable)}")
        print(f"  Sentinels: LOC_OTHER={self.LOC_OTHER}, HOLD_NONE={self.HOLD_NONE}")
        print(f"  States: {self.num_states:,}")
        print(f"  Accept states: {self.num_accept:,}")
        print(f"  Transitions: {sum(len(t) for t in self.transitions.values()):,}")

        # Gamma-valid actions info
        if self.gamma_valid_actions is not None:
            print(f"  γ-valid actions: {len(self.gamma_valid_actions)}")
            # Count unique actions
            action_counts = defaultdict(int)
            for step in self.gamma_valid_actions:
                action_counts[step.action] += 1
            print(f"    Actions: {dict(action_counts)}")

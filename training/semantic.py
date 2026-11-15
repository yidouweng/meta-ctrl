"""Meta-token sequences for action sequencing.

A plan becomes  [ACTION, ID, ..., ACTION, ID, ..., EOS]: each action name
followed by its objects as per-prompt canonical ids (ID0, ID1, ...). Parsing
is prefix-tolerant: a truncated trailing action is dropped.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


def normalize_prop(prop: str) -> str:
    """Minimal, explicit normalization for property names."""
    p = str(prop).strip().upper()
    # Known typo in VirtualHome prompt template: CUTABLE vs CUTTABLE.
    if p == "CUTABLE":
        return "CUTTABLE"
    return p


@dataclass(frozen=True)
class ActionSchema:
    name: str
    arity: int
    # For each object argument position, a list of properties (OR semantics).
    role_reqs: Tuple[Tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.arity < 0:
            raise ValueError("arity must be >= 0")
        if self.arity != len(self.role_reqs):
            raise ValueError(f"arity={self.arity} but role_reqs has len={len(self.role_reqs)} for {self.name}")


def parse_virtualhome_supported_actions_from_text(text: str) -> Dict[str, ActionSchema]:
    """
    Parse "Supported Actions List" blocks of the form:
      ACTION: (arity, [[...],[...]] ) # comment
    """
    lines = str(text).splitlines()
    start_idx: Optional[int] = None
    for i, l in enumerate(lines):
        if l.strip() == "Supported Actions List:":
            start_idx = i + 1
            break
    if start_idx is None:
        return {}

    schemas: Dict[str, ActionSchema] = {}
    for l in lines[start_idx:]:
        s = l.strip()
        if not s:
            if schemas:
                break
            continue
        if s.startswith("Notice:"):
            break
        m = re.match(r"^([A-Z][A-Z0-9_]+)\s*:\s*(\(.*\))\s*(#.*)?$", s)
        if not m:
            if schemas:
                break
            continue
        act = m.group(1).strip()
        tup_str = m.group(2).strip()
        try:
            arity, reqs = ast.literal_eval(tup_str)
        except Exception as e:
            raise ValueError(f"Failed parsing Supported Actions tuple for {act}: {tup_str}") from e
        arity_i = int(arity)
        reqs_by_role: List[Tuple[str, ...]] = []
        for role in (reqs or []):
            reqs_by_role.append(tuple(normalize_prop(p) for p in (role or []) if str(p).strip()))
        schemas[act] = ActionSchema(name=act, arity=arity_i, role_reqs=tuple(reqs_by_role))
    return schemas


# -----------------------------------------------------------------------------
# Behavior: parsing prompts/templates + prompt-only property inference
# -----------------------------------------------------------------------------

_BEH_SECTION_HEADERS = {
    "initial environment state:",
    "target environment state:",
    "interactable objects:",
}


def _collect_block_after(lines: List[str], start_idx: int, header: str) -> List[str]:
    i: Optional[int] = None
    for j in range(start_idx, len(lines)):
        if lines[j].strip() == header:
            i = j + 1
            break
    if i is None:
        return []
    out: List[str] = []
    for k in range(i, len(lines)):
        s = lines[k].strip()
        if s == "":
            if out:
                break
            continue
        if s in _BEH_SECTION_HEADERS and s != header:
            break
        if s.startswith("Please output") or s.startswith("Output:"):
            break
        out.append(s)
    return out


def parse_behavior_prompt(prompt: str) -> Dict[str, object]:
    """
    Returns:
      {
        "objects": [{"name": ..., "category": ...}, ...],
        "initial_states": [python-list, ...],
        "target_states": [python-list, ...],
        "raw": prompt
      }
    """
    lines = str(prompt).splitlines()
    idxs = [i for i, l in enumerate(lines) if l.strip() == "Input:"]
    start = idxs[-1] if idxs else 0

    init_lines = _collect_block_after(lines, start, "initial environment state:")
    tar_lines = _collect_block_after(lines, start, "target environment state:")
    obj_lines = _collect_block_after(lines, start, "interactable objects:")

    initial_states: List[list] = []
    target_states: List[list] = []
    for l in init_lines:
        try:
            v = ast.literal_eval(l)
            if isinstance(v, list):
                initial_states.append(v)
        except Exception:
            pass
    for l in tar_lines:
        try:
            v = ast.literal_eval(l)
            if isinstance(v, list):
                target_states.append(v)
        except Exception:
            pass

    objects: List[dict] = []
    for l in obj_lines:
        try:
            d = ast.literal_eval(l)
        except Exception:
            continue
        if isinstance(d, dict) and "name" in d and "category" in d:
            objects.append({"name": str(d["name"]), "category": str(d["category"])})

    return {"objects": objects, "initial_states": initial_states, "target_states": target_states, "raw": prompt}


def parse_behavior_actions_from_text(text: str) -> Dict[str, int]:
    """
    Parse the Behavior AS action list under:
      "The action_name must be one of the following:"
    """
    lines = str(text).splitlines()
    start_idx: Optional[int] = None
    for i, l in enumerate(lines):
        if l.strip() == "The action_name must be one of the following:":
            start_idx = i + 1
            break
    if start_idx is None:
        return {}

    actions: Dict[str, int] = {}
    for l in lines[start_idx:]:
        s = l.strip()
        if not s:
            if actions:
                break
            continue
        if s.startswith("Format of the interactable objects:"):
            break
        # Format: ACTION # ...
        act = s.split()[0].strip()
        if not re.match(r"^[A-Z][A-Z0-9_]*$", act):
            continue

        # Try to infer arity from the example object string.
        arity = 1
        m = re.search(r"'object'\s*:\s*'([^']+)'", s)
        if m:
            obj_str = m.group(1)
            if "," in obj_str:
                arity = 2
        else:
            if "NEXTTO_ONTOP" in act:
                arity = 2
        actions[act] = arity
    return actions


# -----------------------------------------------------------------------------
# PDDL parsing helpers (for full-domain-schema signature harvesting)
# -----------------------------------------------------------------------------


_BEH_ACTION_DICT_RE = re.compile(
    r"\{\s*\"action\"\s*:\s*\"(?P<action>[A-Z0-9_]+)\"\s*,\s*\"object\"\s*:\s*\"(?P<object>[^\"]*)\"\s*\}"
)

# VirtualHome output pairs in-order: "ACTION": [ ... ]
_VH_ACTION_PAIR_RE = re.compile(
    r"\"(?P<action>[A-Z0-9_]+)\"\s*:\s*(?P<array>\[[^\]]*\])",
    flags=re.DOTALL,
)

# VirtualHome prompt object lines:
#   obj_name, id: 123, properties: [...]
_VH_OBJ_LINE_RE = re.compile(r"^([^,]+),\s*id\s*:\s*(\d+),\s*properties\s*:\s*(\[.*\])\s*$")


def canonical_id_map_from_behavior_prompt(prompt_text: str, *, id_cap: Optional[int] = None) -> Dict[str, str]:
    """
    Map each interactable instance name in a BEHAVIOR AS prompt to a canonical IDk.

    NOTE: This uses only the prompt's "interactable objects" list (matches evaluator guidance).
    """
    parsed = parse_behavior_prompt(prompt_text)
    objs = parsed.get("objects", []) or []
    names = [str(d.get("name", "")).strip() for d in objs if str(d.get("name", "")).strip()]

    if id_cap is not None and len(names) > int(id_cap):
        raise ValueError(f"Prompt has {len(names)} interactable objects, exceeds id_cap={int(id_cap)}")

    return {name: f"ID{i}" for i, name in enumerate(names)}


def _sorted_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    def _key(x):
        name, oid = x
        if str(oid).isdigit():
            return (str(name), int(oid))
        return (str(name), str(oid))

    return sorted(pairs, key=_key)


def parse_virtualhome_objects_from_prompt(prompt_text: str, *, exclude_character: bool = True) -> List[Tuple[str, str, Set[str]]]:
    """
    Parse VirtualHome prompt "Objects in the scene" section.

    Returns list of (name, original_id_str, properties_set).
    """
    lines = str(prompt_text).splitlines()
    try:
        start = lines.index("Objects in the scene:") + 1
    except ValueError:
        return []

    out: List[Tuple[str, str, Set[str]]] = []
    for s in lines[start:]:
        s = s.strip()
        if not s:
            continue
        if s.startswith("-----------------"):
            break
        m = _VH_OBJ_LINE_RE.match(s)
        if not m:
            continue
        name = m.group(1).strip()
        if exclude_character and name.lower() == "character":
            continue
        oid = m.group(2).strip()
        try:
            props_list = ast.literal_eval(m.group(3))
        except Exception:
            props_list = []
        props = {normalize_prop(p) for p in (props_list or []) if str(p).strip()}
        out.append((name, oid, props))
    return out


def canonical_id_map_from_virtualhome_prompt(prompt_text: str) -> Tuple[Dict[str, str], Dict[Tuple[str, str], Set[str]]]:
    """
    Canonicalize VirtualHome object instance IDs into per-prompt IDk.

    Returns:
      - orig_id_to_cid: {"1001": "ID0", ...}
      - props_by_instance: {(name, "1001"): {"GRABBABLE", ...}, ...}
    """
    objs = parse_virtualhome_objects_from_prompt(prompt_text)
    pairs = _sorted_pairs([(n, oid) for (n, oid, _ps) in objs])
    orig_id_to_cid = {oid: f"ID{i}" for i, (_n, oid) in enumerate(pairs)}
    props_by_instance = {(n, oid): set(ps) for (n, oid, ps) in objs}
    return orig_id_to_cid, props_by_instance


def parse_virtualhome_json_pairs_prefix(output_text: str) -> List[Tuple[str, list]]:
    """
    Extract complete (action, array_literal) pairs from a VirtualHome output string.

    Robust to truncation: if an array literal can't be parsed, stop and keep the prefix.
    """
    out: List[Tuple[str, list]] = []
    for m in _VH_ACTION_PAIR_RE.finditer(str(output_text)):
        act = str(m.group("action")).strip().upper()
        arr_s = str(m.group("array")).strip()
        try:
            arr = ast.literal_eval(arr_s)
        except Exception:
            break
        if not isinstance(arr, list):
            break
        out.append((act, arr))
    return out


def virtualhome_pairs_to_semantic_tokens(pairs: Sequence[Tuple[str, list]], *, prompt_text: str) -> List[str]:
    """Meta-tokens of parsed VirtualHome (action, [name, id, ...]) pairs, ending in EOS.
    Stops at the first unknown action, wrong argument count or unknown object id."""
    arity = {k: int(v.arity) for k, v in parse_virtualhome_supported_actions_from_text(prompt_text).items()}
    orig_id_to_cid, _ = canonical_id_map_from_virtualhome_prompt(prompt_text)
    y: List[str] = []
    for act, arr in pairs:
        act = str(act).upper()
        ar = int(arity.get(act, -1))
        if ar < 0 or len(arr) != (0 if ar == 0 else (2 if ar == 1 else 4)):
            break
        y.append(act)
        for role_i in range(ar):
            cid = orig_id_to_cid.get(str(arr[2 * role_i + 1]).strip())
            if cid is None:
                return y + ["EOS"]
            y.append(cid)
    return y + ["EOS"]


def parse_behavior_action_list_prefix(output_text: str) -> List[Tuple[str, str]]:
    """Complete (action, object string) pairs of a BEHAVIOR answer; a truncated
    trailing dict does not match."""
    return [(str(m.group("action")).strip().upper(), str(m.group("object")))
            for m in _BEH_ACTION_DICT_RE.finditer(str(output_text))]


def behavior_output_text_to_semantic_tokens(*, prompt_text: str, output_text: str) -> List[str]:
    """Meta-tokens of a BEHAVIOR answer, ending in EOS. Stops at the first action
    with the wrong number of objects or an object not in the prompt."""
    try:
        obj2cid = canonical_id_map_from_behavior_prompt(prompt_text)
    except Exception:
        return ["EOS"]
    arity = parse_behavior_actions_from_text(prompt_text)
    y: List[str] = []
    for act, obj_str in parse_behavior_action_list_prefix(output_text):
        objs = [o for o in (o.strip() for o in str(obj_str).split(",")) if o]
        if len(objs) != int(arity.get(act, 1)) or any(o not in obj2cid for o in objs):
            break
        y.append(act)
        y.extend(obj2cid[o] for o in objs)
    return y + ["EOS"]

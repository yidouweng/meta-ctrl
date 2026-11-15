"""
Prompt parsing for top-down DFA compilation.

This module parses ONLY object information from prompts.
Actions are NOT parsed - they come from hardcoded canonical lists in action_defs.py.

Object parsing extracts:
- VirtualHome: object names, IDs, and properties (for affordance filtering)
- BEHAVIOR: object names (instance names with IDs) and predicates (for affordance filtering)
"""

import ast
import re
from typing import Dict, List, Set, Tuple


# =============================================================================
# VirtualHome Prompt Parsing
# =============================================================================

def parse_vh_objects(prompt: str) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Parse VirtualHome prompt for objects only (no action parsing).

    Extracts from "Objects in the scene:" section:
    - Object names and their IDs
    - Object properties (for affordance filtering)

    Args:
        prompt: Full VH prompt text

    Returns:
        objects: {name: [id1, id2, ...]}
        properties: {name: {PROP1, PROP2, ...}}
    """
    text = str(prompt or "")
    objects: Dict[str, List[str]] = {}
    properties: Dict[str, Set[str]] = {}

    # Parse "Objects in the scene:" section
    m = re.search(r"Objects in the scene:\n(.*?)\n[-]{3,}", text, re.DOTALL)
    if not m:
        # Fallback: try without the dashes
        m = re.search(r"Objects in the scene:\n(.*?)(?:\n\n|\Z)", text, re.DOTALL)

    if m:
        block = m.group(1)
        for raw in block.strip().split('\n'):
            line = raw.strip()
            if not line:
                continue

            name = None
            obj_id = None
            props = None

            try:
                # Format: name, id: 1001, properties: [PROP1, PROP2, ...]
                if ', id: ' in line:
                    name_part, rest = line.split(', id: ', 1)
                    name = name_part.strip()

                    if ', properties:' in rest:
                        obj_id, rest2 = rest.split(', properties:', 1)
                        obj_id = obj_id.strip()
                        # Extract [...]
                        mprops = re.search(r"\[(.*?)\]", rest2)
                        if mprops:
                            props_str = mprops.group(1)
                            props = [p.strip().strip("'\"") for p in props_str.split(',') if p.strip()]
                        else:
                            props = []
                    else:
                        obj_id = rest.strip().split(',')[0].strip()
                        props = []
                else:
                    # Fallback: simple name, id pattern
                    parts = line.split(',')
                    name = parts[0].strip()
                    for part in parts[1:]:
                        if 'id:' in part:
                            obj_id = part.split(':', 1)[1].strip()
                            break
            except Exception:
                pass

            if name is not None and obj_id is not None:
                # Skip character - it's the agent, not an object in the scene
                if name.lower() == 'character':
                    continue
                objects.setdefault(name, []).append(obj_id)
                if props:
                    properties.setdefault(name, set()).update(props)

    # Try to enrich properties from "Nodes:" section if available
    m2 = re.search(r"Nodes:\n(.*?)\n[-]{3,}", text, re.DOTALL)
    if m2:
        nodes_block = m2.group(1)
        for raw in nodes_block.strip().split('\n'):
            line = raw.strip()
            if not line:
                continue
            try:
                # Format: name, states: [...], properties:[...]
                name_part = line.split(', states:')[0]
                nname = name_part.strip()
                mprops = re.search(r"properties\s*:\s*\[(.*?)\]", line)
                if mprops:
                    props_str = mprops.group(1)
                    props = [p.strip().strip("'\"") for p in props_str.split(',') if p.strip()]
                    if props:
                        properties.setdefault(nname, set()).update(props)
            except Exception:
                continue

    return objects, properties


# =============================================================================
# BEHAVIOR Prompt Parsing
# =============================================================================

def parse_beh_objects(prompt: str) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Parse BEHAVIOR prompt for objects only (no action parsing).

    Extracts from "interactable objects:" section:
    - Object instance names (e.g., "plywood_0", "carton_66")

    Also extracts predicates from init/target state blocks for affordance filtering.

    Args:
        prompt: Full BEHAVIOR prompt text

    Returns:
        objects: {base_name: [id1, id2, ...]}
            where base_name is the name without trailing _<digits>
            and ids are the full instance names
        predicates: {base_name: {pred1, pred2, ...}}
            predicates that mention this object (for affordance filtering)
    """
    text = str(prompt or "")
    objects: Dict[str, List[str]] = {}
    predicates: Dict[str, Set[str]] = {}

    # Parse "interactable objects:" section (pick LAST block to avoid example section)
    blocks = re.findall(
        r"^\s*interactable objects\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    block = blocks[-1] if blocks else ""

    # Extract instance names from JSON-like format
    # Format: {'name': 'plywood_0', ...} or {"name": "plywood_0", ...}
    instance_names = re.findall(
        r"\{\s*['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]\s*,",
        block,
    )

    # Build objects dict: base_name -> [instance_name, ...]
    for inst_name in sorted(set(instance_names)):
        base_name = _get_base_name(inst_name)
        objects.setdefault(base_name, []).append(inst_name)

    # Parse predicates from init/target state blocks
    predicates = _parse_beh_predicates(text)

    return objects, predicates


def _get_base_name(instance_name: str) -> str:
    """Extract base name from instance name (remove trailing _<digits>)."""
    s = str(instance_name).strip()
    m = re.match(r"^(.*)_(\d+)$", s)
    return m.group(1) if m else s


def _parse_beh_predicates(text: str) -> Dict[str, Set[str]]:
    """
    Parse predicates from BEHAVIOR init/target state blocks.

    Returns:
        {base_name: {pred1, pred2, ...}}
        where predicates are those that mention the object (for affordance filtering)
    """
    predicates: Dict[str, Set[str]] = {}

    # Parse init/target state blocks (pick LAST blocks to avoid example section)
    init_blocks = re.findall(
        r"^\s*initial environment state\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    tgt_blocks = re.findall(
        r"^\s*target environment state\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )

    init_block = init_blocks[-1] if init_blocks else ""
    tgt_block = tgt_blocks[-1] if tgt_blocks else ""

    def parse_list_lines(block: str) -> List[List[str]]:
        out = []
        for raw in str(block).splitlines():
            ln = raw.strip()
            if not ln.startswith("["):
                continue
            try:
                v = ast.literal_eval(ln)
            except Exception:
                continue
            if isinstance(v, list):
                out.append([str(x) for x in v])
        return out

    init_facts = parse_list_lines(init_block)
    tgt_facts = parse_list_lines(tgt_block)

    def add_predicate(pred: str, obj: str):
        pred = str(pred).strip().lower()
        base = _get_base_name(obj)
        predicates.setdefault(base, set()).add(pred)

    for lst in (init_facts + tgt_facts):
        if not lst:
            continue
        # Format: ['not', 'open', OBJ] or ['open', OBJ]
        if str(lst[0]).lower() == "not" and len(lst) >= 3:
            add_predicate(lst[1], lst[2])
        elif len(lst) == 2:
            add_predicate(lst[0], lst[1])

    return predicates

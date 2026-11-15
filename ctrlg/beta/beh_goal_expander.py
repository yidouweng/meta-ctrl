"""Goal expansion for BEHAVIOR AS prompts.

Handles simple quantifier patterns so downstream β code only sees concrete
simple predicates:

  [forall, CAT, -, CAT, pred, CAT, Y]
    → one [pred, x_i, Y] per instance x_i of category CAT in the scene

  [not, prop, X]
    → [not_<prop>, X]  (negated property goal)

  [forall, CAT, -, CAT, not, prop, CAT]
    → one [not_<prop>, x_i] per instance of CAT

  [forall, CAT, -, CAT, and, ...]
    → each simple conjunct expanded per instance

  [exists, CAT, -, CAT, ...]
    → the variable is anchored on the first instance of CAT

Unsupported forms (forpairs, exists-forall, deeply nested and/or) pass through
unchanged and are skipped by the β parser.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


LOCATION_PREDICATES = {"onfloor", "ontop", "inside", "nextto", "under"}
PROPERTY_PREDICATES = {"toggled_on", "sliced", "soaked", "frozen", "cooked",
                       "stained", "dusty", "open"}


def build_category_map(objects: List[Dict]) -> Dict[str, List[str]]:
    """Map wordnet category → [instance names] from the prompt's object list."""
    cat2inst: Dict[str, List[str]] = defaultdict(list)
    for o in objects:
        cat = o.get("category", "")
        name = o.get("name", "")
        if cat and name:
            cat2inst[cat].append(name)
    return dict(cat2inst)


def expand_goals(target_states: List[List], objects: List[Dict]) -> List[List]:
    """Expand simple quantifier patterns into concrete predicates.

    Returns a new list. Non-expanded predicates pass through so the β's existing
    parser handles them.
    """
    cat2inst = build_category_map(objects)
    expanded: List[List] = []

    for t in target_states:
        if not t or not isinstance(t, list):
            continue
        head = t[0]

        # Simple passthrough forms.
        if head in LOCATION_PREDICATES or head in PROPERTY_PREDICATES:
            expanded.append(t)
            continue

        # [not, pred, obj]  — negated property on concrete object.
        if head == "not" and len(t) == 3 and t[1] in PROPERTY_PREDICATES:
            expanded.append(["not_" + t[1], t[2]])
            continue

        # forall CAT - CAT body…
        if head == "forall" and len(t) >= 5 and t[2] == "-":
            cat = t[1]
            body = t[4:]
            insts = cat2inst.get(cat, [])
            if not insts:
                expanded.append(t)  # no instances — leave for β to skip
                continue

            # body = [pred, CAT, Y]  → one pred(x_i, Y) per instance
            if len(body) >= 3 and body[0] in LOCATION_PREDICATES:
                pred, y = body[0], body[2]
                # y may itself be a category name; use its first instance
                y_targets = cat2inst.get(y, [y])
                y0 = y_targets[0] if y_targets else y
                for inst in insts:
                    expanded.append([pred, inst, y0])
                continue

            # body = [not, prop, CAT]  → one not_<prop>(x_i)
            if len(body) >= 3 and body[0] == "not" and body[1] in PROPERTY_PREDICATES:
                pred = body[1]
                for inst in insts:
                    expanded.append(["not_" + pred, inst])
                continue

            # body = [prop, CAT]  → one prop(x_i)
            if len(body) >= 2 and body[0] in PROPERTY_PREDICATES:
                pred = body[0]
                for inst in insts:
                    expanded.append([pred, inst])
                continue

            # body = [and, pred1..., pred2...]  → split each conjunct and expand
            # Very common: forall CAT - CAT and inside CAT Y sliced CAT
            if body and body[0] == "and":
                # Walk sub-predicates greedily: each sub is either
                # [pred, var, target] (loc) or [not/prop, var] or [prop, var]
                i = 1
                subs: List[List] = []
                while i < len(body):
                    token = body[i]
                    if token in LOCATION_PREDICATES and i + 2 < len(body):
                        subs.append([token, body[i + 1], body[i + 2]])
                        i += 3
                    elif token == "not" and i + 2 < len(body) and body[i + 1] in PROPERTY_PREDICATES:
                        subs.append(["not", body[i + 1], body[i + 2]])
                        i += 3
                    elif token in PROPERTY_PREDICATES and i + 1 < len(body):
                        subs.append([token, body[i + 1]])
                        i += 2
                    else:
                        subs = []  # give up — unrecognised
                        break
                if subs:
                    for inst in insts:
                        for s in subs:
                            if s[0] in LOCATION_PREDICATES:
                                y = s[2]
                                y_list = cat2inst.get(y, [y])
                                expanded.append([s[0], inst, y_list[0] if y_list else y])
                            elif s[0] == "not":
                                expanded.append(["not_" + s[1], inst])
                            elif s[0] in PROPERTY_PREDICATES:
                                expanded.append([s[0], inst])
                    continue

        # exists CAT - CAT body  → anchor on first instance of CAT
        if head == "exists" and len(t) >= 5 and t[2] == "-":
            cat = t[1]
            body = t[4:]
            insts = cat2inst.get(cat, [])
            if not insts:
                expanded.append(t)
                continue
            anchor = insts[0]

            if len(body) >= 3 and body[0] in LOCATION_PREDICATES:
                pred, x, y = body[0], body[1], body[2]
                # x may be a concrete name, y may be the category variable
                if isinstance(x, str) and x not in cat2inst:
                    expanded.append([pred, x, anchor])
                    continue
                if isinstance(y, str) and y not in cat2inst:
                    expanded.append([pred, anchor, y])
                    continue

            # exists C: and (pred1, pred2, ...) — anchor the variable
            if body and body[0] == "and":
                # Collect simple sub-predicates
                i = 1
                while i < len(body):
                    token = body[i]
                    if token in LOCATION_PREDICATES and i + 2 < len(body):
                        x, y = body[i + 1], body[i + 2]
                        # Substitute: if x == cat (the variable), use anchor; if y == cat, use anchor.
                        xr = anchor if x == cat else x
                        yr = anchor if y == cat else y
                        if isinstance(xr, str) and isinstance(yr, str):
                            expanded.append([token, xr, yr])
                        i += 3
                    elif token == "not" and i + 2 < len(body):
                        # not pred x — may be location
                        sub = body[i + 1]
                        if sub in PROPERTY_PREDICATES:
                            x = body[i + 2]
                            xr = anchor if x == cat else x
                            expanded.append(["not_" + sub, xr])
                            i += 3
                        elif sub in LOCATION_PREDICATES and i + 3 < len(body):
                            i += 4  # negated location goals are skipped
                        else:
                            i += 1
                    elif token in PROPERTY_PREDICATES and i + 1 < len(body):
                        x = body[i + 1]
                        xr = anchor if x == cat else x
                        expanded.append([token, xr])
                        i += 2
                    else:
                        i += 1
                continue

        # Fall-through: unrecognised form, pass through so β skips it.
        expanded.append(t)

    return expanded



"""Token-level acceptance test for the AS training corpus.

A sampled answer enters the AS corpus only if, from its first "[" token
(BEHAVIOR) or "{" token (VirtualHome), a prefix of its tokens spells a
well-formed action list over the prompt's actions and objects:

    BEHAVIOR     [{"action": "LEFT_GRASP", "object": "plate_1"}, ...]
    VirtualHome  {"WALK": ["kitchen", "205"], "PUTIN": ["cup", "1001", "sink", "247"], ...}

`first_accept(...)` returns the index of the token that closes the list, or
None.

Why tokens, not text
--------------------
The test reads the sampled token ids, not the decoded string. Every
*atom* (action name, the keys "action"/"object", object name, object id)
has to be sampled as its canonical tokenization, tokenizer.encode(atom),
with token boundaries at both ends. Text that is well formed but tokenized
differently is rejected. With the Llama 3 tokenizer this drops, e.g.,
  - RIGHT_PLACE_ONTOP sampled as 'RIGHT' '_PLACE' '_ON' 'TOP' (canonical:
    'RIGHT' '_PLACE' '_' 'ONT' 'OP');
  - "a_1, coffee_table_2" sampled as ... ',' ' coffee' ... (the separator
    must be ', ' or ',' ' ', see below);
  - unindented VirtualHome JSON, whose keys come out as '"W' 'ALK'.
A validator on the decoded string cannot see this; on the first 50k
samples per benchmark it would keep 3-4% more answers.

Rules
-----
Tokens are split into *structural* tokens (only JSON whitespace " \\t\\n\\r"
and the characters {}[]:,") and *content* tokens (everything else).

* Structural tokens are matched by their skeleton, the token with its
  whitespace removed, so merges like '",\\n' or ' ["' are accepted where
  their characters are. A skeleton listed in `_SKELETONS` may carry
  whitespace anywhere, except that
    - the skeletons '"', '{"' and '["' must not end in whitespace
      (whitespace after an opening quote would be inside the string), but
      may have whitespace before the quote (so '"WALK "' is read as WALK);
    - the first token must start with the opening bracket ("[0", "{0");
    - '"~' (BEHAVIOR, after an action or the object string) is a quote
      with whitespace on either side;
    - "]_" (an empty VirtualHome argument list) needs whitespace before "]".
  Which merges are allowed differs per position and per benchmark, exactly
  as written in `_behavior` and `_virtualhome`; e.g. '", "' is accepted
  between the action and "object" in BEHAVIOR but not between a name and
  an id in VirtualHome.
* Between items, up to 4 whitespace-only tokens in a row are skipped
  (`_Cursor.ws`); there is no whitespace inside atoms.
* An atom is the maximal run of content tokens at that position; the run
  must equal the canonical tokenization of an allowed value. A BEHAVIOR
  object instance "<base>_<id>" is tokenized as encode("<base>_") followed
  by encode("<id>"); the base is the longest run prefix that is a prefix of
  some encode("<base>_") (greedy, as a trie walk), the rest must be the id.
  Two objects in one BEHAVIOR object string are separated by the tokens
  ", " or "," or "," " ".
* Prompt-derived vocabularies:
    BEHAVIOR     actions of "The action_name must be one of the following:"
                 that are in ctrlg.schemas.beh_as.ACTIONS, with the arity
                 read from the example (training/semantic.py); objects are
                 the interactable "<base>_<digits>" instances. No affordances.
    VirtualHome  actions of "Supported Actions List:" minus PLUGIN, PLUGOUT,
                 RELEASE, with PUTIN's second argument also requiring
                 CONTAINERS; objects of "Objects in the scene:" except
                 character, with their ids; argument r of an action must be
                 an object whose properties include all of that role's
                 requirements (the prompts' misspelled requirement CUTABLE
                 is read as the objects' CUTTABLE, as in semantic.py). An action
                 without arguments must be written with whitespace in the
                 list, "[ ]". Object and action lines are read in the
                 prompts' format ("sink, id: 247, properties: [...]",
                 "PUTIN: (2, [[...], [...]])"), which all 338 prompts use.
"""
import ast
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ctrlg.schemas import beh_as

from semantic import normalize_prop, parse_behavior_actions_from_text, parse_behavior_prompt

_WS = " \t\n\r"
_PUNCT = set('{}[]:,"')
_SKELETONS = {"{", "}", "[", "]", ":", ",", '"', '{"', '["', '":', '":"', '",', '","',
              '"}', '"},', '},', '"]', '"],', '"]}'}
_MAX_WS_RUN = 4


# ---------------------------------------------------------------- token classes

_CLASS_CACHE: Dict[int, List[Optional[frozenset]]] = {}


def _token_texts(tokenizer) -> List[str]:
    """Decoded text of every token id, leading whitespace kept (decoded after a
    special token so the tokenizer does not strip it)."""
    sp_ids = list(tokenizer.all_special_ids or [])
    sp = int(sp_ids[0]) if sp_ids else int(tokenizer.eos_token_id or 0)
    n = max(int(tokenizer.vocab_size or 0), int(tokenizer.eos_token_id) + 1, max(sp_ids, default=-1) + 1, 1)
    pref = tokenizer.decode([sp])
    texts = []
    for tid in range(n):
        s = tokenizer.decode([sp, tid])
        texts.append(s[len(pref):] if s.startswith(pref) else tokenizer.decode([tid]))
    return texts


def _classes(text: str) -> Optional[frozenset]:
    """Class names of a structural token (see the module docstring), or None
    for a content token."""
    if not text or any(c not in _WS and c not in _PUNCT for c in text):
        return None
    skel = "".join(c for c in text if c not in _WS)
    lead, trail = text[0] in _WS, text[-1] in _WS
    if skel == "":
        return frozenset({"ws"})
    out = set()
    if skel in _SKELETONS and not (skel in ('"', '{"', '["') and trail):
        out.add(skel)
    if skel in ("[", "{", '{"') and not lead and not (skel == '{"' and trail):
        out.add(skel + "0")
    if skel == '"':
        out.add('"~')
    if skel == "]" and lead:
        out.add("]_")
    return frozenset(out)


def token_classes(tokenizer) -> List[Optional[frozenset]]:
    key = id(tokenizer)
    if key not in _CLASS_CACHE:
        _CLASS_CACHE[key] = [_classes(t) for t in _token_texts(tokenizer)]
    return _CLASS_CACHE[key]


class _Reject(Exception):
    pass


class _Cursor:
    def __init__(self, toks: Sequence[int], classes: List[Optional[frozenset]]):
        self.toks, self.classes, self.i = toks, classes, 0

    def _cls(self, i: int) -> Optional[frozenset]:
        t = self.toks[i]
        return self.classes[t] if 0 <= t < len(self.classes) else None

    def take(self, *names: str) -> str:
        """Consume one structural token of one of the given classes; return the class."""
        if self.i >= len(self.toks):
            raise _Reject
        cls = self._cls(self.i) or frozenset()
        for name in names:
            if name in cls:
                self.i += 1
                return name
        raise _Reject

    def take_id(self, ids: Set[int]) -> bool:
        """Consume the next token if its id is in ids."""
        if self.i < len(self.toks) and self.toks[self.i] in ids:
            self.i += 1
            return True
        return False

    def ws(self) -> None:
        """Skip at most _MAX_WS_RUN whitespace-only tokens."""
        n = 0
        while self.i < len(self.toks) and "ws" in (self._cls(self.i) or ()):
            self.i += 1
            n += 1
        if n > _MAX_WS_RUN:
            raise _Reject

    def run(self) -> Tuple[int, ...]:
        """Consume the maximal run of content tokens."""
        j = self.i
        while j < len(self.toks) and self._cls(j) is None:
            j += 1
        out, self.i = tuple(self.toks[self.i:j]), j
        return out


def _encode(tokenizer, s: str) -> Tuple[int, ...]:
    return tuple(tokenizer.encode(s, add_special_tokens=False))


# ---------------------------------------------------------------- BEHAVIOR

class BehaviorSpec:
    def __init__(self, tokenizer, prompt_text: str):
        arity = parse_behavior_actions_from_text(prompt_text)
        self.actions = {_encode(tokenizer, a): (a, arity[a]) for a in arity if a in beh_as.ACTIONS}
        self.key_action, self.key_object = _encode(tokenizer, "action"), _encode(tokenizer, "object")
        self.bases: Dict[Tuple[int, ...], str] = {}      # encode("<base>_") -> base
        self.ids: Dict[str, Set[Tuple[int, ...]]] = {}  # base -> {encode("<id>")}
        for o in parse_behavior_prompt(prompt_text).get("objects", []) or []:
            base, sep, oid = str(o.get("name", "")).strip().rpartition("_")
            if sep and oid.isdigit():
                self.bases[_encode(tokenizer, base + "_")] = base
                self.ids.setdefault(base, set()).add(_encode(tokenizer, oid))
        self.base_prefixes = {seq[:k] for seq in self.bases for k in range(len(seq) + 1)}
        single = {s: _encode(tokenizer, s) for s in (", ", ",", " ")}
        self.comma_space, self.comma, self.space = ({seq[0]} if len(seq) == 1 else set() for seq in single.values())

    def instance_ok(self, run: Tuple[int, ...]) -> bool:
        k = 0
        while k < len(run) and run[:k + 1] in self.base_prefixes:
            k += 1
        base = self.bases.get(run[:k])
        return base is not None and run[k:] in self.ids.get(base, ())


def _behavior(c: _Cursor, spec: BehaviorSpec) -> int:
    c.take("[0")
    while True:
        c.ws()                                                    # a list item
        if c.take('{"', "{") == "{":
            c.ws(); c.take('"')
        if c.run() != spec.key_action:
            raise _Reject
        c.ws()
        sep = c.take('":"', '":', '"')
        if sep == '"':
            c.ws(); c.take(":")
        if sep != '":"':
            c.ws(); c.take('"')
        action = spec.actions.get(c.run())
        if action is None:
            raise _Reject
        sep = c.take('","', '",', '"~')
        if sep == '"~':
            c.ws(); c.take(",")
        if sep != '","':
            c.ws(); c.take('"')
        if c.run() != spec.key_object:
            raise _Reject
        c.ws()
        sep = c.take('":"', '":', '"')
        if sep == '"':
            c.ws(); c.take(":")
        if sep != '":"':
            c.ws(); c.take('"')
        for role in range(action[1]):
            if not spec.instance_ok(c.run()):
                raise _Reject
            if role + 1 < action[1]:                              # "a_1, b_2" / "a_1,b_2" / "a_1," " b_2"
                if not c.take_id(spec.comma_space):
                    if not c.take_id(spec.comma):
                        raise _Reject
                    c.take_id(spec.space)
        close = c.take('"},', '"}', '"~')
        if close == '"~':
            c.ws()
            close = '"' + c.take("},", "}")
        if close == '"}':
            c.ws()                                                # between items
            if c.take(",", "]") == "]":
                return c.i - 1


# ---------------------------------------------------------------- VirtualHome

_VH_DISALLOWED = {"PLUGIN", "PLUGOUT", "RELEASE"}


def _vh_supported_actions(prompt_text: str) -> Tuple[Dict[str, int], Dict[str, List[List[str]]]]:
    """{ACTION: arity}, {ACTION: per-argument property requirements} of the
    "Supported Actions List:" lines "PUTIN: (2, [['GRABBABLE'], ['CAN_OPEN']]) # ..."."""
    m = re.search(r"Supported Actions List:\n(.*?)(?:\n\n|\nNotice:)", prompt_text, re.DOTALL)
    arity: Dict[str, int] = {}
    roles: Dict[str, List[List[str]]] = {}
    for line in (m.group(1).strip().split("\n") if m else []):
        name, colon, rhs = line.strip().partition(":")
        name = name.strip().upper()
        m2 = re.search(r"\((\d+)\s*,\s*(\[[\s\S]*\])\)", rhs) if colon else None
        if not m2 or name in _VH_DISALLOWED:
            continue
        try:
            req = ast.literal_eval(m2.group(2))
        except Exception:
            req = []
        arity[name] = int(m2.group(1))
        roles[name] = [[normalize_prop(x) for x in r] for r in req] if isinstance(req, list) else []
    if arity.get("PUTIN", 0) >= 2:
        rp = roles["PUTIN"] + [[]] * max(0, arity["PUTIN"] - len(roles["PUTIN"]))
        if "CONTAINERS" not in {r.strip().upper() for r in rp[1]}:
            rp[1] = list(rp[1]) + ["CONTAINERS"]
        roles["PUTIN"] = rp
    return arity, roles


_VH_OBJ_LINE_RE = re.compile(r"(.*?), id: (.*?), properties:.*?\[(.*?)\]")


def _vh_objects(prompt_text: str) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """{name: [ids]}, {name: properties} of the "Objects in the scene:" lines
    "sink, id: 247, properties: ['CONTAINERS', 'RECIPIENT']"."""
    m = re.search(r"Objects in the scene:\n(.*?)\n[-]{3,}", prompt_text, re.DOTALL)
    ids: Dict[str, List[str]] = {}
    props: Dict[str, Set[str]] = {}
    for line in (m.group(1).strip().split("\n") if m else []):
        mm = _VH_OBJ_LINE_RE.match(line.strip())
        if mm:
            name = mm.group(1).strip()
            ids.setdefault(name, []).append(mm.group(2).strip())
            props.setdefault(name, set()).update(p.strip().strip("'\"") for p in mm.group(3).split(",") if p.strip())
    return ids, props


class VirtualHomeSpec:
    def __init__(self, tokenizer, prompt_text: str):
        arity, roles = _vh_supported_actions(prompt_text)
        ids, props = _vh_objects(prompt_text)
        self.actions = {_encode(tokenizer, a): (a, ar) for a, ar in arity.items()}
        self.names = {_encode(tokenizer, n): n for n in ids if n != "character"}
        self.ids = {n: {_encode(tokenizer, i) for i in ids[n]} for n in self.names.values()}
        self.allowed: Dict[Tuple[str, int], Set[str]] = {}
        for a, ar in arity.items():
            for r in range(ar):
                req = {str(p).upper() for p in (roles[a][r] if r < len(roles[a]) else [])}
                self.allowed[(a, r)] = {n for n, ps in props.items() if req <= {str(p).upper() for p in ps}}


def _virtualhome(c: _Cursor, spec: VirtualHomeSpec) -> int:
    if c.take('{"0', "{0") == "{0":
        c.ws(); c.take('"')
    while True:
        action = spec.actions.get(c.run())                        # "ACTION": [...]
        if action is None:
            raise _Reject
        name_, arity = action
        if c.take('":', '"') == '"':
            c.ws(); c.take(":")
        c.ws()
        if arity == 0:                                            # "[ ]"
            c.take("[")
            if c.take("ws", "]_") == "ws":
                c.ws(); c.take("]")
            close = "]"
        else:
            if c.take("[", '["') == "[":
                c.ws(); c.take('"')
            for role in range(arity):
                name = spec.names.get(c.run())
                if name is None or name not in spec.allowed[(name_, role)]:
                    raise _Reject
                if c.take('",', '"') == '"':
                    c.ws(); c.take(",")
                c.ws(); c.take('"')
                if c.run() not in spec.ids[name]:
                    raise _Reject
                if role + 1 < arity:
                    if c.take('",', '"') == '"':
                        c.ws(); c.take(",")
                    c.ws(); c.take('"')
                else:
                    close = c.take('"],', '"]}', '"]', '"')
                    if close == '"':
                        c.ws(); c.take("]")
        if close == '"]}':
            return c.i - 1
        if close != '"],':
            c.ws()                                                # between pairs
            if c.take(",", "}") == "}":
                return c.i - 1
        c.ws(); c.take('"')


# ---------------------------------------------------------------- entry point

def build_spec(domain: str, tokenizer, prompt_text: str):
    return (VirtualHomeSpec if domain == "virtualhome" else BehaviorSpec)(tokenizer, prompt_text)


def first_accept(domain: str, tokenizer, spec, toks: Sequence[int]) -> Optional[Tuple[int, int]]:
    """(start, end): toks[start:end + 1] is the accepted action list, or None.

    start is the first token that is an opening bracket ("[" for BEHAVIOR,
    "{" or '{"' for VirtualHome) with optional whitespace around it, or 0
    if there is none; the list must start with that token's bracket."""
    classes = token_classes(tokenizer)
    toks = [int(t) for t in toks]
    opens = ("[",) if domain == "behavior" else ("{", '{"')
    start = next((i for i, t in enumerate(toks)
                  if 0 <= t < len(classes) and classes[t] and any(o in classes[t] for o in opens)), 0)
    cursor = _Cursor(toks[start:], classes)
    try:
        end = (_virtualhome if domain == "virtualhome" else _behavior)(cursor, spec)
    except _Reject:
        return None
    return start, start + end

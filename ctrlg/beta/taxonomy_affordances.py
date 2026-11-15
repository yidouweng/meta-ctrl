"""Taxonomy-based affordance provider backed by bddl hierarchy_all.json.

BEHAVIOR prompts supply the synset for every scene object
(``{'name': 'sink_7', 'category': 'sink.n.01'}``), so affordances are
looked up exactly in the bddl object taxonomy instead of guessed from name
keywords.

How the taxonomy is read:

* The file is one rooted tree (root ``entity.n.01``); every node has
  ``name``, ``igibson_categories``, ``abilities`` and, if internal,
  ``children``.
* A node's own ``abilities`` dict is authoritative (abilities are the
  intersection of the children's, not inherited downward), so a lookup reads
  the node itself.
* A synset can sit at several positions (WordNet is a DAG flattened into a
  tree); its abilities are the union over its positions, and ``is_a`` holds
  if any position has the ancestor on its root path (``is_a(x, x)`` is true).
* Useful ancestor classes include furniture.n.01, appliance.n.02,
  fixture.n.01 / plumbing_fixture.n.01, container.n.01, surface.n.01,
  movable_barrier.n.01, framework.n.03, excavation.n.03, piece_of_cloth.n.01
  and brush.n.02. Agents, rooms, walls and ceilings are not in the tree and
  are handled by name in ``beh_affordances.py``.
* Only ability *keys* are exposed. ``waterSource`` appears only on
  ``sink.n.01``.

Synset normalization: AS prompts give dotted synsets (``sink.n.01``);
SD prompts give underscore form (``book_n_02``). Lookup is word-exact on
the dotted form; the sole normalization applied is
``<lemma>_<pos>_<NN>`` -> ``<lemma>.<pos>.<NN>`` (regex on the tail only,
so lemma-internal underscores like ``scrub_brush.n.01`` are preserved).
No keyword/substring fallback of any kind is performed.

Path resolution for the JSON (first hit wins):
1. explicit ``path`` argument to ``TaxonomyAffordances``;
2. the in-repo copy ``data/taxonomy/hierarchy_all.json`` (preferred so the
   generation environment does not need bddl installed);
3. the installed ``bddl`` package via ``importlib``.

The file is loaded and indexed ONCE per resolved path (module-level cache);
use ``get_default()`` for the shared singleton.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["TaxonomyAffordances", "get_default"]

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_REPO_COPY = Path(__file__).resolve().parents[2] / "data" / "taxonomy" / "hierarchy_all.json"


def _bddl_package_path() -> Optional[Path]:
    """Locate hierarchy_all.json inside an installed bddl package, if any."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("bddl")
        if spec is not None and spec.origin:
            cand = Path(spec.origin).parent / "hierarchy_all.json"
            if cand.is_file():
                return cand
    except Exception:
        pass
    return None


def _resolve_path(path: Optional[str]) -> Path:
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"hierarchy_all.json not found at explicit path: {p}")
        return p
    if _REPO_COPY.is_file():
        return _REPO_COPY
    pkg = _bddl_package_path()
    if pkg is not None:
        return pkg
    raise FileNotFoundError(
        "hierarchy_all.json not found. Looked for the repo copy at "
        f"{_REPO_COPY} and in the installed bddl package. Pass an explicit "
        "path=... argument."
    )


# ---------------------------------------------------------------------------
# Index construction (module-level cache: load & index ONCE per path)
# ---------------------------------------------------------------------------

# path -> index
_CACHE: Dict[Path, Any] = {}

_UNDERSCORE_SYNSET_RE = re.compile(r"^(?P<lemma>.+)_(?P<pos>[a-z])_(?P<num>\d{2})$")


def _normalize_synset(synset: str) -> str:
    """Normalize prompt category strings to dotted synset form.

    ``book_n_02`` -> ``book.n.02`` (SD prompt form); dotted forms pass
    through unchanged. Word-exact — no substring matching.
    """
    s = str(synset).strip()
    m = _UNDERSCORE_SYNSET_RE.match(s)
    if m and "." not in s:
        return f"{m.group('lemma')}.{m.group('pos')}.{m.group('num')}"
    return s


def _build_index(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        root = json.load(f)

    abilities: Dict[str, frozenset] = {}
    # ANY-occurrence ancestor sets: ancestors[name] = union over all tree
    # occurrences of the names on the root path, INCLUDING the name itself.
    # A synset can appear at multiple tree positions; is_a(x, a) must be
    # True iff ANY occurrence of x has `a` on its path, which is exactly
    # membership in this union.
    ancestors: Dict[str, set] = {}

    # DFS with explicit path (node, path_tuple).
    stack: List[tuple] = [(root, ())]
    while stack:
        node, path = stack.pop()
        name = node.get("name")
        ab = node.get("abilities") or {}
        if name is not None:
            keys = frozenset(ab.keys())
            if name in abilities:
                # A synset can sit at several tree positions; take the
                # union of their abilities.
                abilities[name] = abilities[name] | keys
            else:
                abilities[name] = keys
            ancestors.setdefault(name, set()).update(path)
            ancestors[name].add(name)
            child_path = path + (name,)
        else:
            child_path = path
        for ch in node.get("children") or []:
            stack.append((ch, child_path))

    return {
        "abilities": abilities,
        "ancestors": {k: frozenset(v) for k, v in ancestors.items()},
    }


def _load_index(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if path not in _CACHE:
        _CACHE[path] = _build_index(path)
    return _CACHE[path]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_EMPTY: frozenset = frozenset()


class TaxonomyAffordances:
    """Synset -> abilities lookup over the bddl taxonomy.

    Unknown synsets return the empty set.
    """

    def __init__(self, path: Optional[str] = None):
        self.source_path: Path = _resolve_path(path)
        idx = _load_index(self.source_path)
        self._abilities: Dict[str, frozenset] = idx["abilities"]
        self._ancestors: Dict[str, frozenset] = idx["ancestors"]

    # -- core -------------------------------------------------------------

    def abilities(self, synset: str) -> frozenset:
        """Ability keys for ``synset`` (word-exact; empty set if unknown)."""
        return self._abilities.get(_normalize_synset(synset), _EMPTY)

    def has(self, synset: str, ability: str) -> bool:
        return ability in self.abilities(synset)

    def known(self, synset: str) -> bool:
        """True iff the synset exists in the taxonomy."""
        return _normalize_synset(synset) in self._abilities

    # -- ISA-subtree queries ------------------------------------------------

    def ancestors(self, synset: str) -> frozenset:
        """All ancestor synsets of ``synset`` (ANY-occurrence union), plus
        the synset itself. Empty set for unknown synsets.
        """
        return self._ancestors.get(_normalize_synset(synset), _EMPTY)

    def is_a(self, synset: str, ancestor_synset: str) -> bool:
        """True iff ANY tree occurrence of ``synset`` has ``ancestor_synset``
        on its root path (reflexive: is_a(x, x) is True for known x).
        """
        return _normalize_synset(ancestor_synset) in self.ancestors(synset)

    def is_a_any(self, synset: str, ancestor_synsets) -> bool:
        """True iff ``synset`` IS-A at least one of ``ancestor_synsets``."""
        anc = self.ancestors(synset)
        return any(_normalize_synset(a) in anc for a in ancestor_synsets)

    # -- convenience helpers ------------------------------------------------

    def openable(self, synset: str) -> bool:
        return self.has(synset, "openable")

    def toggleable(self, synset: str) -> bool:
        return self.has(synset, "toggleable")

    def cleaning_tool(self, synset: str) -> bool:
        return self.has(synset, "cleaningTool")

    def water_source(self, synset: str) -> bool:
        return self.has(synset, "waterSource")


_DEFAULT: Optional[TaxonomyAffordances] = None


def get_default() -> TaxonomyAffordances:
    """Shared singleton using the default path resolution."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TaxonomyAffordances()
    return _DEFAULT

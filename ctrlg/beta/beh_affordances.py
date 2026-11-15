"""BEHAVIOR shared membership layer: the single source for every
affordance/class set that the BEHAVIOR γ automata and β automata consume,
so γ and β (AS and SD) always agree.

Every membership function is derived only from

  P — the prompt's object list ``[{'name': ..., 'category': <synset>}, ...]``
      (plus, where a rule is explicitly prompt-scoped, prompt-stated
      relations passed in by the caller, e.g. container roles), and
  T — the bddl taxonomy (``taxonomy_affordances.TaxonomyAffordances``:
      ability keys + ISA-subtree queries over hierarchy_all.json).

Membership definitions:

  openable(x)          T:openable (bddl's conceptual `openable` mark).
  open_gated(x)        openable(x) minus open-top vessels (bucket/basket/bin
                       subtrees have no lid). β allocates an open bit
                       (default closed unless the prompt's initial states
                       say `open(X)`) only for open_gated(x) ∧ x in
                       container role, where container role = arg-1 of an
                       `inside` relation in the prompt's initial or goal
                       states (callers pass the role set; it is
                       prompt-derived). plate/tray/pan/pot/bowl/sink/
                       bathtub/pool are not openable and are never gated.
  toggleable(x)        T:toggleable ∪ ISA bathtub.n.01 (a tap is turned on
                       and off).
  cleaner(x)           T:cleaningTool ∪ ISA piece_of_cloth.n.01 ∪
                       ISA brush.n.02 (the two parent classes
                       whose instances the prompts use as wipes/brushes).
  water_source(x)      T:waterSource ∪ (T:toggleable ∧ ISA
                       plumbing_fixture.n.01) ∪ ISA bathtub.n.01 ∪ passive
                       water sources. In the taxonomy bathtub.n.01 is a
                       vessel, not a plumbing fixture, so it is named
                       directly. Vessels the prompt states contain water
                       are prompt-scoped: callers OR them in via
                       ``prompt_water_vessels`` (see helper below).
  passive_water_source(x)
                       ISA teapot/kettle/pool (dispense or hold water with
                       no switch), excluding toggleable objects.
  non_graspable(x)     ISA classes (see _NON_GRASPABLE_CLASSES and the
                       rationale note) ∪ name-structural rules for room
                       floors / walls / ceilings / the agent (those concepts
                       have no usable tree class — agent.n.01/ceiling.n.01
                       do not exist in the tree).
  non_containable(x)   NOT (ISA container.n.01 ∪ ISA
                       plumbing_fixture.n.01 ∪ prompt-container-role).
                       "Things that can hold contents are hollow containers
                       or basin fixtures"; the prompt-role union keeps any
                       container the task itself asserts (P beats class
                       membership).
  slicer(x)            T:slicer (knife/cleaver family).

Class-set rationale (see the taxonomy_affordances.py docstring for which
classes exist in the tree):

  _NON_GRASPABLE_CLASSES =
    furniture.n.01        large furnishings (beds/sofas/chairs/cabinets/
                          tables; also table/floor lamps),
    appliance.n.02        household appliances. Chosen over
                          home_appliance.n.01 because dryer.n.01 hangs
                          directly under appliance.n.02, and over
                          white_goods.n.01 because stoves/ovens/microwaves
                          are kitchen_appliance, not white goods.
    fixture.n.01          plumbing fixtures (sink/toilet/shower),
    vehicle.n.01          car/truck/...,
    surface.n.01          countertops + room floors (installed surfaces),
    movable_barrier.n.01  doors/gates,
    framework.n.03        windows/grills (supporting structures),
    excavation.n.03       pools,
    bathtub.n.01          fixture-scale vessel; bddl files bathtubs under
                          vessel.n.03 (a graspable class), so the subtree
                          is named directly rather than dragging in every
                          pot and mug via vessel.n.03.
  NOT used: structure.n.01 (would catch plywood.n.01 — a carryable task
  object — via its lamination.n.01 occurrence); support.n.10 (would catch
  hanger.n.02, which clothes-hanging tasks must grasp).

Name handling: instance names are AS-style ``sink_7`` or SD-style
``carton.66``; base = name minus the trailing ``_<digits>`` / ``.<digits>``.
Category synsets are AS-dotted (``sink.n.01``) or SD-underscored
(``sink_n_01``) — ``taxonomy_affordances`` normalizes. When an object has
no prompt category (e.g. instances scraped from state blocks only), the
layer falls back to the name-derived guess ``<base>.n.01`` iff that synset
exists in the tree; otherwise the object has no taxonomy memberships (only
name-structural rules apply).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .taxonomy_affordances import TaxonomyAffordances, get_default

__all__ = ["BehAffordances"]


# ISA classes — see module docstring for the per-class rationale.
_NON_GRASPABLE_CLASSES: Tuple[str, ...] = (
    "furniture.n.01",
    "appliance.n.02",
    "fixture.n.01",
    "vehicle.n.01",
    "surface.n.01",
    "movable_barrier.n.01",
    "framework.n.03",
    "excavation.n.03",
    "bathtub.n.01",
)

# Containment-capable classes (see non_containable in module docstring).
_CONTAINER_CLASSES: Tuple[str, ...] = (
    "container.n.01",
    "plumbing_fixture.n.01",
)

# Cleaner ISA classes.
_CLEANER_CLASSES: Tuple[str, ...] = (
    "piece_of_cloth.n.01",
    "brush.n.02",
)

_WATER_FIXTURE_CLASS = "plumbing_fixture.n.01"

# ---------------------------------------------------------------------------
# Commonsense additions where the taxonomy's ability marks are missing or too
# coarse (physical constraints).
# ---------------------------------------------------------------------------

# Buckets, baskets and waste bins are open-top vessels: they have no lid, so
# putting things in (or taking them out) requires no opening.
# These subtrees are excluded from open-gating and from OPEN/CLOSE
# admissibility even though bddl marks them conceptually `openable`.
# (hamper.n.02 is ISA basket.n.01; ashcan/trash cans are ISA bin.n.01.)
_OPEN_TOP_VESSEL_CLASSES: Tuple[str, ...] = (
    "bucket.n.01",
    "basket.n.01",
    "bin.n.01",
)

# Bathtubs, like sinks, are water-dispensing plumbing with a tap, and a tap
# is turned on and off. bddl files bathtub under
# vessel.n.03 with no toggleable/waterSource ability, so the class is named
# directly: bathtub subtree => toggleable water fixture.
_TAP_WATER_FIXTURE_SYNSETS: Tuple[str, ...] = (
    "bathtub.n.01",
)

# Kettles and teapots dispense (hot) water; a swimming pool is a standing
# body of water. None has a switch, so they satisfy the water chain without
# a toggled_on step.
_PASSIVE_WATER_SYNSETS: Tuple[str, ...] = (
    "teapot.n.01",
    "kettle.n.01",
    "pool.n.01",
)

_BASE_RE = re.compile(r"^(.*)[._](\d+)$")


def base_of(name: str) -> str:
    """Instance name -> base: strips one trailing ``_<digits>``/``.<digits>``."""
    m = _BASE_RE.match(str(name).strip())
    return m.group(1) if m else str(name).strip()


def _is_agentish(base: str) -> bool:
    # 'agent', 'agent_n_01', 'agent.n.01' (SD scrapes agent_n_01.1 from
    # state blocks without a category entry).
    return base == "agent" or base.startswith("agent_n") or base.startswith("agent.")


class BehAffordances:
    """Shared membership layer for one prompt's object list.

    Args:
        objects: prompt object list ``[{'name', 'category'}, ...]``
            (category optional per entry — name-derived fallback applies).
        taxonomy: TaxonomyAffordances override (default: shared singleton).

    Queries accept an instance name ('sink_7', 'carton.66') OR a base name
    ('sink'); resolution order: exact instance -> base -> synset guess.
    """

    def __init__(
        self,
        objects: Optional[List[Dict]] = None,
        taxonomy: Optional[TaxonomyAffordances] = None,
    ):
        self.tax = taxonomy or get_default()
        # instance name -> category (verbatim from prompt)
        self._name_to_cat: Dict[str, str] = {}
        # base -> set of categories seen for that base
        self._base_to_cats: Dict[str, Set[str]] = {}
        for o in objects or ():
            name = str(o.get("name", "")).strip()
            cat = str(o.get("category", "") or "").strip()
            if not name:
                continue
            if cat:
                self._name_to_cat[name] = cat
                self._base_to_cats.setdefault(base_of(name), set()).add(cat)

    # -- category resolution ------------------------------------------------

    def categories_of(self, x: str) -> Tuple[str, ...]:
        """Candidate synsets for an instance-or-base name (may be empty)."""
        x = str(x).strip()
        cat = self._name_to_cat.get(x)
        if cat:
            return (cat,)
        base = base_of(x)
        cats = self._base_to_cats.get(base) or self._base_to_cats.get(x)
        if cats:
            return tuple(sorted(cats))
        # Name-derived fallback: only when the prompt supplied no category.
        guess = f"{base}.n.01"
        if self.tax.known(guess):
            return (guess,)
        return ()

    def _any_cat(self, x: str, pred) -> bool:
        return any(pred(c) for c in self.categories_of(x))

    # -- membership predicates (γ and β both read these) --------------------

    def openable(self, x: str) -> bool:
        """T:openable — the raw conceptual pool (bddl `openable`)."""
        return self._any_cat(x, self.tax.openable)

    def open_top_vessel(self, x: str) -> bool:
        """Lidless vessels (bucket/basket/bin subtrees): nothing to open,
        regardless of bddl's conceptual `openable` mark."""
        return self._any_cat(
            x, lambda c: self.tax.is_a_any(c, _OPEN_TOP_VESSEL_CLASSES))

    def open_gated(self, x: str) -> bool:
        """The effective open-state pool used by BOTH γ admissibility and β
        gating: taxonomy-openable minus open-top vessels."""
        return self.openable(x) and not self.open_top_vessel(x)

    def toggleable(self, x: str) -> bool:
        """T:toggleable ∪ tap-equipped water fixtures (bathtub)."""
        return self._any_cat(
            x,
            lambda c: self.tax.toggleable(c)
            or self.tax.is_a_any(c, _TAP_WATER_FIXTURE_SYNSETS))

    def cleaner(self, x: str) -> bool:
        return self._any_cat(
            x,
            lambda c: self.tax.cleaning_tool(c)
            or self.tax.is_a_any(c, _CLEANER_CLASSES),
        )

    def water_source(self, x: str) -> bool:
        """T:waterSource ∪ (toggleable ∧ plumbing-fixture ISA) ∪ tap water
        fixtures (bathtub) ∪ passive water vessels (kettle/teapot/pool).
        Prompt-stated vessels-of-water are ORed in by the caller
        (see ``prompt_water_vessels``)."""
        return self._any_cat(
            x,
            lambda c: self.tax.water_source(c)
            or (self.tax.toggleable(c) and self.tax.is_a(c, _WATER_FIXTURE_CLASS))
            or self.tax.is_a_any(c, _TAP_WATER_FIXTURE_SYNSETS)
            or self.tax.is_a_any(c, _PASSIVE_WATER_SYNSETS),
        )

    def passive_water_source(self, x: str) -> bool:
        """Water sources with no switch (kettle/teapot/pool, and
        prompt-stated vessels-of-water via the caller). The water chain
        does not require a toggled_on step for these; tap-equipped
        fixtures (sink, bathtub) still require turning on."""
        return (self._any_cat(
            x, lambda c: self.tax.is_a_any(c, _PASSIVE_WATER_SYNSETS))
            and not self.toggleable(x))

    def slicer(self, x: str) -> bool:
        return self._any_cat(x, lambda c: self.tax.has(c, "slicer"))

    def non_graspable(self, x: str) -> bool:
        base = base_of(x)
        # Name-structural: rooms/floors/walls/ceilings/agent have no usable
        # tree class (agent.n.01 and ceiling.n.01 are absent from the tree).
        if (base.startswith("room_floor") or base.startswith("room_")
                or base in ("floor", "wall", "ceiling")
                or _is_agentish(base)):
            return True
        return self._any_cat(
            x, lambda c: self.tax.is_a_any(c, _NON_GRASPABLE_CLASSES))

    def containable(self, x: str,
                    container_role_names: Optional[Set[str]] = None) -> bool:
        """Containment-capable: ISA container/plumbing-fixture class, or
        the prompt itself uses x in container role (``inside(_, x)``)."""
        if container_role_names:
            if x in container_role_names or base_of(x) in container_role_names:
                return True
        return self._any_cat(
            x, lambda c: self.tax.is_a_any(c, _CONTAINER_CLASSES))

    def non_containable(self, x: str,
                        container_role_names: Optional[Set[str]] = None) -> bool:
        return not self.containable(x, container_role_names)

def prompt_water_vessels(
    initial_inside_pairs: Iterable[Tuple[str, str]],
    aff: BehAffordances,
) -> Set[str]:
    """Vessels the prompt states contain water.

    ``initial_inside_pairs`` = (contained_name, container_name) pairs from
    the prompt's initial states. A container counts as a water vessel iff
    the contained object is water (category water.n.06, or base name
    'water' when no category is given). Purely prompt-grounded.
    """
    out: Set[str] = set()
    for inner, cont in initial_inside_pairs:
        cats = aff.categories_of(inner)
        if any(str(c).replace("_n_", ".n.") == "water.n.06" for c in cats) \
                or base_of(inner) == "water":
            out.add(cont)
    return out

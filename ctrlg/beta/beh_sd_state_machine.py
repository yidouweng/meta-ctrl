"""BEHAVIOR Subgoal Decomposition β state machine.

Compared to `beh_state_machine.py` (AS), the state is deliberately coarse:

  - State = (goal_bits, open_bits, clean_chain_bits). There is no tracking of
    holding / locations / general properties.
  - Transitions are lenient: a γ-valid (pred, neg, args) step is accepted
    unless it violates one of a few gates (container open before `inside`,
    initial container open before touching its contents, the clean/soak
    chain, anchor-before-relation ordering). A step that matches an unset
    goal sets that bit; any other accepted step is an identity transition.
    Extra subgoals are non-fatal in the evaluator and already-satisfied
    subgoals are skipped (evaluator property), so stricter precondition
    tracking would over-reject.
  - Anti-goals (`not pred(args)`) are first-class goal bits, keyed by
    (pred, neg=True, args). The τ parser must emit `neg` alongside the
    predicate.

Same constructor / public API surface as `BehBetaStateMachine` so
`BetaBackwardDP` consumes it without modification:
    transitions: Dict[int, List[Tuple[BehSDStep, int]]]
    accept_mask: List[bool]
    num_states / num_accept
    apply / is_accept / get_initial_state_id
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


# Predicate sets (must stay in sync with `ctrlg/gamma/topdown/behavior_sd_twolevel.py`).
_SD_LOC_PREDS = frozenset({
    'inside', 'ontop', 'onfloor', 'nextto', 'under', 'touching',
})
_SD_PROP_PREDS = frozenset({
    'open', 'cooked', 'dusty', 'frozen', 'sliced', 'soaked', 'stained',
    'toggledon',
})
_SD_HAND_PREDS = frozenset({'holds_rh', 'holds_lh'})

# Non-containment relation predicates: the first arg is positioned RELATIVE to
# the anchor (second arg) and does NOT move with it. So if the goal set also
# repositions the anchor (anchor as arg0 of a location goal), that reposition
# must be sequenced FIRST: the evaluator executes subgoals in order, and moving
# the anchor after placing X next-to/on/under it breaks the relation
# (evaluator property). `inside` is excluded: a contained object follows its
# container when it moves.
_SD_ANCHORED_REL_PREDS = frozenset({'nextto', 'ontop', 'under', 'touching'})


# Membership (openable / cleaner / water-source / prop support) comes from
# the shared taxonomy layer `ctrlg.beta.beh_affordances.BehAffordances`,
# built from the prompt's {'name','category'} object list, the same layer
# the γ DFA reads (behavior_sd_twolevel.py):
#
#   * open-gating: an `open_bit` is allocated ONLY for
#     taxonomy-openable instances in container ROLE (arg-1 of an `inside`
#     relation in the prompt's initial or goal states). The bit defaults to
#     0 (closed) unless the prompt's initial states say `open(X)`. Objects
#     that are not taxonomy-openable have nothing to open: no bit, no gate.
#     γ↔β consistency holds by construction: γ admits `open(C)` for every
#     open-gated C in container role (plus prompt-mentioned ones), a
#     superset of the β-gated set.
#   * cleaner: T:cleaningTool ∪ ISA piece_of_cloth.n.01 ∪ ISA brush.n.02
#     (soap/detergent/napkin are excluded: they are liquids / burnables in
#     BDDL).
#   * water source: T:waterSource ∪ (toggleable ∧ plumbing-fixture ISA)
#     ∪ vessels the prompt states contain water. Passive sources (no
#     switch, e.g. kettles or water-filled vessels) need no `toggledon`.


class BehSDBetaState(NamedTuple):
    """SD β state: goal bitmask + open-state bitmask + clean-chain bits.

    `open_bits` is a bitmask over closeable containers in the scene; bit k
    is 1 iff the kth closeable container has been opened (or was open at
    init). Used to enforce the `inside(X, container)` precondition.

    `clean_chain_bits` gates CLEAN_DUST / CLEAN_STAIN / SOAK on their
    evaluator preconditions. Bits:
       bit 0: any cleaner has been held       (set by holds_*(cleaner))
       bit 1: unused. A wet cleaner is modelled by bits 2+3, because the
              simulator rejects soaked(<cleaner>) as a subgoal.
       bit 2: any water source toggled on     (set by toggledon(ws))
       bit 3: an object placed inside a water source (set by inside(x, ws);
              a passive source sets bits 2+3 at once)
    Gating:
       not dusty(X)   → requires bit 0
       not stained(X) → requires bits 0+2+3 (cleaner held + staged in toggled water)
       soaked(X)      → requires bits 2+3
    """
    goal_bits: int
    open_bits: int
    clean_chain_bits: int = 0


@dataclass(frozen=True)
class BehSDGoal:
    """A parsed (possibly-negated) goal predicate."""
    predicate: str           # lowercase, e.g. 'open', 'inside'
    neg: bool                # True → goal is `not predicate(args)`
    args: Tuple[int, ...]    # canonical IDs
    goal_idx: int            # bit position in goal_bits


@dataclass
class BehSDStep:
    """A grounded subgoal predicate. Same interface shape as BehStep.

    `action` is the upper-cased predicate (with `NOT_` prefix when negated)
    so `BetaBackwardDP._step_to_semantic_tokens` produces a single semantic
    token per polarity. `args` is a tuple of canonical IDs.
    """
    action: str
    args: Tuple[int, ...]


def _step_to_action_token(pred: str, neg: bool) -> str:
    """Canonical action-token form for a (pred, neg) pair."""
    base = pred.upper()
    return f"NOT_{base}" if neg else base


def _action_token_to_pred(action: str) -> Tuple[str, bool]:
    """Inverse of _step_to_action_token."""
    if action.startswith('NOT_'):
        return action[4:].lower(), True
    return action.lower(), False


class BehSDBetaStateMachine:
    """β state machine for BEHAVIOR Subgoal Decomposition.

    Tracks only goal coverage (a bitmask over expanded task goals). Anti-goals
    are first-class bits keyed by (pred, neg=True, args).

    Args:
        initial_states: Initial-state ASTs from `parse_beh_sd_states`; used
            for initial open/closed containers and initial containment.
        target_states: Goal ASTs from `parse_beh_sd_states`, e.g.
            [('atom', 'inside', False, ['book.0', 'carton.66']),
             ('not', ('atom', 'open', False, ['jar.0'])),
             ('forall', 'book_n_02', ('atom', 'inside', False, ['book_n_02', 'carton.66']))]
            Quantifiers are expanded over `objects` (see `_parse_goals`).
        all_instances: Sorted list of canonical instance names (e.g. ['book.0', ...]).
        inst_to_cid: {instance_name: canonical_id}.
        gamma_valid_steps: List of (pred, neg, args) tuples — every step the γ
            DFA can produce. β builds transitions only for these, mirroring
            `BehBetaStateMachine.gamma_valid_actions`.
        max_states: BFS cap. Goal-dropping kicks in if exceeded (one goal at a time).
        max_budget: Stored for API parity with `BehBetaStateMachine`; not
            used by this class.
        debug: Print build statistics.
        objects: List of {name, category} dicts for quantifier expansion and
            the taxonomy membership layer (optional).
        edge_budget: Cap on num_states × num_candidate_steps; goal-dropping
            kicks in when exceeded.
    """

    def __init__(
        self,
        initial_states: List[List],
        target_states: List[List],
        all_instances: List[str],
        inst_to_cid: Dict[str, int],
        gamma_valid_steps: List[Tuple[str, bool, Tuple[int, ...]]],
        max_states: int = 200000,
        max_budget: int = 30,
        debug: bool = False,
        objects: Optional[List[Dict]] = None,
        edge_budget: int = 800_000,
    ):
        self.debug = debug
        self.max_states = max_states
        # DP-cost budget. BetaBackwardDP is built from `self.transitions`
        # (≈ num_states × num_candidate_steps edges) and its table build is
        # O(B × S × U × H²), which a timeout cannot interrupt, so the cost is
        # bounded here at build time: goal-dropping triggers whenever
        # `num_states × num_candidates > edge_budget`. A β SM tracking a
        # goal subset still steers toward a valid partial-goal plan.
        self.edge_budget = edge_budget
        self.all_instances = all_instances
        self.inst_to_cid = inst_to_cid
        self.gamma_valid_steps = list(gamma_valid_steps)
        self._build_max_budget = max_budget

        # SD-specific expansion: forall/exists into concrete (pred, neg, args)
        # tuples. The AS `expand_goals` expects pre-tokenised list form and
        # explicitly skips negated location goals; SD needs both, so we expand
        # directly here.
        self._raw_target_states = list(target_states)
        cat2inst: Dict[str, List[str]] = {}
        if objects:
            for o in objects:
                cat = o.get('category', '')
                name = o.get('name', '')
                if cat and name:
                    cat2inst.setdefault(cat, []).append(name)

        # Parse goals. `goals` is a flat list of registered atom entries;
        # multiple entries may share `goal_idx` when they belong to the same
        # OR group (any branch sets the same bit).
        self.goals: List[BehSDGoal] = []
        self._goal_key_to_idx: Dict[Tuple[str, bool, Tuple[int, ...]], int] = {}
        # For priority goal-dropping: goal_idx → forall-group id (-1 if the
        # goal is not a `forall` expansion). Goals from the same `forall`
        # share a group id so dropping keeps ≥1 representative per group.
        self._goal_idx_to_forall_group: Dict[int, int] = {}
        self._n_forall_groups = 0
        n_atoms = self._parse_goals(target_states, cat2inst)
        # Number of distinct goal slots = number of unique goal_idx values.
        self._goal_slots: List[int] = sorted({g.goal_idx for g in self.goals})
        self.n_goals = len(self._goal_slots)
        if self.debug:
            print(f"[BehSDBetaSM] expansion+parse: {len(self._raw_target_states)} "
                  f"raw → {n_atoms} atoms → {len(self.goals)} entries "
                  f"→ {self.n_goals} distinct goal slots (OR-shared)")

        # Shared taxonomy membership layer (γ built its own copy from the
        # same prompt object list — same sets by construction).
        from .beh_affordances import BehAffordances, prompt_water_vessels
        self.membership = BehAffordances(objects=objects)

        # Container ROLE: instances appearing as arg-1 of an
        # `inside` relation in the prompt's goal or initial states. Only
        # role-scoped ∧ taxonomy-openable instances get an open bit.
        self._initial_inside_pairs = self._collect_initial_inside_pairs(
            initial_states)
        container_role_cids: Set[int] = set()
        for g in self.goals:
            if g.predicate == 'inside' and len(g.args) >= 2:
                container_role_cids.add(g.args[1])
        for _inner, cont in self._initial_inside_pairs:
            c_cid = inst_to_cid.get(cont)
            if c_cid is not None:
                container_role_cids.add(c_cid)

        # Allocate open_bit indices for taxonomy-openable containers in
        # container role. Default state = closed (generic conservative
        # default) unless the prompt's initial states say `open(X)`.
        # Non-openable objects are naturally ungated: no bit, no gate.
        self._cid_to_open_bit: Dict[int, int] = {}
        self._initial_open_bits = 0
        bit = 0
        for inst, cid in inst_to_cid.items():
            if cid in container_role_cids and self.membership.open_gated(inst):
                # open_gated = taxonomy-openable minus open-top vessels
                # (buckets/baskets/bins need no opening).
                self._cid_to_open_bit[cid] = bit
                bit += 1
        # Parse initial states for explicit open(X) / not open(X).
        self._parse_initial_open_states(initial_states)
        self.n_open_bits = bit

        # Source-container gate (simulator constraint).
        # Scan `## Initial States` for `inside(X, C)` where C is a closeable
        # container — X then cannot be grasped (`holds_*`) or sourced out
        # (`inside(X, _)` / `not inside(X, C)`) until C is opened. Maps
        # X's cid → the open_bit mask of its initial container(s).
        self._cid_to_src_open_need: Dict[int, int] = {}
        self._parse_initial_containers(initial_states)

        # CLEAN/SOAK chain: identify cleaner instances in the scene and
        # decide whether the chain is active. Bits and gating are described
        # in the `BehSDBetaState` docstring. When the chain is inactive the
        # bits stay 0 (no state-space growth).
        self._cleaner_cids: Set[int] = set()
        for inst, cid in inst_to_cid.items():
            if self.membership.cleaner(inst):
                self._cleaner_cids.add(cid)
        # Water sources: taxonomy rule ∪ prompt-stated vessels-of-water
        # (third clause — prompt-grounded).
        _water_vessels = prompt_water_vessels(
            self._initial_inside_pairs, self.membership)
        self._water_source_cids: Set[int] = set()
        # Passive water sources (e.g. kettle/teapot/pool, prompt
        # vessels-of-water) provide water without a switch: staging an
        # object inside one satisfies the whole water side of the chain
        # (bits 2+3) with no toggledon step.
        self._passive_water_cids: Set[int] = set()
        for inst, cid in inst_to_cid.items():
            is_vessel = inst in _water_vessels
            if self.membership.water_source(inst) or is_vessel:
                self._water_source_cids.add(cid)
                if (self.membership.passive_water_source(inst)
                        or (is_vessel and not self.membership.toggleable(inst))):
                    self._passive_water_cids.add(cid)
        # Any CLEAN-style or SOAK-target goal in the user-target?
        self._has_clean_goal_dust = any(
            g.neg and g.predicate == 'dusty' for g in self.goals
        )
        self._has_clean_goal_stain = any(
            g.neg and g.predicate == 'stained' for g in self.goals
        )
        self._has_soak_goal = any(
            (not g.neg) and g.predicate == 'soaked' for g in self.goals
        )
        # Active when a cleaner exists and a clean/soak goal needs it, or
        # whenever there is a soak goal (soaking a non-cleaner still needs
        # the toggle + inside chain).
        self._clean_chain_active = (
            (len(self._cleaner_cids) > 0
             and (self._has_clean_goal_dust or self._has_clean_goal_stain
                  or self._has_soak_goal))
            or
            self._has_soak_goal
        )
        if self.debug:
            print(f"[BehSDBetaSM] clean-chain: active={self._clean_chain_active} "
                  f"cleaners={len(self._cleaner_cids)} "
                  f"dust_goal={self._has_clean_goal_dust} "
                  f"stain_goal={self._has_clean_goal_stain}")

        if self.debug:
            print(f"[BehSDBetaSM] open-tracking: {self.n_open_bits} closeable "
                  f"containers; initial_open_bits={self._initial_open_bits:b}; "
                  f"Gate-2: {len(self._cid_to_src_open_need)} objs start "
                  f"inside a closeable container")

        # Build transitions.
        self.states: List[BehSDBetaState] = []
        self.state_to_id: Dict[BehSDBetaState, int] = {}
        self.transitions: Dict[int, List[Tuple[BehSDStep, int]]] = {}
        self.accept_mask: List[bool] = []
        self.dropped_goal_indices: List[int] = []  # filled if we goal-drop
        self._build_with_dropping()

    # ------------------------------------------------------------------
    # Goal parsing
    # ------------------------------------------------------------------

    def _add_goal(self, predicate: str, neg: bool, args: Tuple[int, ...],
                  shared_idx: Optional[int] = None) -> int:
        """Register a (pred, neg, args) goal entry; returns the assigned
        goal_idx. If shared_idx is given, the new key maps onto an existing
        goal slot (used for OR groups: any branch satisfies the same bit).
        """
        key = (predicate, neg, args)
        existing = self._goal_key_to_idx.get(key)
        if existing is not None:
            return existing
        if shared_idx is not None:
            idx = shared_idx
        else:
            # Pick a fresh goal_idx (max used + 1).
            idx = max((g.goal_idx for g in self.goals), default=-1) + 1
        self._goal_key_to_idx[key] = idx
        self.goals.append(BehSDGoal(predicate=predicate, neg=neg,
                                    args=args, goal_idx=idx))
        return idx

    def _resolve_args(self, raw_args: List[str]) -> Optional[Tuple[int, ...]]:
        """Map instance-name tokens to canonical ids; None if any unknown."""
        cids: List[int] = []
        for raw in raw_args:
            cid = self.inst_to_cid.get(raw, -1)
            if cid < 0:
                return None
            cids.append(cid)
        return tuple(cids)

    def _parse_goals(self, target_states: List[List],
                     cat2inst: Dict[str, List[str]]) -> int:
        """Parse target predicates into BehSDGoal entries with quantifier
        expansion. Returns the total number of concrete atoms produced
        (before deduplication).

        Accepted entry forms (output of `parse_beh_sd_states`):
            ('atom', pred, neg, [arg1, arg2, ...])
                — concrete predicate; args are instance-name strings.
            ('forall', VAR, body)            — body is an atom/compound
            ('exists', VAR, body)            — same shape
            ('forpairs', VAR_A, VAR_B, body) — body references both vars
            ('forn', N, VAR, body)           — N instances satisfy body
            ('and', [child1, child2, ...])   — conjunction
            ('or', [child1, child2, ...])    — disjunction (one shared bit)
            ('not', child)                   — negation wrapper

        Forall: emit one expanded child per scene instance of VAR's category.
        Exists: anchor on the first instance.
        Forpairs: zip instances of the two categories index-wise (one
        valid matching). Forn: expand the first N instances of VAR.
        """
        n_atoms = 0
        skipped = 0

        def _resolve_var(token, var_bindings: Dict[str, str]) -> str:
            """If token is a quantifier-bound variable name, substitute its
            anchored instance; otherwise pass through."""
            if isinstance(token, str) and token in var_bindings:
                return var_bindings[token]
            return token

        def _walk(node, neg_outer: bool, var_bindings: Dict[str, str],
                  share_idx: Optional[int], forall_group: int = -1) -> None:
            """Recursively expand goal AST.

            share_idx, when not None, forces all atoms produced by this
            sub-tree to share a single goal_idx — used to implement OR groups
            ("any branch satisfies the same bit").

            forall_group, when >= 0, tags every goal slot produced inside
            this sub-tree as belonging to that `forall` expansion (used by
            priority goal-dropping to keep one representative per group).
            """
            nonlocal n_atoms, skipped

            if not isinstance(node, (list, tuple)) or not node:
                skipped += 1
                return
            tag = node[0]

            if tag == 'atom':
                _, pred, neg, raw_args = node
                resolved = [_resolve_var(a, var_bindings) for a in raw_args]
                args = self._resolve_args(resolved)
                if args is None:
                    skipped += 1
                    return
                expected = 2 if pred in _SD_LOC_PREDS else 1
                if len(args) != expected:
                    skipped += 1
                    return
                idx = self._add_goal(pred, neg ^ neg_outer, args,
                                     shared_idx=share_idx)
                if forall_group >= 0:
                    self._goal_idx_to_forall_group.setdefault(idx, forall_group)
                n_atoms += 1
                return

            if tag == 'not':
                _walk(node[1], not neg_outer, var_bindings,
                      share_idx, forall_group)
                return

            if tag == 'and':
                # Each conjunct must be satisfied → fresh goal_idx per child
                # (unless we're inside an OR group, in which case the whole
                # AND collapses into the shared bit — i.e. the OR branch is
                # a conjunction of atoms that ALL must hold to count as one
                # alternative, but for lenient β we approximate by sharing).
                for child in node[1]:
                    _walk(child, neg_outer, var_bindings,
                          share_idx, forall_group)
                return

            if tag == 'or':
                # OR: allocate ONE new goal slot for the whole disjunction;
                # every branch maps to the same bit so satisfying any one
                # sets it. Nested ORs reuse the same slot.
                if share_idx is None:
                    new_idx = max((g.goal_idx for g in self.goals),
                                  default=-1) + 1
                    or_idx = new_idx
                else:
                    or_idx = share_idx
                for child in node[1]:
                    _walk(child, neg_outer, var_bindings,
                          or_idx, forall_group)
                return

            if tag == 'forall':
                _, var, body = node
                instances = cat2inst.get(var, [])
                if not instances:
                    skipped += 1
                    return
                # Fresh group id so all instances of this `forall` share it.
                fg = self._n_forall_groups
                self._n_forall_groups += 1
                for inst in instances:
                    _walk(body, neg_outer,
                          {**var_bindings, var: inst}, share_idx, fg)
                return

            if tag == 'exists':
                _, var, body = node
                instances = cat2inst.get(var, [])
                if not instances:
                    skipped += 1
                    return
                # Anchor on first instance (matches AS expander).
                _walk(body, neg_outer,
                      {**var_bindings, var: instances[0]},
                      share_idx, forall_group)
                return

            if tag == 'forpairs':
                # Lenient β can't represent "some bijection exists"; zip
                # instances index-wise to steer toward ONE valid matching
                # (which the gold checker still accepts as a bijection).
                _, var_a, var_b, body = node
                inst_a = cat2inst.get(var_a, [])
                inst_b = cat2inst.get(var_b, [])
                if not inst_a or not inst_b:
                    skipped += 1
                    return
                fg = self._n_forall_groups
                self._n_forall_groups += 1
                for ia, ib in zip(inst_a, inst_b):
                    _walk(body, neg_outer,
                          {**var_bindings, var_a: ia, var_b: ib},
                          share_idx, fg)
                return

            if tag == 'forn':
                # N instances of VAR satisfy body — expand the first N.
                _, n, var, body = node
                instances = cat2inst.get(var, [])
                if not instances:
                    skipped += 1
                    return
                fg = self._n_forall_groups
                self._n_forall_groups += 1
                for inst in instances[:max(0, n)]:
                    _walk(body, neg_outer,
                          {**var_bindings, var: inst}, share_idx, fg)
                return

            skipped += 1

        for entry in target_states:
            _walk(entry, False, {}, None, -1)

        if self.debug:
            print(f"[BehSDBetaSM] {n_atoms} concrete atoms produced, "
                  f"{skipped} skipped during expansion")
        return n_atoms

    def _parse_initial_open_states(self, initial_states: List) -> None:
        """Scan the prompt's `## Initial States` AST for `open(X)` /
        `not open(X)` atoms and seed `_initial_open_bits` accordingly.
        Bits for instances not mentioned keep their default (closed)."""
        if not initial_states:
            return
        for entry in initial_states:
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            # Recognise ('atom', 'open', False, ['X']) and ('not', ('atom',
            # 'open', False, ['X'])).
            atom = None
            neg = False
            if entry[0] == 'atom':
                atom = entry
            elif entry[0] == 'not' and len(entry) > 1 and isinstance(entry[1], tuple) \
                    and len(entry[1]) >= 1 and entry[1][0] == 'atom':
                atom = entry[1]
                neg = True
            if atom is None or len(atom) < 4:
                continue
            _, pred, atom_neg, raw_args = atom
            if pred != 'open' or len(raw_args) < 1:
                continue
            inst = raw_args[0]
            cid = self.inst_to_cid.get(inst)
            if cid is None or cid not in self._cid_to_open_bit:
                continue
            bit = self._cid_to_open_bit[cid]
            if (atom_neg ^ neg):  # `not open(X)` → closed
                self._initial_open_bits &= ~(1 << bit)
            else:                 # `open(X)` → open
                self._initial_open_bits |= (1 << bit)

    def _collect_initial_inside_pairs(
        self, initial_states: List,
    ) -> List[Tuple[str, str]]:
        """Positive `inside(X, C)` atoms from the `## Initial States` AST as
        (X_name, C_name) pairs. Used for container-ROLE detection and
        prompt-stated vessels-of-water BEFORE cid maps for bits exist.
        """
        pairs: List[Tuple[str, str]] = []
        if not initial_states:
            return pairs
        for entry in initial_states:
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            atom = None
            neg = False
            if entry[0] == 'atom':
                atom = entry
            elif entry[0] == 'not' and len(entry) > 1 \
                    and isinstance(entry[1], tuple) and len(entry[1]) >= 1 \
                    and entry[1][0] == 'atom':
                atom = entry[1]
                neg = True
            if atom is None or len(atom) < 4:
                continue
            _, pred, atom_neg, raw_args = atom
            if pred != 'inside' or (atom_neg ^ neg) or len(raw_args) < 2:
                continue
            pairs.append((raw_args[0], raw_args[1]))
        return pairs

    def _parse_initial_containers(self, initial_states: List) -> None:
        """Scan `## Initial States` for `inside(X, C)` atoms where C is a
        closeable container (has an open_bit). Record, per X, the open_bit
        mask of its initial container so that any step targeting X can be
        β-gated on C being open first (simulator constraint: acting on an
        object inside a closed container fails)."""
        if not initial_states:
            return
        for entry in initial_states:
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            atom = None
            neg = False
            if entry[0] == 'atom':
                atom = entry
            elif entry[0] == 'not' and len(entry) > 1 \
                    and isinstance(entry[1], tuple) and len(entry[1]) >= 1 \
                    and entry[1][0] == 'atom':
                atom = entry[1]
                neg = True
            if atom is None or len(atom) < 4:
                continue
            _, pred, atom_neg, raw_args = atom
            if pred != 'inside' or (atom_neg ^ neg) or len(raw_args) < 2:
                continue
            x_cid = self.inst_to_cid.get(raw_args[0])
            c_cid = self.inst_to_cid.get(raw_args[1])
            if x_cid is None or c_cid is None:
                continue
            cbit = self._cid_to_open_bit.get(c_cid)
            if cbit is None:
                continue  # container not closeable → no gate needed
            self._cid_to_src_open_need[x_cid] = \
                self._cid_to_src_open_need.get(x_cid, 0) | (1 << cbit)

    # ------------------------------------------------------------------
    # Transition / state-space build
    # ------------------------------------------------------------------

    def _step_sets_goal_bit(self, pred: str, neg: bool,
                            args: Tuple[int, ...]) -> int:
        """Return the goal bit (mask) this step satisfies, or 0 if none."""
        idx = self._goal_key_to_idx.get((pred.lower(), neg, args))
        if idx is None:
            return 0
        return 1 << idx

    def _is_accept(self, state: BehSDBetaState) -> bool:
        # `_active_slot_count` is set by _build_with_dropping; before build,
        # fall back to the original slot count so initial-state queries work.
        n = getattr(self, '_active_slot_count', self.n_goals)
        if n == 0:
            return True
        idx_to_bit = getattr(self, '_goal_idx_to_bit', None)
        if idx_to_bit is None:
            full = (1 << n) - 1
            return (state.goal_bits & full) == full
        # active_slots[bit] = slot
        active_slots = [None] * len(idx_to_bit)
        for slot, bit in idx_to_bit.items():
            active_slots[bit] = slot
        return self._state_satisfies_active(state, active_slots)

    def _state_satisfies_active(
        self, state: BehSDBetaState, active_slots: List[int],
    ) -> bool:
        """True iff every active goal (in bit order) is satisfied by state.

        State-prop predicates (BEH SD only tracks `open`) are derived from
        `open_bits`, matching the evaluator's final-state check: emitting
        `open(X)` then `not open(X)` leaves X closed even though the
        monotonic goal bit for `open(X)` stays set.

        Emission-tracked goals (relations: inside/ontop/nextto; properties:
        dusty/stained/soaked/cooked/...; hand preds) still read goal_bits.
        """
        for bit, slot in enumerate(active_slots):
            g = self._slot_repr_goal(slot)
            if g is None:
                return False
            if g.predicate == 'open' and len(g.args) >= 1:
                obit = self._cid_to_open_bit.get(g.args[0])
                if obit is None:
                    if not (state.goal_bits & (1 << bit)):
                        return False
                    continue
                is_open = bool(state.open_bits & (1 << obit))
                if g.neg and is_open:        # `not open(X)` goal, X is open
                    return False
                if not g.neg and not is_open:  # `open(X)` goal, X is closed
                    return False
            else:
                if not (state.goal_bits & (1 << bit)):
                    return False
        return True

    def _build_steps_for_goal_set(
        self, active_slots: List[int],
    ) -> List[Tuple[BehSDStep, int]]:
        """Precompute (step, set_mask) for every γ-valid step under a chosen
        subset of active goal slots. set_mask is 0 for steps that don't match.

        Each slot may be hit by multiple (pred, neg, args) entries (OR
        branches): all of them produce the same bit.
        """
        slot_to_bit = {slot: bit for bit, slot in enumerate(active_slots)}
        active_key_to_mask: Dict[
            Tuple[str, bool, Tuple[int, ...]], int
        ] = {}
        for g in self.goals:
            bit = slot_to_bit.get(g.goal_idx)
            if bit is None:
                continue
            key = (g.predicate, g.neg, g.args)
            # OR-fold: |= so entries sharing a slot still set the same bit.
            active_key_to_mask[key] = active_key_to_mask.get(key, 0) | (1 << bit)

        # For each step also precompute its effect on open_bits and any
        # `inside`-precondition the step depends on. Tuple shape:
        # (step, set_mask, open_set, open_clear, need_open,
        #  chain_set, chain_need)
        out: List[Tuple[BehSDStep, int, int, int, int, int, int]] = []
        for pred, neg, args in self.gamma_valid_steps:
            # A binary relation between an object and itself
            # (inside(X,X)/ontop(X,X)/...) is never realizable (physical
            # constraint), and no dataset goal contains one.
            if len(args) == 2 and args[0] == args[1]:
                continue
            mask = active_key_to_mask.get((pred.lower(), neg, args), 0)
            step = BehSDStep(action=_step_to_action_token(pred, neg), args=args)
            # open-side effect
            open_set = 0   # bits to set
            open_clear = 0  # bits to clear
            need_open = 0   # bits that must be set on src
            # clean-chain effect (only when active)
            chain_set = 0   # bits to set in clean_chain_bits
            chain_need = 0  # bits required to be set on src
            p = pred.lower()
            if p == 'open' and len(args) >= 1:
                cid = args[0]
                bit = self._cid_to_open_bit.get(cid)
                if bit is not None:
                    if neg:
                        open_clear |= (1 << bit)
                    else:
                        open_set |= (1 << bit)
            elif p == 'inside' and not neg and len(args) >= 2:
                cont_cid = args[1]
                bit = self._cid_to_open_bit.get(cont_cid)
                if bit is not None:
                    need_open |= (1 << bit)
                # Moving X out also requires X's initial container open.
                need_open |= self._cid_to_src_open_need.get(args[0], 0)
            elif p in _SD_HAND_PREDS and not neg and len(args) == 1:
                # Grasping X requires X's initial container open.
                need_open |= self._cid_to_src_open_need.get(args[0], 0)
            elif p in _SD_PROP_PREDS and not neg and len(args) >= 1 and p != 'open':
                # Any property predicate that targets X requires X's
                # initial container open (simulator constraint). `open`
                # itself is excepted: opening a container does not depend
                # on the container already being accessible.
                need_open |= self._cid_to_src_open_need.get(args[0], 0)
            elif p in _SD_LOC_PREDS and not neg and len(args) >= 1 and p != 'inside':
                # Other location predicates (ontop/onfloor/nextto/under/
                # touching) targeting X also require X accessible.
                need_open |= self._cid_to_src_open_need.get(args[0], 0)

            # ---- CLEAN/SOAK chain effects + preconditions ----
            # Must mirror apply() exactly, or runtime `state_to_id` lookups
            # miss states the enumeration did not build. Bit semantics are in
            # the BehSDBetaState docstring.
            if self._clean_chain_active:
                if (p in _SD_HAND_PREDS and not neg and len(args) == 1
                        and args[0] in self._cleaner_cids):
                    chain_set |= 0b0001
                elif (p == 'toggledon' and not neg and len(args) == 1
                        and args[0] in self._water_source_cids):
                    chain_set |= 0b0100
                elif (p == 'inside' and not neg and len(args) >= 2
                        and args[1] in self._water_source_cids):
                    # Cleaner or any soakable object placed in a water
                    # source. Passive sources need no toggling: staging
                    # inside one completes the water side (bits 2+3).
                    if args[1] in self._passive_water_cids:
                        chain_set |= 0b1100
                    else:
                        chain_set |= 0b1000
                # Gating:
                #   not dusty(X)   needs cleaner held (bit 0)
                #   not stained(X) needs cleaner held + staged in toggled
                #                  water (bits 0+2+3 = 0b1101)
                #   soaked(X)      needs toggled-on WS + target in WS
                #                  (bits 2+3)
                if p == 'dusty' and neg and len(args) >= 1:
                    chain_need |= 0b0001
                elif p == 'stained' and neg and len(args) >= 1:
                    chain_need |= 0b1101
                elif p == 'soaked' and not neg and len(args) >= 1:
                    chain_need |= 0b1100
            out.append((step, mask, open_set, open_clear, need_open,
                        chain_set, chain_need))
        return out

    def _build_for_goals(
        self, active_slots: List[int],
    ) -> Tuple[bool, List[BehSDBetaState], Dict[BehSDBetaState, int],
               Dict[int, List[Tuple[BehSDStep, int]]]]:
        """BFS over (goal_bits, open_bits, clean_chain_bits) using the given
        active-goal subset.

        Returns (ok, states, state_to_id, transitions). `ok` is False if BFS
        hits the effective state cap, which is the *tighter* of `max_states`
        and `edge_budget // n_cands` — keeping `num_states × n_cands` (the
        β edge structure E) inside one cost budget.
        """
        step_table = self._build_steps_for_goal_set(active_slots)

        # Anchor-ordering precedence (active-slot bit space). For each anchored
        # relation goal R(X, C) kept in this build, require every kept reposition
        # goal of C (C as arg0 of a location goal) to be set first. Encoded as a
        # per-step `need_goal` mask: a step's set_mask bits map to the precede
        # requirement of the goal(s) it sets. Reads existing goal_bits → no new
        # state. Only fires when both a relation _(X,C) and reposition _(C,_) are
        # kept. (Mirror of the runtime check in apply().)
        _slot_to_bit = {s: i for i, s in enumerate(active_slots)}
        _bit_precede: Dict[int, int] = {}
        for g in self.goals:
            if (g.neg or g.predicate not in _SD_ANCHORED_REL_PREDS
                    or len(g.args) < 2 or g.goal_idx not in _slot_to_bit):
                continue
            anchor = g.args[1]
            m = 0
            for h in self.goals:
                if (h.goal_idx in _slot_to_bit and h.goal_idx != g.goal_idx
                        and not h.neg and h.predicate in _SD_LOC_PREDS
                        and len(h.args) >= 1 and h.args[0] == anchor):
                    m |= (1 << _slot_to_bit[h.goal_idx])
            if m:
                _bit_precede[_slot_to_bit[g.goal_idx]] = m
        # Per-step need_goal: OR the precede masks of the bits this step sets.
        need_goal_table = []
        for entry in step_table:
            sm = entry[1]  # set_mask
            ng = 0
            if _bit_precede and sm:
                b = 0
                while sm:
                    if sm & 1:
                        ng |= _bit_precede.get(b, 0)
                    sm >>= 1
                    b += 1
            need_goal_table.append(ng)

        # Effective state cap: keep E = S × U within `edge_budget`, i.e.
        # S <= edge_budget // n_cands, and never above `max_states`.
        n_cands = max(1, len(step_table))
        eff_cap = min(self.max_states,
                      max(1, self.edge_budget // n_cands))

        initial = BehSDBetaState(
            goal_bits=0,
            open_bits=self._initial_open_bits,
            clean_chain_bits=0,
        )
        states = [initial]
        state_to_id = {initial: 0}
        transitions: Dict[int, List[Tuple[BehSDStep, int]]] = {}

        queue = deque([0])

        while queue:
            src_idx = queue.popleft()
            src = states[src_idx]
            trans: List[Tuple[BehSDStep, int]] = []

            for _si, (step, set_mask, open_set, open_clear, need_open,
                 chain_set, chain_need) in enumerate(step_table):
                # Precondition: required-open bits must already be set on src.
                if need_open and (src.open_bits & need_open) != need_open:
                    continue  # β rejects this transition
                # Anchor-ordering precedence: reposition the anchor first.
                ng = need_goal_table[_si]
                if ng and (src.goal_bits & ng) != ng:
                    continue  # β rejects → β-DP can only order anchor-first
                # Clean-chain precondition (cleaner-held / soaked-cleaner).
                if (chain_need
                        and (src.clean_chain_bits & chain_need) != chain_need):
                    continue
                new_goal_bits = src.goal_bits | set_mask
                new_open_bits = (src.open_bits | open_set) & ~open_clear
                new_chain_bits = src.clean_chain_bits | chain_set
                dst_state = BehSDBetaState(
                    goal_bits=new_goal_bits, open_bits=new_open_bits,
                    clean_chain_bits=new_chain_bits,
                )

                dst_idx = state_to_id.get(dst_state)
                if dst_idx is None:
                    if len(states) >= eff_cap:
                        return False, states, state_to_id, transitions
                    dst_idx = len(states)
                    states.append(dst_state)
                    state_to_id[dst_state] = dst_idx
                    queue.append(dst_idx)

                # Deduplicate: if multiple γ-valid steps share the same dst,
                # they all need to be in the transition list because the τ
                # parser identifies them by `step.action / args`. Don't dedupe
                # by dst_idx — different steps must each appear so DP picks
                # them up by step key, not by edge.
                trans.append((step, dst_idx))

            transitions[src_idx] = trans

        return True, states, state_to_id, transitions

    def _slot_repr_goal(self, slot: int) -> Optional['BehSDGoal']:
        """A representative BehSDGoal entry for a goal slot (slots may have
        several entries when OR-shared; any is fine for arity/neg)."""
        return next((g for g in self.goals if g.goal_idx == slot), None)

    def _pick_drop_slot(self, active: List[int]) -> int:
        """Priority goal-dropping. Pick which active goal slot to drop
        next: the one β-steering helps least. Priority (drop first):

          1. `forall`-tail instances — a slot from a `forall` group that
             still has ≥2 active members of the SAME predicate signature
             and is NOT that subgroup's kept representative (lowest-index
             member). The LM continues a `forall` pattern unaided once one
             instance is steered, so the tail is the cheapest to drop.
          2. simpler 1-arg goals (`sliced(x)`, `dusty(x)`) before 2-arg
             (`inside(x,y)`, `ontop(x,y)`) — 1-arg goals are easier for the
             LM to satisfy solo.
          3. positive goals before anti-goals (`not …`) — anti-goals are
             uncommon in LM priors and need β-steering most, so keep them.
          4. tiebreak: highest parsed index (positional, deterministic).

        The representative is kept per (forall_group, predicate, neg)
        subgroup — NOT per group. A `forall`/`forpairs` body often mixes
        predicates (e.g. `inside(x,dish) and sliced(x)`); keying the rep
        per group alone would let an entire predicate be fully dropped,
        zeroing β-steering for that subgoal.
        """
        # Active members per (forall_group, predicate, neg) subgroup, and
        # each subgroup's kept representative (lowest-index active member).
        sub_members: Dict[Tuple[int, str, bool], List[int]] = {}
        for s in active:
            fg = self._goal_idx_to_forall_group.get(s, -1)
            if fg < 0:
                continue
            g = self._slot_repr_goal(s)
            sig = (fg, g.predicate if g else '', bool(g.neg) if g else False)
            sub_members.setdefault(sig, []).append(s)
        sub_rep = {sig: min(ms) for sig, ms in sub_members.items()}

        def rank(s: int) -> Tuple[int, int, int, int]:
            fg = self._goal_idx_to_forall_group.get(s, -1)
            g = self._slot_repr_goal(s)
            sig = (fg, g.predicate if g else '', bool(g.neg) if g else False)
            is_tail = (fg >= 0 and len(sub_members.get(sig, [])) > 1
                       and s != sub_rep.get(sig))
            arity = len(g.args) if g is not None else 1
            neg = bool(g.neg) if g is not None else False
            # higher tuple == dropped sooner
            return (1 if is_tail else 0,
                    1 if arity <= 1 else 0,
                    1 if not neg else 0,
                    s)

        return max(active, key=rank)

    def _build_with_dropping(self) -> None:
        """BFS-build over distinct goal slots; drop one slot at a time
        whenever `_build_for_goals` hits the effective state cap = min(
        max_states, edge_budget / #candidates), or accept is unreachable.

        The edge-budget bound keeps `BetaBackwardDP`'s table build
        tractable on heavy prompts. Drop order is priority-based
        (`_pick_drop_slot`): drop the goals β-steering helps least
        (forall-tails → 1-arg → positive), keep anti-goals and one
        representative per `forall` subgroup.

        Uses the shared `build_with_dropping` helper. A build is accepted
        when BOTH (a) BFS did not hit the cap AND (b) ≥1 built state is
        accepting under the active subset; otherwise drop and rebuild. With
        0 active goals every state is vacuously accepting, so the loop
        always terminates.
        """
        from .goal_drop import build_with_dropping

        def _build(active: List[int]):
            ok, states, state_to_id, transitions = self._build_for_goals(active)
            return (states, state_to_id, transitions), ok

        def _has_accept(result, active: List[int]) -> bool:
            states, _, _ = result
            n = len(active)
            if n == 0:
                return bool(states)  # vacuously accept everywhere
            # Dynamic state-prop check (active[bit] = slot).
            return any(self._state_satisfies_active(s, active) for s in states)

        def _pick(active: List[int]) -> int:
            slot = self._pick_drop_slot(active)
            return active.index(slot)

        result, active, dropped = build_with_dropping(
            list(self._goal_slots),
            build_fn=_build,
            has_accept_fn=_has_accept,
            pick_drop_fn=_pick,
            min_active=0,
            debug=self.debug,
            label='BehSDBetaSM',
        )
        states, state_to_id, transitions = result

        self.dropped_goal_indices = dropped
        self._active_slot_count = len(active)
        self.states = states
        self.state_to_id = state_to_id
        self.transitions = transitions

        # Map goal_idx → bit position in the kept slot ordering. _is_accept
        # reads it, so it must exist before accept_mask is computed.
        self._goal_idx_to_bit: Dict[int, int] = {
            slot: bit for bit, slot in enumerate(active)
        }
        self.accept_mask = [self._is_accept(s) for s in self.states]

        # Anchor-ordering precedence (no new state — reads existing goal_bits).
        # For each anchored relation goal R(X, C) (C = args[1]), require every
        # reposition goal of C (C as arg0 of a location goal) to be set first.
        # apply() rejects R until that mask is satisfied → the β-DP can only
        # reach accept in an order that positions C before placing X relative
        # to it, matching the evaluator's sequential execution. Only fires when both
        # a relation _(X,C) and a reposition _(C,_) goal coexist.
        self._goal_precede_mask: Dict[int, int] = {}
        for g in self.goals:
            if g.neg or g.predicate not in _SD_ANCHORED_REL_PREDS or len(g.args) < 2:
                continue
            anchor = g.args[1]
            mask = 0
            for h in self.goals:
                if (h.goal_idx != g.goal_idx and not h.neg
                        and h.predicate in _SD_LOC_PREDS and len(h.args) >= 1
                        and h.args[0] == anchor):
                    hbit = self._goal_idx_to_bit.get(h.goal_idx, -1)
                    if hbit >= 0:
                        mask |= (1 << hbit)
            if mask:
                self._goal_precede_mask[g.goal_idx] = mask

        if self.debug:
            print(f"[BehSDBetaSM] built: {len(self.states)} states, "
                  f"{sum(self.accept_mask)} accept, "
                  f"{sum(len(v) for v in self.transitions.values())} edges, "
                  f"goals_kept={len(self.goals) - len(dropped)}/"
                  f"{len(self.goals)}")

    # ------------------------------------------------------------------
    # Public API (matches BehBetaStateMachine for BetaBackwardDP)
    # ------------------------------------------------------------------

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
        """Apply a step at runtime.

        Accepts either a BehSDStep (with `.args`) or a MetaStep (with `.ids`)
        so callers from either side of the pipeline work uniformly. Mirrors
        the same goal-bit and open-bit semantics that `_build_for_goals`
        used during BFS, including the `inside`-precondition gate (returns
        None if container is closed, which the DP treats as -inf).
        """
        if not (0 <= state_id < len(self.states)):
            return None
        src = self.states[state_id]
        pred, neg = _action_token_to_pred(step.action)
        raw_args = getattr(step, 'args', None)
        if raw_args is None:
            raw_args = getattr(step, 'ids', ())
        args = tuple(raw_args) if hasattr(raw_args, '__iter__') \
            else (raw_args,)

        # Reject a binary relation of an object with itself (physical
        # constraint). Mirrors the build-time filter in
        # _build_steps_for_goal_set.
        if len(args) == 2 and args[0] == args[1]:
            return None

        # inside-precondition: require container's open bit to be set.
        if pred == 'inside' and not neg and len(args) >= 2:
            cont_cid = args[1]
            bit = self._cid_to_open_bit.get(cont_cid)
            if bit is not None and not (src.open_bits & (1 << bit)):
                return None  # precondition fail → β-DP sees -inf

        # Any predicate that touches X requires X's initial container to be
        # open (simulator constraint: an action on an object checks that no
        # enclosing container is closed). `open` itself is excluded: opening
        # a container does not require it to already be accessible.
        if not neg and len(args) >= 1 and pred != 'open':
            src_need = self._cid_to_src_open_need.get(args[0], 0)
            if src_need and (src.open_bits & src_need) != src_need:
                return None  # source-container closed → β-DP sees -inf

        # Clean-chain precondition + effect (mirrors _build_steps_for_goal_set).
        chain_need = 0
        chain_set = 0
        if self._clean_chain_active:
            if (pred in _SD_HAND_PREDS and not neg and len(args) == 1
                    and args[0] in self._cleaner_cids):
                chain_set |= 0b0001
            # Bit 1 is unused: `soaked(<cleaner>)` sets nothing. A cleaner
            # counts as wet once staged in a running water source (bits 2+3).
            elif (pred == 'toggledon' and not neg and len(args) == 1
                    and args[0] in self._water_source_cids):
                chain_set |= 0b0100
            elif (pred == 'inside' and not neg and len(args) >= 2
                    and args[1] in self._water_source_cids):
                # Passive sources need no toggle: staging inside completes
                # bits 2+3.
                if args[1] in self._passive_water_cids:
                    chain_set |= 0b1100
                else:
                    chain_set |= 0b1000
            if pred == 'dusty' and neg and len(args) >= 1:
                chain_need |= 0b0001
            elif pred == 'stained' and neg and len(args) >= 1:
                # `not stained(X)` requires cleaner-held (bit 0) AND
                # cleaner-staged-in-running-water (bits 2+3), i.e. a wet
                # cleaner (evaluator precondition).
                chain_need |= 0b1101
            elif pred == 'soaked' and not neg and len(args) >= 1:
                chain_need |= 0b1100
        if chain_need and (src.clean_chain_bits & chain_need) != chain_need:
            return None  # clean-chain precondition fail

        # Anchor-ordering precedence: a relation goal R(X, C) may only be set
        # after every reposition goal of its anchor C is satisfied (else the
        # evaluator moves C afterward and breaks the relation).
        idx = self._goal_key_to_idx.get((pred, neg, args))
        if idx is not None:
            pmask = self._goal_precede_mask.get(idx, 0)
            if pmask and (src.goal_bits & pmask) != pmask:
                return None  # anchor not yet repositioned → β-DP sees -inf

        # goal-bit update
        bit_pos = self._goal_idx_to_bit.get(idx, -1) if idx is not None else -1
        if bit_pos < 0:
            new_goal_bits = src.goal_bits
        else:
            new_goal_bits = src.goal_bits | (1 << bit_pos)

        # open-state update (open / not_open atoms)
        new_open_bits = src.open_bits
        if pred == 'open' and len(args) >= 1:
            cid = args[0]
            bit = self._cid_to_open_bit.get(cid)
            if bit is not None:
                if neg:
                    new_open_bits &= ~(1 << bit)
                else:
                    new_open_bits |= (1 << bit)

        new_chain_bits = src.clean_chain_bits | chain_set

        next_id = self.state_to_id.get(
            BehSDBetaState(goal_bits=new_goal_bits, open_bits=new_open_bits,
                           clean_chain_bits=new_chain_bits)
        )
        # Destination not enumerated ⇒ atom touches only dropped goal bits
        # (or otherwise has no representable effect in the coarse β under
        # goal-drop). Identity is the sound coarsening. Precondition fails
        # earlier in this method explicitly return None for NEG_INF.
        # Self-loops at non-accept states are allowed: this β tracks only
        # goal/open/clean-chain bits, so many valid scaffolding subgoals have
        # no abstract state change.
        return state_id if next_id is None else next_id


# ----------------------------------------------------------------------
# Prompt parser — extracts `## Initial States` and `## Goal States` blocks.
# Returns a structured AST that `_parse_goals` can walk:
#   ('atom', pred, neg, [arg1, arg2, ...])
#   ('and', [children])
#   ('or', [children])
#   ('not', child)
#   ('forall', VAR, body)
#   ('exists', VAR, body)
#   ('forpairs', VAR_A, VAR_B, body)
#   ('forn', N, VAR, body)
# ----------------------------------------------------------------------

def parse_beh_sd_states(prompt: str) -> Tuple[List[Tuple], List[Tuple]]:
    """Parse a BEHAVIOR SD prompt's `## Initial States` and `## Goal States`
    sections into structured ASTs (shapes listed in the comment above).

    Robust to:
      - Whitespace / newlines inside parens
      - Nested `and`/`not`
      - `forall(VAR, body)` and `exists(VAR, body)` with arbitrary body shape
    """
    import re

    def _extract_section(text: str, header: str) -> str:
        m = re.search(rf'##\s*{re.escape(header)}\s*\n(.*?)'
                      r'(?=\n##\s|\n## Output|\Z)',
                      text, re.DOTALL)
        return m.group(1) if m else ''

    def _balanced_split(s: str, sep: str) -> List[str]:
        """Split `s` on top-level `sep` (paren-aware)."""
        parts: List[str] = []
        depth = 0
        cur = []
        i = 0
        sep_len = len(sep)
        while i < len(s):
            ch = s[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and s[i:i + sep_len] == sep:
                parts.append(''.join(cur))
                cur = []
                i += sep_len
                continue
            cur.append(ch)
            i += 1
        parts.append(''.join(cur))
        return [p.strip() for p in parts if p.strip()]

    def _strip_outer_parens(s: str) -> str:
        s = s.strip()
        while s.startswith('(') and s.endswith(')'):
            # Verify the leading `(` matches the trailing `)`
            depth = 0
            for i, ch in enumerate(s):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        if i == len(s) - 1:
                            s = s[1:-1].strip()
                            break
                        return s
            else:
                break
        return s

    def _parse_expr(s: str) -> Optional[Tuple]:
        """Parse a goal expression into an AST tuple, or None if unparseable."""
        s = _strip_outer_parens(s)
        if not s:
            return None

        # Top-level `and` / `or` splits.
        for op in (' and ', ' or '):
            parts = _balanced_split(s, op)
            if len(parts) > 1:
                children = [_parse_expr(p) for p in parts]
                children = [c for c in children if c is not None]
                if len(children) == 1:
                    return children[0]
                if children:
                    return (op.strip(), children)
                return None

        # `not <expr>`
        m = re.match(r'^not\s+(.*)$', s, re.DOTALL)
        if m:
            inner = _parse_expr(m.group(1))
            if inner is None:
                return None
            return ('not', inner)

        # `forall(VAR, body)` / `exists(VAR, body)`
        m = re.match(r'^(forall|exists)\s*\((.*)\)\s*$', s, re.DOTALL)
        if m:
            head = m.group(1)
            inner = m.group(2).strip()
            # Split on the first top-level comma — VAR, body
            parts = _balanced_split(inner, ',')
            if len(parts) < 2:
                return None
            var = parts[0].strip()
            body_str = ','.join(parts[1:])
            body = _parse_expr(body_str)
            if body is None:
                return None
            return (head, var, body)

        # `forpairs(VAR_A, VAR_B, body)` — body references both vars.
        m = re.match(r'^forpairs\s*\((.*)\)\s*$', s, re.DOTALL)
        if m:
            parts = _balanced_split(m.group(1), ',')
            if len(parts) < 3:
                return None
            body = _parse_expr(','.join(parts[2:]))
            if body is None:
                return None
            return ('forpairs', parts[0].strip(), parts[1].strip(), body)

        # `forn(N, VAR, body)` — N instances of VAR satisfy body.
        m = re.match(r'^forn\s*\((.*)\)\s*$', s, re.DOTALL)
        if m:
            parts = _balanced_split(m.group(1), ',')
            if len(parts) < 3:
                return None
            try:
                n = int(parts[0].strip())
            except ValueError:
                return None
            body = _parse_expr(','.join(parts[2:]))
            if body is None:
                return None
            return ('forn', n, parts[1].strip(), body)

        # `pred(arg1, arg2, ...)`
        m = re.match(r'^(\w+)\s*\((.*)\)\s*$', s, re.DOTALL)
        if m:
            pred = m.group(1)
            args = [a.strip() for a in _balanced_split(m.group(2), ',')]
            return ('atom', pred, False, args)

        return None

    def _parse_block(text: str) -> List[Tuple]:
        """Parse a section's body — newline-separated, but respect parens."""
        out: List[Tuple] = []
        # Reflow: collapse a continuation line into the previous one if parens
        # are still unbalanced. Then split on top-level newlines.
        depth = 0
        cur: List[str] = []
        for line in text.splitlines():
            for ch in line:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            cur.append(line)
            if depth == 0:
                expr = ' '.join(cur).strip()
                cur = []
                if not expr:
                    continue
                ast = _parse_expr(expr)
                if ast is not None:
                    out.append(ast)
        # Trailing buffer (unbalanced — likely malformed; ignore)
        return out

    initial = _parse_block(_extract_section(prompt, 'Initial States'))
    goals = _parse_block(_extract_section(prompt, 'Goal States'))
    return initial, goals

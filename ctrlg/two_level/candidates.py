"""Candidate trackers: the grounded actions (or atoms) a partial token block can still
become, for VirtualHome AS and SD. The BEHAVIOR trackers are in beh_candidates.py and
beh_sd_candidates.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import time

from .tau import MetaStep, _unwrap_ws_state


def _is_dead(gamma_dfa, state) -> bool:
    return state == getattr(gamma_dfa, "DEAD", ("DEAD",))


@dataclass(frozen=True)
class Candidate:
    step: MetaStep
    # The candidate's single canonical token witness: action tokens followed by
    # the object-name (and, where the format has them, id) tokens of each role.
    content_tokens: Tuple[int, ...]

    @property
    def content_variants(self) -> Tuple[Tuple[int, ...], ...]:
        """The canonical witness as a one-element tuple (empty if there is none)."""
        return (self.content_tokens,) if self.content_tokens else tuple()


class _BaseCandidateTracker:
    """Tracks the candidate set A_t of grounded actions within a single gamma action
    block, using one deterministic token witness per candidate, and groups A_t by
    the next required content token.

    Punctuation and whitespace tokens are "glue" and leave the candidates unchanged;
    only content tokens (those consumed inside the action/name/id tries) shrink A_t.
    Scoring by the semantic automaton is done by the processor, not here.
    """

    _SENTINEL_NEXT_TOKEN = -1

    def __init__(self, *, tokenizer, gamma_dfa, tau_parser):
        self.tokenizer = tokenizer
        self.gamma_dfa = gamma_dfa
        self.tau = tau_parser

        self.gamma_state = self.gamma_dfa.initial()
        self._in_block = False
        self._content_pos = 0
        self._cands: List[Candidate] = []
        # In-block content tokens that matched no candidate (guarantee violations).
        self.content_violations = 0
        # Groups by next content token, rebuilt only when A_t changes.
        self._groups_version: int = 0
        self._groups_steps: Dict[int, List[MetaStep]] = {}
        self._groups_candidates: Dict[int, List[Candidate]] = {}
        # Set whenever A_t changes.
        self._cands_dirty: bool = False
        # Timing counters (seconds).
        self.perf_tau_s: float = 0.0
        self.perf_cand_s: float = 0.0
        self.perf_groups_s: float = 0.0
        # Snapshot of the tracker state at the last emitted action.
        self._last_A_before_emit: Optional[List[MetaStep]] = None
        self._last_num_candidates_before_emit: Optional[int] = None
        self._last_groups_count_before_emit: Optional[int] = None
        self._last_T_count_before_emit: Optional[int] = None
        self._last_emitted_step: Optional[MetaStep] = None
        self._last_emit_prev_phase: Optional[str] = None
        self._last_emit_new_phase: Optional[str] = None

    def reset(self) -> None:
        self.gamma_state = self.gamma_dfa.initial()
        self.tau.reset()
        self._in_block = False
        self._content_pos = 0
        self._cands = []
        self._groups_version = 0
        self._groups_steps = {}
        self._groups_candidates = {}
        self._cands_dirty = False
        self.perf_tau_s = 0.0
        self.perf_cand_s = 0.0
        self.perf_groups_s = 0.0
        self._last_A_before_emit = None
        self._last_num_candidates_before_emit = None
        self._last_groups_count_before_emit = None
        self._last_T_count_before_emit = None
        self._last_emitted_step = None
        self._last_emit_prev_phase = None
        self._last_emit_new_phase = None

    def _is_content_token(self, prev_base: tuple, new_base: tuple) -> bool:
        raise NotImplementedError

    def _expects_content(self, state) -> bool:
        """True if gamma allows, at `state`, the next content token of some candidate."""
        nxt = [t for t in self._groups_candidates if t >= 0]
        if not nxt:
            return False
        allowed = self.gamma_dfa.allowed_tokens(state)
        return any(t in allowed for t in nxt)

    def _build_candidates(self, steps: Sequence[MetaStep]) -> List[Candidate]:
        raise NotImplementedError

    def begin_block(self, steps: Sequence[MetaStep]) -> None:
        """Begin a block whose candidates are the given steps."""
        self._in_block = True
        self._content_pos = 0
        self._cands = self._build_candidates(list(steps))
        self._rebuild_groups()
        self._cands_dirty = True
    
    def begin_block_from_gamma(self) -> None:
        """Begin a block whose candidates are all grounded actions gamma allows.
        Filtering by the semantic automaton is applied afterwards through
        `filter_candidates`."""
        self._in_block = True
        self._content_pos = 0

        all_steps = self._enumerate_gamma_valid_steps()
        self._cands = self._build_candidates(list(all_steps))
        self._rebuild_groups()

    def filter_candidates(self, keep_fn) -> int:
        """Filter the current candidate set in-place to those for which
        `keep_fn(candidate) is True`, then rebuild groups so that
        `groups_by_next_token_candidates()` reflects only the kept set.

        The processor calls this at block start with
            keep_fn = lambda c: beta.apply(state, c.step) is not None,
        so every remaining candidate has a beta transition and, at each content
        position, the grouped next tokens are exactly the beta-admissible
        continuations; gamma-valid tokens outside them are masked by the processor.

        Returns the number of candidates kept.
        """
        kept = [c for c in self._cands if keep_fn(c)]
        self._cands = kept
        self._rebuild_groups()
        self._cands_dirty = True
        return len(kept)
        self._cands_dirty = True
    
    def _enumerate_gamma_valid_steps(self) -> List[MetaStep]:
        """All grounded actions gamma allows; implemented per output format."""
        raise NotImplementedError("Subclass must implement _enumerate_gamma_valid_steps")

    def end_block(self) -> None:
        self._in_block = False
        self._content_pos = 0
        self._cands = []
        self._groups_steps = {}
        self._groups_candidates = {}
        self._groups_version += 1
        self._cands_dirty = False

    def A_t(self) -> List[MetaStep]:
        return [c.step for c in self._cands]

    def num_candidates(self) -> int:
        return int(len(self._cands))

    def last_A_before_emit(self) -> Optional[List[MetaStep]]:
        return self._last_A_before_emit

    def last_num_candidates_before_emit(self) -> Optional[int]:
        return self._last_num_candidates_before_emit

    def last_groups_count_before_emit(self) -> Optional[int]:
        return self._last_groups_count_before_emit

    def last_T_count_before_emit(self) -> Optional[int]:
        return self._last_T_count_before_emit

    def last_emitted_step(self) -> Optional[MetaStep]:
        return self._last_emitted_step

    def last_emit_phases(self) -> Tuple[Optional[str], Optional[str]]:
        return self._last_emit_prev_phase, self._last_emit_new_phase

    def in_block(self) -> bool:
        return bool(self._in_block)

    def groups_version(self) -> int:
        return int(self._groups_version)

    def _rebuild_groups(self) -> None:
        """Group the current candidates by their token at content_pos.
        Called only at block start and after a content token."""
        t0 = time.perf_counter()
        pos = int(self._content_pos)

        groups_steps: Dict[int, List[MetaStep]] = {}
        groups_cands: Dict[int, List[Candidate]] = {}

        for c in self._cands:
            if pos < len(c.content_tokens):
                tok = int(c.content_tokens[pos])
            else:
                # No remaining content tokens: sentinel bucket.
                tok = int(self._SENTINEL_NEXT_TOKEN)

            groups_cands.setdefault(tok, []).append(c)
            bucket = groups_steps.setdefault(tok, [])
            if c.step not in bucket:
                bucket.append(c.step)

        self._groups_steps = groups_steps
        self._groups_candidates = groups_cands
        self._groups_version += 1
        self.perf_groups_s += float(time.perf_counter() - t0)

    def groups_by_next_token(self) -> Dict[int, List[MetaStep]]:
        """Partition A_t by the next required content token. Candidates whose
        witness is fully consumed go under the sentinel key -1."""
        return dict(self._groups_steps)

    def consume_cands_dirty(self) -> bool:
        """Return and clear candidate dirty flag."""
        dirty = self._cands_dirty
        self._cands_dirty = False
        return dirty

    def content_pos(self) -> int:
        return int(self._content_pos)

    def groups_by_next_token_candidates(self) -> Dict[int, List[Candidate]]:
        """Like `groups_by_next_token`, but with full Candidate objects so callers
        can access each candidate's witness tokens."""
        return dict(self._groups_candidates)

    def _apply_content_token(self, tok_id: int) -> None:
        # A_t <- groups[tok_id], without rescanning all candidates.
        pos = int(self._content_pos)
        self._cands = list(self._groups_candidates.get(int(tok_id), []))
        self._content_pos = pos + 1
        self._rebuild_groups()
        self._cands_dirty = True

    def advance(self, tok_id: int) -> Optional[MetaStep]:
        """Consume one token: step gamma, the parser tau and the candidates.
        Returns the grounded action if a block completes at this token."""
        prev_state = self.gamma_state
        self.gamma_state = self.gamma_dfa.step(self.gamma_state, int(tok_id))
        if _is_dead(self.gamma_dfa, self.gamma_state):
            try:
                tok_decoded = self.tokenizer.decode([int(tok_id)]) if self.tokenizer else f"<{tok_id}>"
            except Exception:
                tok_decoded = f"<{tok_id}>"
            prev_phase = getattr(self.gamma_dfa, "get_phase", lambda s: str(prev_state))(prev_state)
            allowed = getattr(self.gamma_dfa, "allowed", lambda s: set())(prev_state)
            was_allowed = int(tok_id) in allowed
            raise ValueError(
                f"CandidateTracker advanced into DEAD γ state (prefix not γ-valid). "
                f"Token {tok_id} ({tok_decoded!r}), prev_phase={prev_phase}, was_γ_allowed={was_allowed}, "
                f"allowed_count={len(allowed)}"
            )

        # tau emits a grounded action when a block completes.
        t_tau0 = time.perf_counter()
        adv = getattr(self.tau, "advance_with_states", None)
        if callable(adv):
            emitted = adv(tok_id=int(tok_id), prev_state=prev_state, new_state=self.gamma_state)
        else:
            emitted = self.tau.advance(int(tok_id))
        self.perf_tau_s += float(time.perf_counter() - t_tau0)

        prev_base, _ = _unwrap_ws_state(prev_state)
        new_base, _ = _unwrap_ws_state(self.gamma_state)
        if not isinstance(prev_base, tuple) or not isinstance(new_base, tuple):
            return emitted

        # On emission, record a snapshot of the block and clear the candidates.
        if emitted is not None:
            self._last_A_before_emit = [c.step for c in self._cands]
            self._last_num_candidates_before_emit = int(len(self._cands))
            self._last_groups_count_before_emit = int(len(self._groups_steps))
            self._last_T_count_before_emit = int(len([k for k in self._groups_steps.keys() if int(k) >= 0]))
            self._last_emitted_step = emitted
            self._last_emit_prev_phase = getattr(self.gamma_dfa, "get_phase", lambda s: prev_base[0] if isinstance(prev_base, tuple) and prev_base else None)(prev_state)
            self._last_emit_new_phase = getattr(self.gamma_dfa, "get_phase", lambda s: new_base[0] if isinstance(new_base, tuple) and new_base else None)(self.gamma_state)
            self.end_block()
            return emitted

        # Candidate updates happen only on content tokens *within a block*.
        if self._in_block:
            if self._is_content_token(prev_base, new_base):
                # Only shrink candidates if this token is a valid next content token.
                if int(tok_id) in self._groups_candidates:
                    t_c0 = time.perf_counter()
                    self._apply_content_token(int(tok_id))
                    self.perf_cand_s += float(time.perf_counter() - t_c0)
                elif self._expects_content(prev_state):
                    # gamma allowed a group token here and another content token
                    # was written: the block no longer matches any candidate.
                    self.content_violations += 1
                    self._cands = []
                    self._rebuild_groups()
                # Otherwise it is glue (a quote or bracket that lands in a content
                # phase but is not part of the candidates' content): absorbed.
            return None

        # Outside a block candidates are not created here; the caller begins the block.
        return None


class SubgoalDecompCandidateTracker(_BaseCandidateTracker):
    """Candidate tracking for the VirtualHome Subgoal Decomposition output format.

    Content tokens are those landing in a trie phase (PRED_NAME, ARG_NAME,
    ARG_ID), including the token that enters the trie.
    """

    def __init__(
        self,
        *,
        tokenizer,
        gamma_dfa,
        tau_parser,
        canonical_id_to_obj: List[Tuple[str, str]],
        gamma_valid_predicates: List[Tuple[str, Tuple[int, ...]]],
    ):
        super().__init__(tokenizer=tokenizer, gamma_dfa=gamma_dfa, tau_parser=tau_parser)
        self.canonical_id_to_obj = [(str(n), str(oid)) for (n, oid) in (canonical_id_to_obj or [])]
        self.gamma_valid_predicates = gamma_valid_predicates

    def _enumerate_gamma_valid_steps(self) -> List[MetaStep]:
        """All gamma-valid predicates as MetaSteps."""
        return [MetaStep(action=pred, ids=cids) for pred, cids in self.gamma_valid_predicates]

    def _is_content_token(self, prev_base: tuple, new_base: tuple) -> bool:
        new_p = new_base[0] if isinstance(new_base, tuple) and new_base else str(new_base)
        # The tok_id-in-groups check in advance() keeps structural tokens that land
        # in a trie phase (e.g. '(' landing in ARG_NAME) from advancing content_pos.
        return str(new_p) in {"PRED_NAME", "ARG_NAME", "ARG_ID"}

    def _build_candidates(self, steps: Sequence[MetaStep]) -> List[Candidate]:
        """Witness = predicate tokens + (name, '.', id tokens) per argument,
        from gamma's precomputed token sequences."""
        out: List[Candidate] = []
        dfa = self.gamma_dfa

        for st in steps:
            pred = str(st.action).upper()

            # Predicate name tokens
            pred_seq = getattr(dfa, "pred_token_seqs", {}).get(pred)
            if pred_seq is None:
                continue  # skip predicates not in trie

            content_tokens: List[int] = list(pred_seq)

            # Argument tokens: name + id per arg
            for cid in st.ids:
                if not (0 <= int(cid) < len(self.canonical_id_to_obj)):
                    continue
                name, oid = self.canonical_id_to_obj[int(cid)]

                name_seq = getattr(dfa, "name_token_seqs", {}).get(str(name))
                id_seq = getattr(dfa, "id_token_seqs", {}).get((str(name), str(oid)))
                if name_seq is None or id_seq is None:
                    continue
                content_tokens.extend(list(name_seq))
                # The '.' between name and id is a content token, so when one
                # object name is a prefix of another (toothbrush / toothbrush_holder)
                # the semantic term can choose between ending the name ('.') and
                # continuing it. The '.' lands in ARG_ID, a content phase.
                content_tokens.extend(list(getattr(dfa, "dot_seq", ()) or ()))
                content_tokens.extend(list(id_seq))

            out.append(Candidate(step=st, content_tokens=tuple(int(x) for x in content_tokens)))
        return out


class VirtualHomeCandidateTracker(_BaseCandidateTracker):
    """Candidate tracking for the VirtualHome Action Sequencing output format.

    Content tokens are those landing in EXPECT_ACTION / EXPECT_NAME / EXPECT_ID
    (the action, name and id tries) and in the fixed name-to-id separator.
    """

    def __init__(
        self,
        *,
        tokenizer,
        gamma_dfa,
        tau_parser,
        canonical_id_to_obj: List[Tuple[str, str]],
    ):
        super().__init__(tokenizer=tokenizer, gamma_dfa=gamma_dfa, tau_parser=tau_parser)
        self.canonical_id_to_obj = [(str(n), str(oid)) for (n, oid) in (canonical_id_to_obj or [])]
        # In `["name", "id"]` the text between a name and its id is always `", "`
        # (unlike the context-dependent id terminator `"]` / `", ["`). Treating it
        # as part of the witness, like the '.' in Subgoal Decomposition, lets the
        # semantic term choose between prefix-sibling names (bed / bedroom).
        from ..gamma.topdown.trie_builder import canonical_tokenization
        self._name_id_sep = tuple(canonical_tokenization(tokenizer, '", "'))

    def _enumerate_gamma_valid_steps(self) -> List[MetaStep]:
        """All grounded actions gamma allows: actions from `action_arity`, objects
        per role filtered by `allowed_objects_per_role[(action, role)]` (object
        affordances, e.g. GRAB only on GRABBABLE objects). Actions that do not
        advance the goal are still included."""
        steps: List[MetaStep] = []
        
        dfa = self.gamma_dfa
        action_arity = getattr(dfa, 'action_arity', {})
        allowed_objects_per_role = getattr(dfa, 'allowed_objects_per_role', {})
        
        actions = list(action_arity.keys())
        if not actions and hasattr(dfa, 'canonical_token_seq'):
            actions = list(dfa.canonical_token_seq.keys())
        
        name_to_cids: Dict[str, List[int]] = {}
        for cid, (name, oid) in enumerate(self.canonical_id_to_obj):
            name_to_cids.setdefault(str(name).lower(), []).append(cid)
        
        def get_allowed_cids(action: str, role_idx: int) -> List[int]:
            """Canonical ids allowed for (action, role_idx) by the affordance constraints."""
            key = (str(action).upper(), int(role_idx))
            allowed_names = allowed_objects_per_role.get(key, None)
            if allowed_names is None:
                return list(range(len(self.canonical_id_to_obj)))
            result = []
            for name in allowed_names:
                result.extend(name_to_cids.get(str(name).lower(), []))
            return result
        
        for action in actions:
            action_u = str(action).upper()
            arity = int(action_arity.get(action_u, 1))
            
            if arity == 0:
                steps.append(MetaStep(action=action_u, ids=()))
            elif arity == 1:
                allowed_cids = get_allowed_cids(action_u, 0)
                for cid in allowed_cids:
                    steps.append(MetaStep(action=action_u, ids=(cid,)))
            else:  # arity >= 2
                allowed_cids_0 = get_allowed_cids(action_u, 0)
                allowed_cids_1 = get_allowed_cids(action_u, 1)
                for cid1 in allowed_cids_0:
                    for cid2 in allowed_cids_1:
                        steps.append(MetaStep(action=action_u, ids=(cid1, cid2)))

        return steps

    def _is_content_token(self, prev_base: tuple, new_base: tuple) -> bool:
        new_p = getattr(self.gamma_dfa, "get_phase", lambda s: new_base[0])(new_base)
        # Trie phases, plus the phases of the `", "` name-to-id separator (part of
        # the witness) so content_pos advances through it. The id terminator and
        # id-to-name separator are glue. The tok_id-in-groups check in advance()
        # keeps structural tokens from advancing content_pos.
        return str(new_p) in {
            "EXPECT_ACTION", "EXPECT_NAME", "EXPECT_ID",
            "VALUE_QUOTE_CLOSE_NAME", "NAME_ID_SEP", "VALUE_QUOTE_OPEN_ID",
        }

    def _build_candidates(self, steps: Sequence[MetaStep]) -> List[Candidate]:
        """Witness = action tokens + (name, `", "`, id tokens) per argument."""
        out: List[Candidate] = []
        for st in steps:
            act = str(st.action).upper()
            canonical_seq = getattr(self.gamma_dfa, "canonical_token_seq", {}).get(act, None)
            if canonical_seq is None:
                act_vars = getattr(self.gamma_dfa, "action_token_seqs_by_action", {}).get(act, [])
                if not act_vars:
                    raise RuntimeError(
                        f"γ DFA missing canonical tokenization for action '{act}'."
                    )
                canonical_seq = act_vars[0]

            content_tokens: List[int] = list(canonical_seq)
            for cid in st.ids:
                if not (0 <= int(cid) < len(self.canonical_id_to_obj)):
                    continue
                name, oid = self.canonical_id_to_obj[int(cid)]
                # gamma's precomputed sequences avoid tokenizer calls per decoding step.
                name_seq = getattr(self.gamma_dfa, "name_token_seqs_by_name", {}).get(str(name))
                id_seq = getattr(self.gamma_dfa, "id_token_seqs_by_name_and_id", {}).get((str(name), str(oid)))
                if name_seq is None or id_seq is None:
                    raise RuntimeError(
                        "γ DFA missing name/id token sequences; would require tokenizer.encode() inside the per-token processor."
                    )
                content_tokens.extend(list(name_seq))
                content_tokens.extend(list(self._name_id_sep))
                content_tokens.extend(list(id_seq))

            out.append(Candidate(step=st, content_tokens=tuple(int(x) for x in content_tokens)))
        return out



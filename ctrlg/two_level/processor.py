"""Two-level constrained decoding (Eq. 7 of the paper) as a HuggingFace logits processor.

At every step the LM's logits are masked by the syntax automaton (gamma) and,
inside an action block, reweighted by the semantic term

    factor[y] = log sum_{c in G(y)} p_meta(c, accept | a_<l) - log sum_{c in G(y)} p_meta(c | a_<l)

where G(y) are the grounded actions the partial block can still become after
token y. Both quantities come from the meta-token HMM and the backward DP over
the semantic automaton (beta); actions that beta rejects stay in the
denominator with zero numerator. At a block boundary the LM chooses between
stopping and continuing; stopping has weight 1 if beta accepts and 0 otherwise,
continuing has the expected semantic term of the next block.

Dead ends. If no live candidate has mass (inside a block, or for the block that
continuing would open while beta does not accept), the block is decoded with a
precondition-only mask: every candidate that is a valid beta transition from the
current state is allowed, ignoring goal reachability and HMM weights, and the
LM's logits choose. If no candidate is a valid transition, the plan is closed.
Every committed action is therefore a valid beta transition; a rejected commit
or an untracked content token is counted as a violation (and asserted with
debug=True).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import LogitsProcessor

from ..common.constants import NEG_INF
from .tau import MetaStep
from .token_mask import TokenMask


# get_stats() counters of the guarantee machinery (see the module docstring).
GUARANTEE_STATS = ("dead_end_blocks", "precondition_only_steps", "forced_closes", "violations")


class _Step:
    """Action adapter with .action / .args, as the semantic automata expect."""
    __slots__ = ("action", "args")

    def __init__(self, action, args):
        self.action = action
        self.args = args


def _key(c) -> Tuple:
    return (c.step.action, c.step.ids)


def _lse(values: List[float]) -> float:
    return torch.logsumexp(torch.tensor(values, dtype=torch.float64), 0).item()


def meta_factors(groups: Dict[int, List], joint: Dict[Tuple, float], prior: Dict[Tuple, float],
                 extra_den: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    """Semantic term per next-token group: log sum joint - log sum prior over the
    group's candidates. `extra_den` adds the prior mass of beta-rejected actions."""
    out: Dict[int, float] = {}
    for tok, cands in groups.items():
        if tok < 0:
            continue
        num, den = [], []
        for c in cands:
            pp = prior.get(_key(c))
            if pp is None:
                continue
            den.append(pp)
            pj = joint.get(_key(c), float("-inf"))
            if pj > -1e20:
                num.append(pj)
        if extra_den is not None:
            ed = extra_den.get(tok)
            if ed is not None and ed > -1e20:
                den.append(float(ed))
        if not den:
            continue
        out[tok] = (_lse(num) - _lse(den)) if num else float("-inf")
    return out


class TwoLevelProcessor(LogitsProcessor):
    """Syntax mask over tokens plus semantic lookahead over actions.

    Args:
        dfa_model: compiled syntax automaton over the LM's tokens.
        vocab_size, eos_token_id: the LM's logit width and end-of-text token.
        prompt_ids: prompt tokens (generated tokens follow them).
        gamma_dfa: the syntax automaton as a state machine (phases, closer tokens).
        candidate_tracker: maps the tokens of the current block to the grounded
            actions they can still become.
        beta_lookahead: semantic automaton over grounded actions.
        beta_dp: backward DP of the meta-token HMM times the semantic automaton;
            None disables the lookahead.
        beta_initial_belief: prompt-conditioned initial distribution over the
            meta-token HMM's hidden states (log space); uniform if None.
        ctrlg_scale: lookahead calibration exponent c (Appendix "Calibrating the
            Lookahead"); 1 leaves the decoder uncalibrated.
        block_start_fn: gamma state -> True at the start of an action block.
        eos_gate_phase: gamma phase at which the plan may be closed.
        close_tokens: tokens that close the plan at `eos_gate_phase`.
    """

    def __init__(self, dfa_model, *, vocab_size: int, eos_token_id: int, prompt_ids: List[int],
                 tokenizer=None, gamma_dfa=None, candidate_tracker=None, beta_lookahead=None,
                 beta_dp=None, beta_initial_belief=None, max_meta_budget: int = 20,
                 ctrlg_scale: float = 1.0, block_start_fn=None,
                 eos_gate_phase: str = "BLOCK_BOUNDARY", close_tokens=None,
                 device: str = "cuda", debug: bool = False):
        self.device = device
        self.debug = debug
        self.tokenizer = tokenizer
        self.ctrlg_scale = float(ctrlg_scale)
        self.dfa = dfa_model
        self.gamma_dfa = gamma_dfa
        self.candidate_tracker = candidate_tracker
        self._beta_lookahead = beta_lookahead
        self._beta_dp = beta_dp
        self._mask = TokenMask(dfa_model, prompt_ids, vocab_size, eos_token_id, device)
        self._prompt_ids = list(prompt_ids)
        self._block_start_fn = block_start_fn
        self._eos_gate_phase = eos_gate_phase
        self._close_tokens = close_tokens

        if beta_initial_belief is not None:
            self._beta_belief = beta_initial_belief.to(device)
        elif beta_dp is not None:
            self._beta_belief = torch.full((int(beta_dp.H),), -math.log(int(beta_dp.H)), device=device)
        else:
            self._beta_belief = None
        self._beta_state_id = beta_lookahead.get_initial_state_id() if beta_lookahead is not None else None

        self._meta_budget = max_meta_budget
        self._completed_actions: List[MetaStep] = []
        self._in_block = False
        self._pending_block_start = True
        self._block_prepared = False
        self._block_stats: Optional[Dict[str, Any]] = None
        self._block_steps: List = []
        self._beta_factor: Dict[Tuple, float] = {}
        self._beta_prior: Dict[Tuple, float] = {}
        self._succ_cache: Dict[Tuple, Dict[str, Any]] = {}
        self._pending_succ = None
        self._dead_wits: List[Tuple[Tuple, Tuple[int, ...]]] = []

        self._step_count = 0
        self._block_starts = 0
        self._completions = 0
        self._block_aborts = 0
        self._precond_only = False      # current block uses the precondition-only mask
        self._dead_end_blocks = 0
        self._precondition_only_steps = 0
        self._forced_closes = 0
        self._violations = 0

    def _phase(self):
        if self.gamma_dfa is None or self.candidate_tracker is None:
            return None
        return getattr(self.gamma_dfa, "get_phase", lambda s: None)(self.candidate_tracker.gamma_state)

    # ------------------------------------------------------------------ step
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._step_count += 1

        masked = self._mask(input_ids, scores)
        valid_mask = masked[0] > NEG_INF
        logits = masked + self._compute_factor(input_ids, valid_mask)

        # Inside a block, only tokens that continue a beta-admissible candidate are allowed.
        if self._in_block and self.candidate_tracker is not None:
            group_tokens = set(self.candidate_tracker.groups_by_next_token_candidates().keys()) - {-1}
            at_content = False
            if group_tokens:
                gt = torch.tensor([t for t in group_tokens if 0 <= t < valid_mask.shape[0]],
                                  device=valid_mask.device, dtype=torch.long)
                at_content = bool(valid_mask[gt].any().item()) if gt.numel() else False
            if at_content:
                keep = torch.tensor([t for t in group_tokens if 0 <= t < logits.shape[1]],
                                    device=logits.device, dtype=torch.long)
                canon = torch.full((logits.shape[1],), NEG_INF, device=logits.device)
                if keep.numel():
                    canon[keep] = 0.0
                logits = logits + canon.unsqueeze(0)

        logits = self._gate_eos(logits)

        # Every token masked: close the plan rather than emit an arbitrary token.
        if not bool((logits[0] > -1e20).any()):
            eos_id = getattr(self.tokenizer, "eos_token_id", 128001)
            logits[:, :] = NEG_INF
            logits[:, eos_id] = 0.0
        return logits

    def _compute_factor(self, input_ids: torch.LongTensor, valid_mask: torch.Tensor) -> torch.FloatTensor:
        factor = torch.zeros(input_ids.shape[0], self.dfa.vocab_size, device=input_ids.device)
        if self.candidate_tracker is None:
            return factor
        if input_ids.shape[1] > len(self._prompt_ids):
            emitted = self._advance_tracking(int(input_ids[0, -1].item()))
            if emitted is not None:
                self._handle_completed_action(emitted)
                self._completions += 1
        if self._pending_block_start and not self._block_prepared and self._meta_budget > 0:
            self._prepare_block()
        if self._pending_block_start and self._is_at_action_start():
            if self._meta_budget > 0:
                self._start_new_block()
            else:
                self._pending_block_start = False
        return self._apply_factors(factor, valid_mask)

    # -------------------------------------------------------------- tracking
    def _advance_tracking(self, token_id: int) -> Optional[MetaStep]:
        pos_before = self.candidate_tracker.content_pos() if self._in_block else None
        n_viol = getattr(self.candidate_tracker, "content_violations", 0)
        try:
            emitted = self.candidate_tracker.advance(token_id)
        except ValueError as e:
            if self.debug:
                print(f"[TwoLevelProcessor] tracker error: {e}")
            return None
        if getattr(self.candidate_tracker, "content_violations", 0) > n_viol:
            self._violation(f"content token {token_id} matches no candidate")
        if self._in_block and pos_before is not None and self._dead_wits:
            if self.candidate_tracker.content_pos() == pos_before + 1:
                self._dead_wits = [(k, w) for (k, w) in self._dead_wits
                                   if len(w) > pos_before and w[pos_before] == token_id]
        return emitted

    def _handle_completed_action(self, step: MetaStep) -> None:
        self._pending_succ = self._succ_cache.get(self._succ_key(step))
        if self._beta_lookahead is not None and self._beta_state_id is not None:
            new_state = self._beta_lookahead.apply(self._beta_state_id, _Step(step.action, step.ids))
            if new_state is None:  # impossible: the in-block mask only allows live candidates
                self._violation(f"semantic automaton rejected completed action {step.action}{tuple(step.ids)}")
                new_state = self._beta_state_id
            self._beta_state_id = new_state
        self._meta_budget = max(0, self._meta_budget - 1)
        self._completed_actions.append(step)
        if (self._beta_dp is not None and self._beta_belief is not None
                and not getattr(self._beta_dp, "hard_mask", False)):
            self._beta_belief = self._beta_dp.posterior_after_step(self._beta_belief, _Step(step.action, step.ids))
        self._in_block = False
        self._pending_block_start = True
        self._precond_only = False
        self._beta_factor = {}
        self._block_prepared = False
        self._block_stats = None
        self._dead_wits = []

    # ---------------------------------------------------------------- blocks
    def _prepare_block(self) -> None:
        """Enumerate the next block's candidates, keep the beta-admissible ones
        live, and compute joint and prior weights for all of them."""
        tracker = self.candidate_tracker
        tracker.begin_block_from_gamma()
        all_cands = list(tracker._cands)
        if (self._beta_lookahead is not None and self._beta_state_id is not None
                and hasattr(tracker, "filter_candidates")):
            sm, sid = self._beta_lookahead, self._beta_state_id
            tracker.filter_candidates(lambda c: sm.apply(sid, _Step(c.step.action, c.step.ids)) is not None)
        live_keys = {_key(c) for c in tracker._cands}
        self._dead_wits = [(_key(c), tuple(c.content_tokens)) for c in all_cands if _key(c) not in live_keys]
        self._block_steps = [c.step for c in all_cands]

        cached, self._pending_succ = self._pending_succ, None
        if cached is not None and cached.get("state") == self._beta_state_id and cached.get("budget") == self._meta_budget:
            self._beta_factor = dict(cached["joint"])
            self._beta_prior = dict(cached["prior"])
        else:
            self._compute_beta_factors(all_cands)
            self._compute_meta_priors(all_cands)
        self._succ_cache = {}
        self._block_stats = self._aggregate_stats(self._beta_factor, self._beta_prior,
                                                  self._beta_state_id, self._beta_belief)
        if not any(self._beta_factor.get(k, NEG_INF) > -1e20 for k in live_keys):
            accept = self._check_beta_accept()
            if live_keys and not accept:
                # Dead end: no live action can still reach acceptance. Continue
                # with the precondition-only mask (continuing is allowed).
                self._dead_end_blocks += 1
                self._precond_only = True
                self._block_stats["F_cont"] = 0.0
            else:
                # Stop (beta accepts) or nothing is a valid transition: close the plan.
                self._block_aborts += 1
                if not accept:
                    self._forced_closes += 1
                self._pending_block_start = False
                self._meta_budget = 0
        self._block_prepared = True

    def _start_new_block(self) -> None:
        if not self._block_prepared:
            self._prepare_block()
            if not self._pending_block_start:
                return
        self._block_starts += 1
        self._in_block = True
        self._pending_block_start = False

    def _is_at_action_start(self) -> bool:
        if self.gamma_dfa is None:
            return False
        if self._block_start_fn is not None:
            return self._block_start_fn(self.candidate_tracker.gamma_state)
        return self._phase() == "EXPECT_ACTION"

    # --------------------------------------------------------------- weights
    def _compute_beta_factors(self, candidates) -> None:
        """log p_meta(a, accept within the remaining budget | a_<l) per candidate,
        times the calibration exponent; -inf if beta rejects a."""
        if self._beta_lookahead is None or self._beta_state_id is None:
            self._beta_factor = {_key(c): 0.0 for c in candidates}
            return
        if self._beta_dp is None:
            raise RuntimeError("a semantic automaton was given without its backward DP")
        steps = [_Step(c.step.action, c.step.ids) for c in candidates]
        values = self._beta_dp.compute_beta_factors_batch(
            belief=self._beta_belief, steps=steps, state_id=self._beta_state_id, budget=self._meta_budget)
        self._beta_factor = {_key(c): (NEG_INF if v <= -1e20 else v * self.ctrlg_scale)
                             for c, v in zip(candidates, values)}

    def _compute_meta_priors(self, candidates) -> None:
        """log p_meta(a | a_<l) per candidate (zero when there is no meta-HMM weighting)."""
        keys = [_key(c) for c in candidates]
        if (self._beta_dp is None or self._beta_lookahead is None or self._beta_state_id is None
                or self._beta_belief is None or getattr(self._beta_dp, "hard_mask", False)):
            self._beta_prior = {k: 0.0 for k in keys}
            return
        steps = [_Step(c.step.action, c.step.ids) for c in candidates]
        pri = self._beta_dp.compute_meta_prior_batch(self._beta_belief, steps, state_id=self._beta_state_id)
        self._beta_prior = {k: float(v) for k, v in zip(keys, pri)}

    def _log_stop_prior(self, belief) -> float:
        """log p_meta(STOP | belief); 0 when the DP does not price stopping."""
        if self._beta_dp is None or belief is None:
            return 0.0
        try:
            return float(self._beta_dp.log_stop_prior(belief))
        except RuntimeError:
            return 0.0

    def _aggregate_stats(self, joint, prior, state_id, belief) -> Dict[str, Any]:
        """Weights for the tokens that end a block:
            F_cont = log sum_a p(a, accept) - log sum_a p(a)       (commits to continuing)
            F_stop = log 1[state accepting]                         (commits to stopping)
            F_open = log (p_stop 1[acc] + sum_a p(a, accept)) - log (p_stop + sum_a p(a))
        """
        fin_j = [v for v in joint.values() if v > -1e20]
        pri = [v for v in prior.values() if v > -1e20]
        lse_j = _lse(fin_j) if fin_j else float("-inf")
        lse_p = _lse(pri) if pri else float("-inf")
        acc = (bool(self._beta_lookahead.is_accept(state_id))
               if (self._beta_lookahead is not None and state_id is not None) else True)
        log_stop = self._log_stop_prior(belief)
        num = torch.logaddexp(torch.tensor(log_stop if acc else float("-inf"), dtype=torch.float64),
                              torch.tensor(lse_j, dtype=torch.float64)).item()
        den = torch.logaddexp(torch.tensor(log_stop, dtype=torch.float64),
                              torch.tensor(lse_p, dtype=torch.float64)).item()
        return {"F_cont": (lse_j - lse_p) if (fin_j and pri) else float("-inf"),
                "F_stop": 0.0 if acc else float("-inf"),
                "F_open": (num - den) if den > -1e20 else float("-inf"),
                "state": state_id, "budget": self._meta_budget}

    def _succ_key(self, step) -> Tuple:
        return (self._beta_state_id, self._meta_budget, str(step.action), tuple(step.ids))

    def _successor_stats(self, step) -> Dict[str, Any]:
        """Boundary weights of the block that would follow `step` (for merged
        tokens that close a block and commit to stop or continue at once)."""
        key = self._succ_key(step)
        if key in self._succ_cache:
            return self._succ_cache[key]
        dead = {"F_cont": float("-inf"), "F_stop": float("-inf"), "F_open": float("-inf"),
                "state": None, "budget": self._meta_budget - 1, "joint": {}, "prior": {}}
        if self._beta_dp is None or self._beta_lookahead is None or self._beta_state_id is None:
            st = dict(dead, F_cont=0.0, F_stop=0.0, F_open=0.0)
            self._succ_cache[key] = st
            return st
        bstep = _Step(step.action, step.ids)
        s_next = self._beta_lookahead.apply(self._beta_state_id, bstep)
        if s_next is None:
            self._succ_cache[key] = dead
            return dead
        post = self._beta_dp.posterior_after_step(self._beta_belief, bstep) if self._beta_belief is not None else None
        budget = self._meta_budget - 1
        steps = [_Step(s.action, s.ids) for s in self._block_steps]
        keys = [(s.action, s.ids) for s in self._block_steps]
        if budget > 0 and steps:
            jl = self._beta_dp.compute_beta_factors_batch(belief=post, steps=steps, state_id=s_next, budget=budget)
            pl = ([0.0] * len(steps) if getattr(self._beta_dp, "hard_mask", False)
                  else self._beta_dp.compute_meta_prior_batch(post, steps, state_id=s_next))
        else:
            jl, pl = [NEG_INF] * len(steps), [0.0] * len(steps)
        joint = {k: (NEG_INF if v <= -1e20 else float(v) * self.ctrlg_scale) for k, v in zip(keys, jl)}
        prior = {k: float(v) for k, v in zip(keys, pl)}
        saved, self._meta_budget = self._meta_budget, budget
        st = self._aggregate_stats(joint, prior, s_next, post)
        self._meta_budget = saved
        if (st["F_cont"] <= -1e20 and budget > 0 and not self._beta_lookahead.is_accept(s_next)
                and any(self._beta_lookahead.apply(s_next, s) is not None for s in steps)):
            # The next block would be a dead end with valid transitions: it is
            # decoded with the precondition-only mask, so continuing stays open.
            st["F_cont"] = st["F_open"] = 0.0
        st.update({"joint": joint, "prior": prior, "state": s_next, "budget": budget})
        self._succ_cache[key] = st
        return st

    def _merged_closer_classes(self, valid_tokens: List[int]) -> Dict[int, str]:
        """Allowed tokens that close the block and also commit to continuing or
        stopping through a BPE merge (e.g. BEHAVIOR `"},` and `"}]`)."""
        alts = getattr(self.gamma_dfa, "_close_dict_alts", None)
        if not alts:
            return {}
        names = {"close_dict_sep": "cont", "close_dict_end": "stop", "close_dict": "open"}
        out = {int(t): names[alts[int(t)]] for t in valid_tokens if alts.get(int(t)) in names}
        return out if any(v in ("cont", "stop") for v in out.values()) else {}

    def _apply_factors(self, factor: torch.FloatTensor, valid_mask: torch.Tensor) -> torch.FloatTensor:
        if not self._block_prepared:
            return factor
        tracker = self.candidate_tracker
        if self._in_block:
            groups = tracker.groups_by_next_token_candidates()
            extra_den = None
            if self._dead_wits:
                pos = tracker.content_pos()
                acc: Dict[int, List[float]] = {}
                for k, w in self._dead_wits:
                    pr = self._beta_prior.get(k)
                    if len(w) > pos and pr is not None and pr > -1e20:
                        acc.setdefault(int(w[pos]), []).append(pr)
                if acc:
                    extra_den = {t: float(torch.logsumexp(torch.tensor(v, dtype=torch.float64), 0))
                                 for t, v in acc.items()}
            mf = meta_factors(groups, self._beta_factor, self._beta_prior, extra_den=extra_den)
            if not self._precond_only and mf and all(v <= -1e20 for v in mf.values()):
                # Dead end inside the block: no live continuation has mass.
                self._dead_end_blocks += 1
                self._precond_only = True
            if self._precond_only:
                # Precondition-only mask: every live candidate (valid beta
                # transition) stays allowed, unweighted; the LM chooses.
                self._precondition_only_steps += 1
                mf = {t: 0.0 for t in groups if t >= 0}
            for tok, f in mf.items():
                if 0 <= tok < factor.shape[1]:
                    factor[:, tok] += f
            classes = self._merged_closer_classes(valid_mask.nonzero(as_tuple=True)[0].tolist())
            bucket = groups.get(-1, [])  # candidates whose tokens are fully written
            if classes:
                # A merged closer ends the action and commits to continuing ("},),
                # stopping ("}]) or deciding later ("}). Continuing needs mass in the
                # next block (or a dead end with valid transitions) and budget left.
                f_cls = {}
                for tok, cls in classes.items():
                    num, den = [], []
                    for c in bucket:
                        pp = 0.0 if self._precond_only else self._beta_prior.get(_key(c))
                        if pp is None or pp <= -1e20:
                            continue
                        x = self._successor_stats(c.step)["F_" + cls]
                        den.append(pp)
                        if x > -1e20:
                            num.append(pp + x)
                    f_cls[tok] = (_lse(num) - _lse(den)) if num else float("-inf")
                if all(v <= -1e20 for v in f_cls.values()):
                    # Nothing can follow this action: only plan-closing tokens.
                    f_cls = {t: (0.0 if classes[t] in ("stop", "open") else float("-inf")) for t in f_cls}
                for tok, f in f_cls.items():
                    factor[:, tok] += f
        elif (self._pending_block_start and self._block_stats is not None
              and self._phase() == self._eos_gate_phase):
            close = (set(self._close_tokens) if self._close_tokens is not None
                     else set(getattr(self.gamma_dfa, "RBRACE_ALTS", [])))
            f_cont, f_stop = self._block_stats["F_cont"], self._block_stats["F_stop"]
            for tok in valid_mask.nonzero(as_tuple=True)[0].tolist():
                factor[:, tok] += f_stop if tok in close else f_cont
        return factor

    # ----------------------------------------------------------- termination
    def _gate_eos(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Close the plan when the action budget is spent; otherwise forbid an
        empty plan, and EOS outside the END phase or before beta accepts."""
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            return logits
        phase = self._phase()
        accept = self._check_beta_accept()
        empty = len(self._completed_actions) == 0
        if self._close_tokens is not None:
            close = set(self._close_tokens)
        elif self.gamma_dfa is not None:
            close = set(getattr(self.gamma_dfa, "RBRACE_ALTS", []))
        else:
            close = set()
        if self._meta_budget <= 0:
            if phase == "END":
                logits[:, :] = NEG_INF
                logits[:, eos_id] = 0.0
            elif phase == self._eos_gate_phase:
                seq = getattr(self.gamma_dfa, "close_list_seq", None)
                term = {seq[0]} if seq else close
                logits[:, :] = NEG_INF
                for tok in term:
                    if tok < logits.shape[1]:
                        logits[:, tok] = 0.0
            return logits
        if empty and phase == self._eos_gate_phase:
            for tok in close:
                if tok < logits.shape[1]:
                    logits[:, tok] = NEG_INF
        if (not accept or empty) and phase != "END":
            logits[:, eos_id] = NEG_INF
        return logits

    def _violation(self, msg: str) -> None:
        """A step the guarantee rules out happened: count it (assert with debug=True)."""
        self._violations += 1
        print(f"[TwoLevelProcessor][VIOLATION] {msg}")
        assert not self.debug, msg

    def _check_beta_accept(self) -> bool:
        if self._beta_lookahead is not None and self._beta_state_id is not None:
            return self._beta_lookahead.is_accept(self._beta_state_id)
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {"step_count": self._step_count, "block_starts": self._block_starts,
                "completions": self._completions, "block_aborts": self._block_aborts,
                "dead_end_blocks": self._dead_end_blocks,
                "precondition_only_steps": self._precondition_only_steps,
                "forced_closes": self._forced_closes, "violations": self._violations,
                "meta_budget": self._meta_budget,
                "completed_actions": len(self._completed_actions), "beta_state": self._beta_state_id}

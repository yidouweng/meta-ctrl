"""
β backward DP for the Meta-Ctrl decoder.

Computes C[B, s, h] = log P(reach β-accept and stop within ≤B actions |
β state s, meta-HMM state h).

Formulation:

1. Action Transition Matrix F_a:
   F_a[h, h'] = log P(emit action tokens AND end at h' | start at h)

   Iterative computation for multi-token actions:
   G^{(0)} = identity (log-space)
   G^{(t)}[h0, ht] = logsumexp_{h_{t-1}} (
       G^{(t-1)}[h0, h_{t-1}] + log P(y_t | h_{t-1}) + log P(ht | h_{t-1})
   )
   F_a = G^{(k)} for k tokens

2. Backward DP Table C[B, s, h]:
   Stop term: stop[s, h] = log p(EOS | h) if s is accepting, else -inf
   Base: C[0, s, h] = stop[s, h]
   Recurrence: C[B, s, h] = logaddexp(
       stop[s, h],  // stop now (accepting states only)
       logsumexp_{a: s→s'} logsumexp_{h'} (F_a[h, h'] + C[B-1, s', h'])
   )
   There is no carry term C[B-1, s, h]: every budget unit is one action.

3. Runtime β_factor:
   β_factor[a] = logsumexp_h (log b(h) + logsumexp_{h'} (F_a[h, h'] + C[B-1, s', h']))

   Note: Use B-1 because this action consumes one budget unit.

Usage:
    from ctrlg.beta.backward_dp import BetaBackwardDP, SemanticVocab

    vocab = SemanticVocab.from_json(vocab_path)
    dp = BetaBackwardDP(state_machine, semantic_hmm, vocab, max_budget=20)

    # At runtime, for the candidate steps of one block
    factors = dp.compute_beta_factors_batch(belief, steps, state_id, budget)
"""
from __future__ import annotations

import json
import math
import torch
from typing import Dict, List, Tuple, Optional

from ..common.constants import NEG_INF


class SemanticVocab:
    """
    Semantic vocabulary for β HMM.

    Maps action names and canonical IDs to token indices.
    Vocabulary structure: action/predicate symbols, canonical-ID symbols
    (ID0, ID1, ...), EOS and UNK.
    """

    def __init__(self, token_to_idx: Dict[str, int], unk_idx: int = None):
        """
        Args:
            token_to_idx: Mapping from token string to index
            unk_idx: Index for unknown tokens (defaults to last index if not specified)
        """
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        self.vocab_size = len(token_to_idx)
        # Set by ctrlg.beta.sd_vocab.load_sd_semantic_vocab for SD meta-HMMs:
        # aliases maps a decoder symbol (e.g. BEH "NEXTTO") to the vocab symbol
        # whose column it reads; strict_actions turns an unknown predicate into
        # a hard error instead of a silent UNK.
        self.aliases: Dict[str, str] = {}
        self.strict_actions: bool = False
        self.source_path: Optional[str] = None

        # Find UNK index
        if unk_idx is not None:
            self.unk_idx = unk_idx
        elif "UNK" in token_to_idx:
            self.unk_idx = token_to_idx["UNK"]
        elif "<unk>" in token_to_idx:
            self.unk_idx = token_to_idx["<unk>"]
        else:
            # Default to last index
            self.unk_idx = self.vocab_size - 1

    @classmethod
    def from_json(cls, path: str) -> "SemanticVocab":
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, dict):
            if "token_to_idx" in data:
                token_to_idx = data["token_to_idx"]
            elif "full_vocab" in data:
                # Semantic vocab format: has "full_vocab" list
                vocab_list = data["full_vocab"]
                token_to_idx = {tok: i for i, tok in enumerate(vocab_list)}
            else:
                # Check if values are lists (not valid as mapping)
                first_val = next(iter(data.values()), None) if data else None
                if isinstance(first_val, list):
                    raise ValueError(f"Unknown vocab format: dict with list values")
                # Assume the dict itself is the mapping
                token_to_idx = data
        elif isinstance(data, list):
            # List of tokens - create index mapping
            token_to_idx = {tok: i for i, tok in enumerate(data)}
        else:
            raise ValueError(f"Unknown vocab format: {type(data)}")

        return cls(token_to_idx)

    def get(self, token: str, default: int = None) -> int:
        """Get index for token, returning default (or unk_idx) if not found."""
        if default is None:
            default = self.unk_idx
        return self.token_to_idx.get(token, default)

    def __len__(self) -> int:
        return self.vocab_size


class BetaBackwardDP:
    """
    Backward DP table for β constraint with correct multi-token action handling.

    Key features:
    - Computes F_a matrices for multi-token actions (proper HMM transitions)
    - Uses ≤B semantics: an accepting state may stop at any remaining budget
    - Provides β_factor computation with belief transformation
    """

    def __init__(
        self,
        state_machine,  # VHBetaStateMachine
        semantic_hmm,   # SemanticHMM with alpha_exp, beta
        semantic_vocab: SemanticVocab,
        max_budget: int = 20,
        device: str = "cuda",
        hard_mask: bool = False,
        uniform_action_logw: Optional[float] = None,
    ):
        """
        Build backward DP table from state machine.

        uniform_action_logw: when set, the surrogate is the *uniform meta-token*
            measure: every grounded action (and STOP) carries the same weight
            exp(uniform_action_logw) = 1/(|W|+1), independent of its arity, and
            hidden states are irrelevant (F_a is the constant matrix
            logw - log H). This is uniformity over candidate actions, i.e. over
            the alphabet the semantic automaton distinguishes.

        Args:
            state_machine: VHBetaStateMachine with states, transitions, accept_mask
            semantic_hmm: Semantic HMM with alpha_exp [H,H], beta [H,V]
            semantic_vocab: SemanticVocab for step -> token mapping
            max_budget: Maximum step budget (meta budget)
            device: Torch device
            hard_mask: ablation that replaces the HMM-weighted lookahead
                (a *measure* over accepting continuations) with pure
                accept-reachability (an *indicator*): every admissible
                action scores 0.0, inadmissible NEG_INF. Same horizon,
                same EOS gating (compute_log_M keeps its reachability
                semantics), same fallback; weighting is the only change.
            An accepting continuation ends by emitting the EOS meta-token once,
            paying the surrogate's own log p(EOS | h); the HMM was trained on
            sequences that end with exactly one EOS.
        """
        self.sm = state_machine
        self.hmm = semantic_hmm
        self.vocab = semantic_vocab
        self.max_budget = max_budget
        self.device = device
        self.hard_mask = hard_mask
        self.uniform_action_logw = (float(uniform_action_logw)
                                    if uniform_action_logw is not None else None)

        # Dimensions
        self.S = state_machine.num_states
        self.H = semantic_hmm.hidden_size
        self.V = len(semantic_vocab)  # Vocab size from SemanticVocab
        self.V_hmm = semantic_hmm.beta.shape[1]  # Actual HMM emission vocab size

        # Extract HMM parameters
        self.alpha_exp = semantic_hmm.alpha_exp.to(device)  # [H, H] transition probs
        self.emission_log = semantic_hmm.beta.to(device)    # [H, V_hmm] log emission probs

        # The vocabulary must index the HMM's emission columns exactly.
        from .sd_vocab import check_vocab_matches_hmm
        check_vocab_matches_hmm(semantic_hmm, semantic_vocab)
        if self.V > self.V_hmm:
            raise ValueError(f"semantic vocab has {self.V} symbols but the HMM only "
                             f"{self.V_hmm} emission columns")

        # Precompute log transition matrix
        self.log_trans = torch.log(self.alpha_exp + 1e-30)  # [H, H]

        # Termination emission log p(EOS | h).
        if self.uniform_action_logw is not None:
            # Uniform meta-token measure: STOP is one more meta-token with the
            # same weight as every grounded action.
            self._log_eos = torch.full((self.H,), self.uniform_action_logw, device=device)
        else:
            eos_idx = self.vocab.token_to_idx.get("EOS")
            if eos_idx is None or eos_idx >= self.emission_log.shape[1]:
                raise ValueError(
                    "the semantic vocab needs an EOS symbol "
                    f"semantic vocab (got idx={eos_idx}, V_hmm="
                    f"{self.emission_log.shape[1]})")
            self._log_eos = self.emission_log[:, int(eos_idx)].clone()  # [H]

        # Precompute F matrices and edge structure for vectorized DP
        import time
        print(f"[BetaBackwardDP] Precomputing F matrices...", flush=True)
        t0 = time.time()
        self.F_matrices: Dict[Tuple, torch.Tensor] = {}
        self._precompute_F_matrices()
        print(f"[BetaBackwardDP] F matrices: {time.time()-t0:.2f}s, "
              f"U={len(self.F_matrices)}", flush=True)

        # Build vectorized edge structure
        t0 = time.time()
        self._build_edge_structure()
        print(f"[BetaBackwardDP] Edge structure: {time.time()-t0:.2f}s, "
              f"E={self._total_edges}", flush=True)

        # Build backward DP table (vectorized)
        print(f"[BetaBackwardDP] Building DP table (vectorized)...", flush=True)
        t0 = time.time()
        self.C = self._build_dp_table()
        print(f"[BetaBackwardDP] DP table: {time.time()-t0:.2f}s", flush=True)

        # Hard-mask mode: boolean accept-reachability table (unweighted DP).
        # Built independently of C so no HMM weighting can leak through
        # (C's finiteness would coincide for uniform HMMs, but a trained
        # HMM with -inf emissions could disagree; the indicator must be
        # exactly D_β reachability).
        self.R = self._build_reach_table() if hard_mask else None

        print(f"[BetaBackwardDP] Built: S={self.S}, H={self.H}, V={self.V}, "
              f"budget={max_budget}, U={len(self.F_matrices)}"
              + (", HARD-MASK (unweighted reachability)" if hard_mask else ""),
              flush=True)

    def _step_key(self, step) -> Tuple[str, Tuple[int, ...]]:
        """
        Canonical hashable key for VHStep.

        Returns (action_name, args_tuple) where args is tuple of canonical IDs.
        """
        return (step.action, tuple(step.args) if hasattr(step.args, '__iter__') else (step.args,))

    def _step_to_semantic_tokens(self, step) -> List[int]:
        """
        Convert VHStep to semantic token indices.

        VHStep("GRAB", (3,)) -> ["GRAB", "ID3"] -> [idx("GRAB"), idx("ID3")]

        Unknown tokens map to UNK (or raise, if `vocab.strict_actions`), and
        a warning is emitted once per distinct miss: a predicate present in
        the γ automaton but missing from the semantic vocab would otherwise
        make the DP table score it as UNK.
        """
        tokens = []

        # Action token (uppercase)
        action_name = step.action.upper()
        action_name = self.vocab.aliases.get(action_name, action_name)
        action_idx = self.vocab.token_to_idx.get(action_name)
        if action_idx is None:
            if self.vocab.strict_actions:
                raise KeyError(f"[BetaBackwardDP] predicate {action_name!r} has no column in the "
                               f"semantic vocab {self.vocab.source_path}")
            action_idx = self.vocab.unk_idx
            self._warn_unknown_token("action", action_name)
        tokens.append(action_idx)

        # Canonical ID tokens: cid -> "IDk"
        args = step.args if hasattr(step.args, '__iter__') \
            else (step.args,)
        for cid in args:
            id_token = f"ID{cid}"
            id_idx = self.vocab.token_to_idx.get(id_token)
            if id_idx is None:
                id_idx = self.vocab.unk_idx
                self._warn_unknown_token("id", id_token)
            tokens.append(id_idx)

        return tokens

    def _warn_unknown_token(self, kind: str, token: str) -> None:
        """Emit a one-time warning per (kind, token) pair so we don't spam,
        but every fresh vocab miss surfaces."""
        if not hasattr(self, '_seen_unk_tokens'):
            self._seen_unk_tokens = set()
        key = (kind, token)
        if key in self._seen_unk_tokens:
            return
        self._seen_unk_tokens.add(key)
        print(
            f"[BetaBackwardDP] WARNING: semantic vocab has no entry for "
            f"{kind}={token!r}; falling back to UNK. β scoring for this "
            f"token will be meaningless. Likely cause: vocab file out of "
            f"sync with γ DFA / β state machine predicate set.",
            flush=True,
        )

    def _compute_F_matrix(self, tokens: List[int]) -> torch.Tensor:
        """
        Compute F_a[h, h'] = log P(emit tokens AND end at h' | start at h).

        Uses iterative DP through all tokens:
        G^{(0)} = identity
        G^{(t)}[h0, ht] = logsumexp_{h_{t-1}} (
            G^{(t-1)}[h0, h_{t-1}] + emit[h_{t-1}, y_t] + trans[h_{t-1}, ht]
        )

        IMPORTANT: Never skip tokens. Unknown tokens should already be mapped to UNK.
        """
        H = self.H

        if self.uniform_action_logw is not None:
            # Uniform meta-token measure: p(a, h' | h) = w / H for every action.
            return torch.full((H, H), self.uniform_action_logw - math.log(H), device=self.device)

        if len(tokens) == 0:
            # No tokens = identity (probability 1 of staying in same state)
            F = torch.full((H, H), NEG_INF, device=self.device)
            F.diagonal().fill_(0.0)
            return F

        # G[h_start, h_current] - start with identity
        G = torch.full((H, H), NEG_INF, device=self.device)
        G.diagonal().fill_(0.0)

        for tok in tokens:
            # Clamp to valid range (should already be valid if using UNK)
            tok = max(0, min(tok, self.V - 1))

            # emit[h] = log P(tok | h)
            emit = self.emission_log[:, tok]  # [H]

            # G_new[h0, h_new] = logsumexp_{h_prev} (
            #     G[h0, h_prev] + emit[h_prev] + trans[h_prev, h_new]
            # )
            #
            # G has shape [H, H]: G[h0, h_prev]
            # emit has shape [H]: emit[h_prev]
            # trans has shape [H, H]: trans[h_prev, h_new]

            G_plus_emit = G + emit.unsqueeze(0)  # [H, H] broadcast emit over h0

            # G_plus_emit[h0, h_prev] + trans[h_prev, h_new]
            # -> expand to [H, H, H] then logsumexp over h_prev (dim=1)
            G = torch.logsumexp(
                G_plus_emit.unsqueeze(2) + self.log_trans.unsqueeze(0),  # [H, H, H]
                dim=1  # sum over h_prev
            )  # [H, H]

        return G

    def _precompute_F_matrices(self):
        """
        Precompute F matrices for all unique actions in the state machine.

        Each unique (action, args) tuple gets one F matrix.
        """
        seen_keys = set()

        for src_idx, trans_list in self.sm.transitions.items():
            for step, dst_idx in trans_list:
                key = self._step_key(step)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Convert step to semantic tokens
                tokens = self._step_to_semantic_tokens(step)

                # Compute F matrix
                F = self._compute_F_matrix(tokens)
                self.F_matrices[key] = F

        print(f"[BetaBackwardDP] Precomputed {len(self.F_matrices)} F matrices")

    def _build_edge_structure(self):
        """
        Build vectorized edge structure for per-action batched DP.

        Creates:
        - self._step_keys: List of unique step keys (length U)
        - self._F_stacked: [U, H, H] tensor of stacked F matrices
        - self._edge_src: List[Tensor] - src[u] is [K_u] int64 tensor
        - self._edge_dst: List[Tensor] - dst[u] is [K_u] int64 tensor
        - self._total_edges: Total edge count
        """
        # Assign integer indices to unique step keys
        key_to_u: Dict[Tuple, int] = {}
        self._step_keys = []
        for key in self.F_matrices:
            key_to_u[key] = len(self._step_keys)
            self._step_keys.append(key)

        U = len(self._step_keys)

        # Stack F matrices: [U, H, H]
        if U > 0:
            self._F_stacked = torch.stack(
                [self.F_matrices[k] for k in self._step_keys], dim=0
            )  # [U, H, H]
        else:
            self._F_stacked = torch.empty((0, self.H, self.H), device=self.device)

        # Build per-action edge lists
        edge_src_lists: List[List[int]] = [[] for _ in range(U)]
        edge_dst_lists: List[List[int]] = [[] for _ in range(U)]

        for src_idx, trans_list in self.sm.transitions.items():
            for step, dst_idx in trans_list:
                key = self._step_key(step)
                u = key_to_u.get(key)
                if u is None:
                    continue
                edge_src_lists[u].append(src_idx)
                edge_dst_lists[u].append(dst_idx)

        # Convert to GPU tensors
        self._edge_src = []
        self._edge_dst = []
        self._total_edges = 0
        for u in range(U):
            src_t = torch.tensor(edge_src_lists[u], dtype=torch.int64, device=self.device)
            dst_t = torch.tensor(edge_dst_lists[u], dtype=torch.int64, device=self.device)
            self._edge_src.append(src_t)
            self._edge_dst.append(dst_t)
            self._total_edges += len(edge_src_lists[u])

    @staticmethod
    def _contrib_for_action(F_u: torch.Tensor, C_dst: torch.Tensor) -> torch.Tensor:
        """
        Compute contribution for one action across all its edges.

        Args:
            F_u: [H, H] action transition matrix
            C_dst: [K, H] future values at destination states

        Returns:
            [K, H] contribution: contrib[k, h] = logsumexp_{h'} (F_u[h, h'] + C_dst[k, h'])
        """
        # F_u[h, h'] + C_dst[k, h'] -> [K, H, H] via broadcasting, logsumexp over h'
        return torch.logsumexp(F_u.unsqueeze(0) + C_dst.unsqueeze(1), dim=2)

    def _build_dp_table(self) -> torch.Tensor:
        """
        Build backward DP table C[B, s, h] using per-action batching with
        2-pass scatter logsumexp.

        Base case: C[0, accept, :] = 0, C[0, non_accept, :] = -inf

        Recurrence:
        C[B, s, h] = logaddexp(
            C[B-1, s, h],
            logsumexp_{a: s→s'} logsumexp_{h'} (F_a[h, h'] + C[B-1, s', h'])
        )

        Vectorized via per-action batching:
        - For each unique action u, gather all (src, dst) edges
        - Compute contrib = logsumexp_{h'} (F_u[h,h'] + C[B-1, dst, h']) in batch
        - Aggregate per source state using stable 2-pass scatter logsumexp

        Returns:
            C: [max_budget+1, S, H] tensor
        """
        S, H = self.S, self.H
        horizon = self.max_budget
        U = len(self._step_keys)

        # Initialize C table
        C = torch.full((horizon + 1, S, H), NEG_INF, device=self.device)

        accept_mask = torch.tensor(self.sm.accept_mask, dtype=torch.bool, device=self.device)
        accept_indices = accept_mask.nonzero(as_tuple=True)[0]

        # An accepting continuation stops by emitting EOS once; the recurrence
        # below adds the branch that keeps acting.
        stop_term = torch.full((S, H), NEG_INF, device=self.device)
        if len(accept_indices) > 0:
            stop_term[accept_indices, :] = self._log_eos.unsqueeze(0)
        C[0] = stop_term

        # C[B, s] = logaddexp(stop[s], logsumexp over (a, s'=δ(s,a)) of F_a + C[B-1, s'])
        #
        # No carry term C[B-1, s, h]: every budget unit is an action emission,
        # so there is no "wait without acting" option. A carry would let
        # C[B, non_accept] saturate to ≈0 once B exceeds the depth to accept,
        # removing the preference for shorter accepting paths. C[budget, s] is
        # finite iff accept is reachable from s in ≤ budget actions.
        for B in range(1, horizon + 1):
            C_prev = C[B - 1]  # [S, H]

            # Accumulators for stable scatter logsumexp
            max_contrib = torch.full((S, H), NEG_INF, device=self.device)
            sum_exp = torch.zeros((S, H), device=self.device)

            # --- Pass 1: compute max_contrib[s, h] ---
            for u in range(U):
                src = self._edge_src[u]   # [K_u]
                dst = self._edge_dst[u]   # [K_u]
                if src.shape[0] == 0:
                    continue

                C_dst = C_prev.index_select(0, dst)  # [K_u, H]
                contrib = self._contrib_for_action(self._F_stacked[u], C_dst)  # [K_u, H]

                # Scatter max into max_contrib
                idx = src.unsqueeze(1).expand_as(contrib)  # [K_u, H]
                max_contrib.scatter_reduce_(0, idx, contrib, reduce="amax", include_self=True)

            # --- Pass 2: compute sum_exp[s, h] ---
            for u in range(U):
                src = self._edge_src[u]
                dst = self._edge_dst[u]
                if src.shape[0] == 0:
                    continue

                C_dst = C_prev.index_select(0, dst)  # [K_u, H]
                contrib = self._contrib_for_action(self._F_stacked[u], C_dst)  # [K_u, H]

                # Shift by max for numerical stability, then scatter add exp
                idx = src.unsqueeze(1).expand_as(contrib)
                max_at_src = max_contrib.index_select(0, src)  # [K_u, H]
                shifted = contrib - max_at_src  # [K_u, H]
                sum_exp.scatter_add_(0, idx, torch.exp(shifted))

            # Combine: action_sum = max_contrib + log(sum_exp)
            # Where sum_exp == 0, log gives -inf → -inf. Correct (no path).
            action_sum = max_contrib + torch.log(sum_exp.clamp_min(1e-30))

            C[B] = torch.logaddexp(action_sum, stop_term)  # stop now (accept only) or act

        return C

    def _build_reach_table(self) -> torch.Tensor:
        """Boolean accept-reachability table R[B, s] (hard-mask mode).

        R[B, s] = True iff an accept state is reachable from s in ≤ B
        action emissions. Mirrors _build_dp_table's no-carry semantics:
        accept rows are absorbing/pinned True at every budget, so a path
        of length k < B stays reachable at budget B.
        """
        S = self.S
        R = torch.zeros((self.max_budget + 1, S), dtype=torch.bool,
                        device=self.device)
        accept_mask = torch.tensor(self.sm.accept_mask, dtype=torch.bool,
                                   device=self.device)
        R[0] = accept_mask
        # Flatten edges across all actions once.
        if self._total_edges > 0:
            all_src = torch.cat([self._edge_src[u]
                                 for u in range(len(self._step_keys))
                                 if self._edge_src[u].shape[0] > 0])
            all_dst = torch.cat([self._edge_dst[u]
                                 for u in range(len(self._step_keys))
                                 if self._edge_dst[u].shape[0] > 0])
            for B in range(1, self.max_budget + 1):
                reach_prev = R[B - 1]
                # src reachable this step iff any edge lands on a reachable dst
                # (int8: scatter_reduce_ amax has no bool support)
                step_reach = torch.zeros(S, dtype=torch.int8, device=self.device)
                hits = reach_prev.index_select(0, all_dst).to(torch.int8)
                step_reach.scatter_reduce_(0, all_src, hits, reduce="amax",
                                           include_self=True)
                R[B] = step_reach.bool() | accept_mask
        else:
            R[1:] = accept_mask.unsqueeze(0)
        return R

    def compute_beta_factors_batch(
        self,
        belief: torch.Tensor,   # [H] log-space belief
        steps: List,            # list of steps, all from the SAME state/budget
        state_id: int,
        budget: int,
    ) -> List[float]:
        """β_factor (module docstring, item 3) for candidates sharing one
        (state_id, budget). Returns a list aligned with `steps`.

        apply contract (uniform across all β-SMs):
          - int != state_id : state-changing transition
          - state_id        : identity (no abstract effect / goal-dropped)
          - None            : hard reject (precondition fail / blocked) → NEG_INF
        """
        n = len(steps)
        out: List[float] = [NEG_INF] * n
        budget_after = budget - 1
        if budget_after < 0:
            return out
        bc = min(budget_after, self.max_budget)


        if self.hard_mask:
            # Indicator per candidate; no HMM weighting.
            reach = self.R[bc]
            for i, step in enumerate(steps):
                s_next = self.sm.apply(state_id, step)
                if s_next is not None and bool(reach[s_next]):
                    out[i] = 0.0
            return out

        belief = belief.to(self.device)
        valid_idx: List[int] = []
        F_list: List[torch.Tensor] = []
        snext_list: List[int] = []
        for i, step in enumerate(steps):
            s_next = self.sm.apply(state_id, step)
            if s_next is None:
                continue  # hard reject → out[i] stays NEG_INF
            key = self._step_key(step)
            F_a = self.F_matrices.get(key)
            if F_a is None:
                F_a = self._compute_F_matrix(self._step_to_semantic_tokens(step))
            valid_idx.append(i)
            F_list.append(F_a)
            snext_list.append(s_next)
        if not valid_idx:
            return out

        F_stack = torch.stack(F_list, dim=0)                       # [U,H,H]
        snext_t = torch.tensor(snext_list, dtype=torch.long,
                               device=self.device)
        C_next = self.C[bc, snext_t, :]                            # [U,H]
        # inner[u,h] = logsumexp_h'(F[u,h,h'] + C_next[u,h'])
        inner = torch.logsumexp(F_stack + C_next.unsqueeze(1), dim=2)  # [U,H]
        # bf[u] = logsumexp_h(belief[h] + inner[u,h])
        bf = torch.logsumexp(belief.unsqueeze(0) + inner, dim=1)   # [U]
        bf_list = bf.tolist()
        for j, i in enumerate(valid_idx):
            out[i] = bf_list[j]
        return out

    def _F_for_step(self, step) -> torch.Tensor:
        key = self._step_key(step)
        F_a = self.F_matrices.get(key)
        if F_a is None:
            F_a = self._compute_F_matrix(self._step_to_semantic_tokens(step))
            self.F_matrices[key] = F_a
        return F_a

    def compute_meta_prior_batch(
        self,
        belief: torch.Tensor,   # [H] log-space meta belief
        steps: List,
        state_id: Optional[int] = None,
    ) -> List[float]:
        """log p_meta(a | belief) = logsumexp_h(b[h] + logsumexp_h' F_a[h,h']) for
        every step. A TRAINED meta-HMM puts (learned, usually small) prior mass on
        precondition-violating actions too; that mass stays in the
        normaliser. The UNIFORM measure gives weight 1/(|W|+1) to every
        *admissible* meta-token and zero to inadmissible ones (the DP only sums
        over the automaton's edges), so in uniform mode dead ends get prior −inf
        when `state_id` is supplied."""
        if not steps:
            return []
        belief = belief.to(self.device)
        F_stack = torch.stack([self._F_for_step(st) for st in steps], dim=0)  # [U,H,H]
        row = torch.logsumexp(F_stack, dim=2)                                 # [U,H]
        pri = torch.logsumexp(belief.unsqueeze(0) + row, dim=1)               # [U]
        out = pri.tolist()
        if self.uniform_action_logw is not None and state_id is not None:
            for i, st in enumerate(steps):
                if self.sm.apply(state_id, st) is None:
                    out[i] = NEG_INF
        return out

    def posterior_after_step(self, belief: torch.Tensor, step) -> torch.Tensor:
        """Meta-HMM forward step: log p(h' | a_<l, a) ∝ logsumexp_h(b[h] + F_a[h,h']),
        normalised. This is the belief the β-DP must be queried with after `a`."""
        belief = belief.to(self.device)
        F_a = self._F_for_step(step)
        post = torch.logsumexp(belief.unsqueeze(1) + F_a, dim=0)  # [H']
        return post - torch.logsumexp(post, dim=0)

    def log_stop_prior(self, belief: torch.Tensor) -> float:
        """log p_meta(STOP | belief) = logsumexp_h(b[h] + log p(EOS | h)).
"""
        return torch.logsumexp(belief.to(self.device) + self._log_eos, dim=0).item()

    def compute_log_M(
        self,
        log_rho: torch.Tensor,
        state_idx: int,
        budget: int,
    ) -> torch.Tensor:
        """
        Compute log M_β(s, B) = logsumexp_h(log_rho[h] + C[B, s, h]).

        This is the log probability of reaching accept from state s with budget B,
        marginalized over HMM states with belief log_rho.

        Args:
            log_rho: [H] log forward belief
            state_idx: State index in state machine
            budget: Remaining step budget

        Returns:
            Scalar log mass
        """
        budget_clamped = min(budget, self.max_budget)
        if self.hard_mask:
            # Reachability indicator; callers only test > -1e20.
            return torch.tensor(
                0.0 if bool(self.R[budget_clamped, state_idx]) else NEG_INF,
                device=self.device)
        log_rho = log_rho.to(self.device)
        return torch.logsumexp(log_rho + self.C[budget_clamped, state_idx, :], dim=0)

    @property
    def initial_state_id(self) -> int:
        """Get the initial state ID."""
        return self.sm.get_initial_state_id()

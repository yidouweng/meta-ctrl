"""Uniform meta-token surrogate: detection and per-action weight.

The uniform surrogate is uniform over *candidate actions*: every admissible
meta-token --- each grounded action in W, and STOP --- carries the same weight
w = 1/(|W|+1), independent of its arity.  The backward DP under this measure
therefore counts the automaton's accepting completions, one factor of w per
action plus one for stopping.  It is parameter-free and needs no corpus.

|W| is the per-prompt set of gamma-valid grounded actions, so the measure is
uniform over the alphabet the semantic automaton actually distinguishes for
that scene.  `resolve_uniform_logw` supplies that weight to
`BetaBackwardDP(uniform_action_logw=...)` when the loaded meta-token HMM is
uniform.
"""
from __future__ import annotations

import math
from typing import Optional



def detect_uniform_meta(semantic_hmm, tol: float = 1e-4) -> bool:
    """True if the meta-HMM has constant emission log-probs and constant transitions."""
    try:
        beta = semantic_hmm.beta
        alpha = semantic_hmm.alpha_exp
    except AttributeError:
        return False
    b = beta[beta > -1e8]   # drop -inf and the -1e9 / -1e30 padding columns
    if b.numel() == 0:
        return False
    emis_uniform = bool((b.max() - b.min()).item() < tol)
    trans_uniform = bool((alpha.max() - alpha.min()).item() < tol)
    return emis_uniform and trans_uniform


def uniform_action_logw(n_grounded_actions: int) -> float:
    """log weight of one meta-token when |W| grounded actions and STOP share the mass."""
    return -math.log(int(n_grounded_actions) + 1)


def resolve_uniform_logw(semantic_hmm, n_grounded_actions: int) -> Optional[float]:
    """The per-action log weight if `semantic_hmm` is uniform, else None."""
    if semantic_hmm is None or not detect_uniform_meta(semantic_hmm):
        return None
    return uniform_action_logw(n_grounded_actions)

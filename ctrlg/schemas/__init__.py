"""Canonical per-(module, dataset) action / predicate schemas.

One source of truth for the action set, arity, affordances, and
goal-state→action mapping per module. All other sites (γ DFA builders,
β state machine, validation, vocab JSON generation) should import from
here, never redefine.
"""

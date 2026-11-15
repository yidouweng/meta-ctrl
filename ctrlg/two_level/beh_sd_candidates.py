"""Candidate tracker for the BEHAVIOR Subgoal Decomposition output format.

Counterpart of `beh_candidates.py` for the gamma automaton in
`ctrlg/gamma/topdown/behavior_sd_twolevel.py`.

A candidate is a (predicate, negation, canonical ids) atom. The tracker builds
the token sequence the LM would emit for it, so the processor can score it.

Content layout per candidate (canonical, no BPE merges):
    [`not `]?  pred_tokens  `(`  inst_tokens  [`, ` inst_tokens]*  `)`
where each segment is one of the precomputed token sequences exposed on the
gamma automaton:
    - `not_seq`              : tokens for `not `
    - `pred_token_seqs[pred]`: tokens for the predicate name
    - `lparen_seq`           : tokens for `(`
    - `instance_token_seqs[inst]` : tokens for an instance name
    - `obj_sep_seq`          : tokens for `, ` between args
    - `rparen_seq`           : tokens for `)`
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .tau import MetaStep
from .candidates import _BaseCandidateTracker, Candidate


class BehaviorSDTwoLevelCandidateTracker(_BaseCandidateTracker):
    """Candidate tracker for the BEHAVIOR Subgoal Decomposition format.

    Args:
        gamma_valid_steps: (predicate, neg, canonical-id tuple) atoms gamma can
            produce; the candidates of every block are drawn from them.
        canonical_id_to_instance: canonical id -> instance name
            (e.g. ['agent_n_01.1', 'apple.0', ...]); gamma's
            `instance_token_seqs` is keyed by instance name.
    """

    def __init__(self, *, tokenizer, gamma_dfa, tau_parser,
                 canonical_id_to_instance: List[str],
                 gamma_valid_steps: List[Tuple[str, bool, Tuple[int, ...]]]):
        super().__init__(tokenizer=tokenizer, gamma_dfa=gamma_dfa,
                         tau_parser=tau_parser)
        self.canonical_id_to_instance = [str(x) for x in canonical_id_to_instance]
        self.gamma_valid_steps = list(gamma_valid_steps)

    # ------------------------------------------------------------------

    def _enumerate_gamma_valid_steps(self) -> List[MetaStep]:
        """(pred, neg, args) atoms as MetaSteps; negated atoms get a NOT_ prefix,
        as tau emits them."""
        steps: List[MetaStep] = []
        for pred, neg, args in self.gamma_valid_steps:
            action = (("NOT_" + pred.upper()) if neg else pred.upper())
            steps.append(MetaStep(action=action, ids=tuple(args)))
        return steps

    def _is_content_token(self, prev_base: tuple, new_base: tuple) -> bool:
        """True if the token just consumed belongs to the candidates' witness,
        so content_pos advances in lockstep with it.

        Witness layout (see `_build_candidates`):
            [not_seq]?  pred_seq  '('  inst_seq  [obj_sep_seq  inst_seq]*  ')'

        The `not ` prefix is part of a negated atom's witness, so NEG_SEQ must
        count as content; otherwise content_pos would lag by the prefix and no
        predicate-name token would match its group.

        Content phases (transitions into a state that consumes a witness token):
            NEG_SEQ          — consuming `not_seq` bytes (negation prefix)
            EXPECT_PRED      — consuming predicate-name bytes
            LPAREN_SEQ       — multi-token `(` (rare; usually 1-token)
            EXPECT_INSTANCE  — consuming instance-name bytes
            OBJ_SEP_SEQ      — multi-token `, ` between binary args
            RPAREN_SEQ       — multi-token `)` (rare; usually 1-token)

        Block-boundary phases (AFTER_PRED, AND_SEQ, PRED_START, AFTER_STEP,
        SUBGOAL_SEP_SEQ, CLOSE_LIST_SEQ, END, NEED_QUOTE, NEED_RBRACE,
        NEED_SPACE_PRED1) are not content: they are structure between blocks or
        the end of the output.
        """
        new_p = new_base[0] if isinstance(new_base, tuple) and new_base \
            else str(new_base)
        return str(new_p) in {
            "NEG_SEQ", "EXPECT_PRED", "LPAREN_SEQ", "EXPECT_INSTANCE",
            "OBJ_SEP_SEQ", "RPAREN_SEQ",
        }

    # ------------------------------------------------------------------

    def _build_candidates(self, steps: Sequence[MetaStep]) -> List[Candidate]:
        out: List[Candidate] = []
        dfa = self.gamma_dfa

        # Canonical token sequences (no BPE merges) precomputed by gamma.
        not_seq:        List[int] = list(getattr(dfa, 'not_seq',        ()) or ())
        lparen_seq:     List[int] = list(getattr(dfa, 'lparen_seq',     ()) or ())
        rparen_seq:     List[int] = list(getattr(dfa, 'rparen_seq',     ()) or ())
        obj_sep_seq:    List[int] = list(getattr(dfa, 'obj_sep_seq',    ()) or ())
        pred_token_seqs = getattr(dfa, 'pred_token_seqs', {})
        instance_token_seqs = getattr(dfa, 'instance_token_seqs', {})

        for st in steps:
            action = str(st.action)
            if action.startswith('NOT_'):
                pred = action[4:].lower()
                neg = True
            else:
                pred = action.lower()
                neg = False

            pred_seq = pred_token_seqs.get(pred)
            if pred_seq is None:
                continue

            content_tokens: List[int] = []
            if neg:
                content_tokens.extend(not_seq)
            content_tokens.extend(list(pred_seq))
            content_tokens.extend(lparen_seq)

            # Args — instances separated by `, `.
            arity_ok = True
            for i, cid in enumerate(st.ids):
                if not (0 <= int(cid) < len(self.canonical_id_to_instance)):
                    arity_ok = False
                    break
                inst = self.canonical_id_to_instance[int(cid)]
                inst_seq = instance_token_seqs.get(inst)
                if inst_seq is None:
                    arity_ok = False
                    break
                if i > 0:
                    content_tokens.extend(obj_sep_seq)
                content_tokens.extend(list(inst_seq))

            if not arity_ok:
                continue

            content_tokens.extend(rparen_seq)

            out.append(Candidate(
                step=st,
                content_tokens=tuple(int(x) for x in content_tokens),
            ))

        return out

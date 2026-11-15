"""Candidate tracker for the BEHAVIOR Action Sequencing output format.

Works with BehaviorTwoLevelStateMachine. A candidate's content is

    action tokens, `", "object": "` (mid_seq), instance tokens [`,` instance tokens]

The fixed separators are part of the content (as VH's `", "` and SD's `.`):
at an action name that is a prefix of another (LEFT_PLACE_NEXTTO /
LEFT_PLACE_NEXTTO_ONTOP) the separator is then a group token, so the
processor's in-block mask stays on and only continuations of live
candidates are allowed.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple

from .tau import MetaStep
from .candidates import _BaseCandidateTracker, Candidate


class BehaviorTwoLevelCandidateTracker(_BaseCandidateTracker):
    """Witness = action tokens + instance-name tokens per argument."""

    def __init__(self, *, tokenizer, gamma_dfa, tau_parser,
                 canonical_id_to_instance: List[str],
                 gamma_valid_actions: List[Tuple[str, Tuple[int, ...]]]):
        super().__init__(tokenizer=tokenizer, gamma_dfa=gamma_dfa, tau_parser=tau_parser)
        self.canonical_id_to_instance = [str(x) for x in canonical_id_to_instance]
        self.gamma_valid_actions = gamma_valid_actions

    def _enumerate_gamma_valid_steps(self) -> List[MetaStep]:
        return [MetaStep(action=act, ids=cids) for act, cids in self.gamma_valid_actions]

    def _is_content_token(self, prev_base: tuple, new_base: tuple) -> bool:
        new_p = new_base[0] if isinstance(new_base, tuple) and new_base else str(new_base)
        return str(new_p) in {"EXPECT_ACTION", "MID_SEQ", "EXPECT_INSTANCE", "OBJ_SEP"}

    def _build_candidates(self, steps: Sequence[MetaStep]) -> List[Candidate]:
        out: List[Candidate] = []
        dfa = self.gamma_dfa

        for st in steps:
            action = str(st.action).upper()
            action_seq = dfa.action_token_seqs.get(action)
            if action_seq is None:
                continue

            content_tokens: List[int] = list(action_seq) + list(dfa.mid_seq)

            for i, cid in enumerate(st.ids):
                if not (0 <= int(cid) < len(self.canonical_id_to_instance)):
                    continue
                inst = self.canonical_id_to_instance[int(cid)]
                inst_seq = dfa.instance_token_seqs.get(inst)
                if inst_seq is None:
                    continue
                if i > 0:
                    content_tokens.extend(list(dfa.obj_sep_seq))
                content_tokens.extend(list(inst_seq))

            out.append(Candidate(step=st, content_tokens=tuple(int(x) for x in content_tokens)))
        return out

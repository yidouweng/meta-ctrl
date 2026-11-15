"""BEHAVIOR action-sequencing γ: the syntax automaton over tokens.

Uses a single instance trie (instance names are not split into name and id).

Output format:
    [{"action": "ACTION", "object": "instance_name"}, ...]

Content phases:
    EXPECT_ACTION  — action name trie
    EXPECT_INSTANCE — instance name trie (full: plywood_0, room_floor_kitchen_0)

All other phases are structural (JSON punctuation) — tracked but not
content for the candidate tracker.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .base import CanonicalStateMachine
from .action_defs import BEHAVIOR_ACTIONS, BEHAVIOR_ACTION_ARITY, compute_behavior_affordances
from .trie_builder import TokenTrie, canonical_tokenization
from .prompt_parser import parse_beh_objects


class BehaviorTwoLevelStateMachine(CanonicalStateMachine):
    """
    BEHAVIOR action-sequencing γ automaton.

    Grammar: [{"action":"ACTION","object":"instance"}, ...]
    Single instance trie per object position (no name/ID split).
    """

    def __init__(
        self,
        tokenizer,
        objects: Dict[str, List[str]],
        predicates: Dict[str, Set[str]],
        eos_token_id: int,
        objects_meta: Optional[List[Dict]] = None,
        container_role_bases: Optional[Set[str]] = None,
    ):
        import time as _t
        _t0 = _t.time()
        def _p(msg): print(f"  [BehTwoLevel {_t.time()-_t0:.2f}s] {msg}", flush=True)

        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.objects = objects  # {base_name: [instance_name, ...]}

        # Flatten all instance names
        self.all_instances: List[str] = []
        for base in sorted(objects.keys()):
            for inst in sorted(objects[base]):
                self.all_instances.append(inst)
        _p(f"{len(self.all_instances)} instances")

        # Shared taxonomy membership layer (γ and β read the same sets).
        from ctrlg.beta.beh_affordances import BehAffordances
        self.membership = BehAffordances(objects=objects_meta)

        # Compute affordances
        self.allowed_objects_per_role = compute_behavior_affordances(
            predicates, all_objects=objects, membership=self.membership,
            container_role_bases=container_role_bases,
        )
        self.action_arity = dict(BEHAVIOR_ACTION_ARITY)
        self.actions = sorted(BEHAVIOR_ACTIONS)

        # Compute viable actions (where all roles have ≥1 valid instance)
        self.viable_actions = self._compute_viable_actions()
        _p(f"{len(self.viable_actions)}/{len(self.actions)} viable actions")

        # Build tries
        _p("build action trie...")
        self.action_trie = self._build_action_trie(tokenizer)

        _p("build instance trie...")
        self.instance_trie = self._build_instance_trie(tokenizer)

        # Precompute token sequences for candidates
        _p("precompute token seqs...")
        self.action_token_seqs: Dict[str, Tuple[int, ...]] = {}
        for action in self.viable_actions:
            seq = canonical_tokenization(tokenizer, action)
            if seq:
                self.action_token_seqs[action] = seq

        self.instance_token_seqs: Dict[str, Tuple[int, ...]] = {}
        for inst in self.all_instances:
            seq = canonical_tokenization(tokenizer, inst)
            if seq:
                self.instance_token_seqs[inst] = seq

        # Precompute structural token alternative sets
        _p("precompute structural token alts...")
        self._precompute_structural_alts(tokenizer)

        # Viable instance nodes per (action, role) for affordance filtering
        self._viable_inst_cache: Dict[Tuple[str, int], Optional[Set[int]]] = {}

        self._states = None
        _p("done")

    def _compute_viable_actions(self) -> Set[str]:
        viable = set()
        for action in self.actions:
            arity = self.action_arity.get(action, 1)
            if arity == 0:
                viable.add(action)
                continue
            all_ok = True
            for role in range(arity):
                allowed = self.allowed_objects_per_role.get((action, role))
                if allowed is not None and not allowed:
                    all_ok = False
                    break
            if all_ok:
                viable.add(action)
        return viable

    def _build_action_trie(self, tokenizer) -> TokenTrie:
        trie = TokenTrie()
        for action in sorted(self.viable_actions):
            seq = canonical_tokenization(tokenizer, action)
            if seq:
                trie.insert(seq, label=action)
        return trie

    def _build_instance_trie(self, tokenizer) -> TokenTrie:
        trie = TokenTrie()
        for inst in self.all_instances:
            seq = canonical_tokenization(tokenizer, inst)
            if seq:
                trie.insert(seq, label=inst)
        return trie

    def _precompute_structural_alts(self, tokenizer):
        """Pre-compute token alternative sets for structural positions.

        These are positions where the LLM generates JSON punctuation.
        We accept any natural BPE token that covers the expected structural chars.
        """
        # Canonical structural sequences
        # Prefix alternatives: LLM prefers '[\n' over '[' at position 0
        self._prefix_start_alts: Set[int] = set()
        for tok_id in range(tokenizer.vocab_size):
            d = tokenizer.decode([tok_id])
            if d and d == '[' or (d and d.startswith('[') and len(d) == 2 and d[1] in '\n\r\t '):
                self._prefix_start_alts.add(tok_id)  # [ or [\n

        self.prefix_seq = canonical_tokenization(tokenizer, '[{"action": "')
        self.mid_seq = canonical_tokenization(tokenizer, '", "object": "')
        self.sep_seq = canonical_tokenization(tokenizer, ', {"action": "')
        # The processor closes the plan with ']' once the action budget is spent.
        self.close_list_seq = canonical_tokenization(tokenizer, ']')
        self.eos_tokens = {self.eos_token_id}

        # For arity-2 actions: comma between objects in the value
        self.obj_sep_seq = canonical_tokenization(tokenizer, ',')

        # Token alternatives for dict close + list transitions
        # At EXPECT_INSTANCE leaf (last role): expect '"}'
        # But LLM may generate '"}, ' or '"},' or '"}, {"action": "' etc.
        _STRUCT = set('"}], {\n\r\t')
        _WS_ONLY = set(' \n\r\t')
        self._close_dict_alts: Dict[int, str] = {}
        for tok_id in range(tokenizer.vocab_size):
            d = tokenizer.decode([tok_id])
            if not d or not d.startswith('"}'):
                continue
            rest = d[2:]
            if rest and not all(c in _STRUCT for c in rest):
                continue
            if rest == '' or all(c in _WS_ONLY for c in rest):
                # `"}` or `"}\n` or `"} ` — enter AFTER_DICT phase
                self._close_dict_alts[tok_id] = 'close_dict'
            elif rest.lstrip().startswith(','):
                self._close_dict_alts[tok_id] = 'close_dict_sep'     # "},  or "}, {
            elif rest.lstrip().startswith(']'):
                self._close_dict_alts[tok_id] = 'close_dict_end'     # "}]

        # A close_dict_sep token may cover more than the comma: e.g. the BPE
        # token '"},{"' also opens the next dict and its first quote. Resume
        # SEP_SEQ after the last separator piece the token covers (whitespace
        # ignored -- JSON-insignificant), so the next piece is not duplicated.
        # A token that ends inside a piece has no consistent continuation and
        # is dropped. Tokens covering only the comma resume at 1.
        _sep_pieces = [tokenizer.decode([t]) for t in self.sep_seq]
        _nows = lambda x: ''.join(c for c in x if c not in ' \n\r\t')
        self._close_dict_sep_pos: Dict[int, int] = {}
        for tok_id, dest in list(self._close_dict_alts.items()):
            if dest != 'close_dict_sep':
                continue
            rem = _nows(tokenizer.decode([tok_id])[2:])     # after '"}'
            pos = 0
            while pos < len(_sep_pieces) and rem and rem.startswith(_nows(_sep_pieces[pos])) and _nows(_sep_pieces[pos]):
                rem = rem[len(_nows(_sep_pieces[pos])):]
                pos += 1
            if rem or pos == 0:
                del self._close_dict_alts[tok_id]          # ends mid-piece
            else:
                self._close_dict_sep_pos[tok_id] = pos

        # At AFTER_DICT: expect ', {' (next action) or ']' (end)
        # Also allow pure-whitespace tokens (e.g. '\n', '  ', '\n  ') so the
        # model can pretty-print: Llama tends to emit '\n' before the closing
        # ']', and blocking it would push the model toward ',' (continue)
        # even when β would accept.
        _WS = set(' \n\r\t')
        self._after_dict_alts: Dict[int, str] = {}
        self._after_dict_ws_tokens: Set[int] = set()
        for tok_id in range(tokenizer.vocab_size):
            d = tokenizer.decode([tok_id])
            if not d:
                continue
            if d.startswith(','):
                rest = d[1:]
                if rest and not all(c in _STRUCT for c in rest):
                    continue
                self._after_dict_alts[tok_id] = 'sep'
            elif d.startswith(']'):
                rest = d[1:]
                if rest and not all(c in _STRUCT for c in rest):
                    continue
                self._after_dict_alts[tok_id] = 'close'
            elif all(c in _WS for c in d):
                # Pure whitespace token — consumed without state change
                self._after_dict_ws_tokens.add(tok_id)
            else:
                # Leading whitespace + ',' or ']'?
                stripped = d.lstrip()
                if stripped.startswith(','):
                    rest = stripped[1:]
                    if not rest or all(c in _STRUCT for c in rest):
                        self._after_dict_alts[tok_id] = 'sep'
                elif stripped.startswith(']'):
                    rest = stripped[1:]
                    if not rest or all(c in _STRUCT for c in rest):
                        self._after_dict_alts[tok_id] = 'close'

        # Same resume rule for AFTER_DICT separator tokens: ',{"' or ', {"'
        # cover more of ', {"action": "' than the comma.
        self._after_dict_sep_pos: Dict[int, int] = {}
        for tok_id, dest in list(self._after_dict_alts.items()):
            if dest != 'sep':
                continue
            rem = _nows(tokenizer.decode([tok_id]))
            pos = 0
            while pos < len(_sep_pieces) and rem and _nows(_sep_pieces[pos]) and rem.startswith(_nows(_sep_pieces[pos])):
                rem = rem[len(_nows(_sep_pieces[pos])):]
                pos += 1
            if rem or pos == 0:
                del self._after_dict_alts[tok_id]
            else:
                self._after_dict_sep_pos[tok_id] = pos

    def _viable_instance_nodes(self, action: str, role: int) -> Optional[Set[int]]:
        """Get instance trie nodes leading to affordance-valid instances."""
        key = (action, role)
        if key in self._viable_inst_cache:
            return self._viable_inst_cache[key]

        allowed_bases = self.allowed_objects_per_role.get(key)
        if allowed_bases is None:
            self._viable_inst_cache[key] = None
            return None

        # Find instances whose base name is in allowed_bases
        allowed_instances = set()
        for base in allowed_bases:
            for inst in self.objects.get(base, []):
                allowed_instances.add(inst)

        # Trace trie paths to allowed instance leaves
        viable = set()
        for inst in allowed_instances:
            seq = self.instance_token_seqs.get(inst)
            if not seq:
                continue
            node = 0
            viable.add(node)
            for tok in seq:
                next_node = self.instance_trie.next_node(node, tok)
                if next_node is not None:
                    viable.add(next_node)
                    node = next_node
                else:
                    break

        self._viable_inst_cache[key] = viable if viable else None
        return viable if viable else None

    # =========================================================================
    # CanonicalStateMachine ABC
    # =========================================================================

    def enumerate_states(self) -> List[Any]:
        """Not used — BFS compilation discovers states."""
        if self._states is not None:
            return self._states
        # Minimal: just return initial state. BFS will discover the rest.
        self._states = [self.initial_state()]
        return self._states

    def initial_state(self) -> Any:
        return ('PREFIX', 0)

    def is_accept(self, state: Any) -> bool:
        return isinstance(state, tuple) and state[0] == 'END'

    def allowed_tokens(self, state: Any) -> Set[int]:
        phase = state[0]

        if phase == 'PREFIX':
            idx = state[1]
            if idx == 0:
                return set(self._prefix_start_alts)
            if idx < len(self.prefix_seq):
                return {self.prefix_seq[idx]}
            # At action start: enter action name trie
            return self.action_trie.next_tokens(0)

        if phase == 'EXPECT_ACTION':
            node = state[1]
            tokens = set(self.action_trie.next_tokens(node))
            # At leaf: allow first token of mid_seq (transition to structural)
            if self.action_trie.is_leaf_node(node):
                tokens.add(self.mid_seq[0])
            return tokens

        if phase == 'MID_SEQ':
            idx = state[2]
            if idx < len(self.mid_seq):
                return {self.mid_seq[idx]}
            return set()

        if phase == 'EXPECT_INSTANCE':
            action, role, node = state[1], state[2], state[3]
            viable = self._viable_instance_nodes(action, role)
            raw_tokens = self.instance_trie.next_tokens(node)
            if viable is not None:
                tokens = set()
                for tok in raw_tokens:
                    next_n = self.instance_trie.next_node(node, tok)
                    if next_n is not None and next_n in viable:
                        tokens.add(tok)
            else:
                tokens = set(raw_tokens)
            # At leaf: allow close or object separator
            if self.instance_trie.is_leaf_node(node):
                arity = self.action_arity.get(action, 1)
                if role < arity - 1:
                    tokens.add(self.obj_sep_seq[0])
                else:
                    # Last object: allow close dict alternatives
                    tokens |= set(self._close_dict_alts.keys())
            return tokens

        if phase == 'OBJ_SEP':
            # After comma between objects in arity-2 action
            idx = state[2]
            if idx < len(self.obj_sep_seq):
                return {self.obj_sep_seq[idx]}
            return set()

        if phase == 'AFTER_DICT':
            if self._after_dict_ws_run(state) >= self.MAX_AFTER_DICT_WS:
                return set(self._after_dict_alts.keys())
            return set(self._after_dict_alts.keys()) | self._after_dict_ws_tokens

        if phase == 'SEP_SEQ':
            idx = state[1]
            if idx < len(self.sep_seq):
                return {self.sep_seq[idx]}
            return set()

        if phase == 'END':
            return set(self.eos_tokens)

        return set()

    def next_state(self, state: Any, token: int) -> Optional[Any]:
        token = int(token)
        phase = state[0]

        if phase == 'PREFIX':
            idx = state[1]
            if idx == 0:
                if token in self._prefix_start_alts:
                    return ('PREFIX', 1)  # consumed '[' or '[\n', need '{"action": "'
                return None
            if idx < len(self.prefix_seq):
                if token == self.prefix_seq[idx]:
                    return ('PREFIX', idx + 1)
                return None
            # Enter action trie
            next_node = self.action_trie.next_node(0, token)
            if next_node is not None:
                return ('EXPECT_ACTION', next_node)
            return None

        if phase == 'EXPECT_ACTION':
            node = state[1]
            # Check transition to mid_seq (action complete)
            if self.action_trie.is_leaf_node(node) and token == self.mid_seq[0]:
                action = self.action_trie.get_leaf_label(node)
                if action:
                    if len(self.mid_seq) == 1:
                        return ('EXPECT_INSTANCE', action, 0, 0)  # instance trie root
                    return ('MID_SEQ', action, 1)
            # Continue action trie
            next_node = self.action_trie.next_node(node, token)
            if next_node is not None:
                return ('EXPECT_ACTION', next_node)
            return None

        if phase == 'MID_SEQ':
            action, idx = state[1], state[2]
            if idx < len(self.mid_seq) and token == self.mid_seq[idx]:
                if idx + 1 >= len(self.mid_seq):
                    return ('EXPECT_INSTANCE', action, 0, 0)  # instance trie root
                return ('MID_SEQ', action, idx + 1)
            return None

        if phase == 'EXPECT_INSTANCE':
            action, role, node = state[1], state[2], state[3]
            arity = self.action_arity.get(action, 1)

            # Check transitions at leaf
            if self.instance_trie.is_leaf_node(node):
                if role < arity - 1 and token == self.obj_sep_seq[0]:
                    if len(self.obj_sep_seq) == 1:
                        return ('EXPECT_INSTANCE', action, role + 1, 0)
                    return ('OBJ_SEP', action, 1, role)
                if role == arity - 1:
                    dest = self._close_dict_alts.get(token)
                    if dest == 'close_dict':
                        return ('AFTER_DICT',)
                    elif dest == 'close_dict_sep':
                        # "}, ... — dict closed + some prefix of the separator
                        # consumed; resume after the pieces the token covered
                        # (1 = comma only; 2 for '"},{"', etc.).
                        pos = self._close_dict_sep_pos.get(token, 1)
                        if pos >= len(self.sep_seq):
                            return ('PREFIX', len(self.prefix_seq))  # action start
                        return ('SEP_SEQ', pos)
                    elif dest == 'close_dict_end':
                        return ('END',)  # "}] — closed everything
            # Continue instance trie
            next_node = self.instance_trie.next_node(node, token)
            if next_node is not None:
                viable = self._viable_instance_nodes(action, role)
                if viable is not None and next_node not in viable:
                    return None
                return ('EXPECT_INSTANCE', action, role, next_node)
            return None

        if phase == 'OBJ_SEP':
            action, idx, role = state[1], state[2], state[3]
            if idx < len(self.obj_sep_seq) and token == self.obj_sep_seq[idx]:
                if idx + 1 >= len(self.obj_sep_seq):
                    return ('EXPECT_INSTANCE', action, role + 1, 0)
                return ('OBJ_SEP', action, idx + 1, role)
            return None

        if phase == 'AFTER_DICT':
            if token in self._after_dict_ws_tokens:
                run = self._after_dict_ws_run(state)
                if run >= self.MAX_AFTER_DICT_WS:
                    return None
                return ('AFTER_DICT', run + 1)  # consume whitespace (bounded)
            dest = self._after_dict_alts.get(token)
            if dest == 'sep':
                pos = self._after_dict_sep_pos.get(token, 1)
                if pos >= len(self.sep_seq):
                    return ('PREFIX', len(self.prefix_seq))  # action start
                return ('SEP_SEQ', pos)
            elif dest == 'close':
                return ('END',)
            return None

        if phase == 'SEP_SEQ':
            idx = state[1]
            if idx < len(self.sep_seq) and token == self.sep_seq[idx]:
                if idx + 1 >= len(self.sep_seq):
                    return ('PREFIX', len(self.prefix_seq))  # action start
                return ('SEP_SEQ', idx + 1)
            return None

        if phase == 'END':
            if token in self.eos_tokens:
                return ('END',)
            return None

        return None

    # =========================================================================
    # Two-Level API
    # =========================================================================

    DEAD = ('DEAD',)

    def initial(self) -> Any:
        return self.initial_state()

    def step(self, state: Any, token: int) -> Any:
        next_s = self.next_state(state, token)
        return next_s if next_s is not None else self.DEAD

    # Consecutive pure-whitespace tokens allowed between an action dict and the
    # following ',' or ']' (pretty-printing). Unbounded, this self-loop is a
    # cycle neither factor prices, and a sharp LM can spin in it until the
    # token cap truncates the plan. Same bound as the VirtualHome gamma
    # (max_ws_tokens=5).
    MAX_AFTER_DICT_WS = 5

    @staticmethod
    def _after_dict_ws_run(state: Any) -> int:
        return int(state[1]) if isinstance(state, tuple) and len(state) > 1 else 0

    def get_phase(self, state: Any) -> str:
        if isinstance(state, tuple) and state:
            return str(state[0])
        return str(state)


def build_beh_twolevel_state_machine(
    tokenizer, prompt: str, eos_token_id: int,
) -> BehaviorTwoLevelStateMachine:
    """Factory: parse prompt and build BEH two-level state machine."""
    import re as _re
    objects, predicates = parse_beh_objects(prompt)
    # {'name','category'} pairs for the shared taxonomy membership layer
    # (the interactable-objects block carries wordnet synsets per instance).
    # Take the LAST interactable-objects block — same convention as
    # parse_beh_objects — to skip the in-context example's scene.
    _blocks = _re.findall(
        r"^\s*interactable objects\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
        str(prompt or ""), _re.IGNORECASE | _re.DOTALL | _re.MULTILINE)
    _block = _blocks[-1] if _blocks else ""
    objects_meta = [
        {'name': n, 'category': c}
        for n, c in _re.findall(
            r"\{\s*['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]\s*,\s*"
            r"['\"]category['\"]\s*:\s*['\"]([^'\"]+)['\"]", _block)
    ]
    # Container-ROLE bases for the OPEN/CLOSE evidence rule: arg-1 of any
    # `inside` relation in the prompt's initial/target states (including
    # quantified goals — token lists are flat, container = tok[i+2]).
    # Quantified goals name containers by category synset (cabinet.n.01);
    # those resolve to every scene base with that category (so e.g.
    # bottom_cabinet instances are whitelisted — matching β's expansion).
    from ctrlg.beta.beh_state_machine import parse_beh_states
    from ctrlg.beta.beh_affordances import base_of as _base_of
    _cat_norm = lambda c: str(c).strip()
    cat_to_bases: Dict[str, Set[str]] = {}
    for om in objects_meta:
        cat_to_bases.setdefault(_cat_norm(om.get('category', '')), set()).add(
            _base_of(om['name']))
    init_states, target_states = parse_beh_states(prompt)
    container_role_bases = set()
    for pred_list in list(init_states) + list(target_states):
        if not pred_list:
            continue
        for i, tok in enumerate(pred_list):
            if tok == 'inside' and i + 2 < len(pred_list):
                cont = str(pred_list[i + 2])
                if '.n.' in cont:
                    container_role_bases |= cat_to_bases.get(cont, set())
                else:
                    container_role_bases.add(_base_of(cont))
    return BehaviorTwoLevelStateMachine(tokenizer, objects, predicates,
                                        eos_token_id,
                                        objects_meta=objects_meta,
                                        container_role_bases=container_role_bases)

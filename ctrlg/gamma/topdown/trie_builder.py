"""
Trie building utilities for top-down DFA compilation.

This module builds TokenTrie structures for:
- Action names (from hardcoded canonical lists)
- Object names (from prompt parsing)
- Object IDs per name (from prompt parsing)
- JSON keys like "action", "object" (hardcoded)
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple


class TokenTrie:
    """
    Trie for efficient token sequence matching.

    Used by DFA to recognize action names, object names, and IDs during generation.
    """

    def __init__(self):
        self.children: Dict[int, Dict[int, int]] = {0: {}}  # node_id -> {token_id: child_node_id}
        self.is_leaf: Set[int] = set()
        self.next_id: int = 1
        self.leaf_label: Dict[int, str] = {}

    def insert(self, seq: Iterable[int], label: Optional[str] = None) -> None:
        """Insert a token sequence into the trie with optional label."""
        node = 0
        for tok in seq:
            tok = int(tok)
            if tok not in self.children[node]:
                self.children[node][tok] = self.next_id
                self.children[self.next_id] = {}
                self.next_id += 1
            node = self.children[node][tok]
        self.is_leaf.add(node)
        if label is not None:
            self.leaf_label[node] = str(label)

    def next_node(self, node: int, tok: int) -> Optional[int]:
        """Get next node for token, or None if no transition."""
        return self.children.get(int(node), {}).get(int(tok), None)

    def next_tokens(self, node: int) -> Set[int]:
        """Get all valid next tokens from a node."""
        return set(self.children.get(int(node), {}).keys())

    def is_leaf_node(self, node: int) -> bool:
        """Check if node is a leaf (end of valid sequence)."""
        return int(node) in self.is_leaf

    def get_leaf_label(self, node: int) -> Optional[str]:
        """Get label at leaf node, or None."""
        return self.leaf_label.get(int(node), None)

    def all_nodes(self) -> Set[int]:
        """Get all node IDs in the trie."""
        nodes = set()
        nodes.add(0)  # Root
        for parent, children in self.children.items():
            nodes.add(parent)
            for child in children.values():
                nodes.add(child)
        return nodes


def canonical_tokenization(tokenizer, s: str) -> Tuple[int, ...]:
    """
    Return the canonical token-id sequence for string s using tokenizer.encode.
    This is the single deterministic tokenization used by γ DFA.
    """
    s = str(s)
    if not s:
        return tuple()
    try:
        return tuple(int(x) for x in tokenizer.encode(s, add_special_tokens=False))
    except Exception:
        return tuple()


class TrieBuilder:
    """Builder for creating tries from action lists and object data."""

    @staticmethod
    def build_action_trie(tokenizer, actions: Iterable[str]) -> TokenTrie:
        """
        Build trie for action names (hardcoded canonical list).

        Args:
            tokenizer: HuggingFace tokenizer
            actions: Iterable of action names (e.g., VH_ACTIONS)

        Returns:
            TokenTrie with action names as leaves
        """
        trie = TokenTrie()
        for action in actions:
            action_str = str(action).upper()
            seq = canonical_tokenization(tokenizer, action_str)
            if seq:
                trie.insert(seq, label=action_str)
        return trie

    @staticmethod
    def build_name_trie(tokenizer, object_names: Iterable[str]) -> TokenTrie:
        """
        Build trie for object names (parsed from prompt).

        Args:
            tokenizer: HuggingFace tokenizer
            object_names: Iterable of object names (e.g., ["washing_machine", "clothes_pants"])

        Returns:
            TokenTrie with object names as leaves
        """
        trie = TokenTrie()
        for name in object_names:
            name_str = str(name)
            seq = canonical_tokenization(tokenizer, name_str)
            if seq:
                trie.insert(seq, label=name_str)
        return trie

    @staticmethod
    def build_id_tries(
        tokenizer,
        objects: Dict[str, List[str]],
    ) -> Dict[str, TokenTrie]:
        """
        Build per-name ID tries (parsed from prompt).

        Args:
            tokenizer: HuggingFace tokenizer
            objects: {name: [id1, id2, ...]} mapping

        Returns:
            {name: TokenTrie} where each trie contains the IDs for that name
        """
        id_tries = {}
        for name, ids in objects.items():
            name_str = str(name)
            trie = TokenTrie()
            for obj_id in ids:
                id_str = str(obj_id)
                seq = canonical_tokenization(tokenizer, id_str)
                if seq:
                    trie.insert(seq, label=id_str)
            if trie.is_leaf:  # Only include if has at least one ID
                id_tries[name_str] = trie
        return id_tries

def build_vh_tries(
    tokenizer,
    actions: Iterable[str],
    objects: Dict[str, List[str]],
) -> Tuple[TokenTrie, TokenTrie, Dict[str, TokenTrie]]:
    """
    Build all tries for VirtualHome DFA.

    Args:
        tokenizer: HuggingFace tokenizer
        actions: Iterable of action names (from action_defs.VH_ACTIONS)
        objects: {name: [id1, id2, ...]} mapping (from prompt parsing)

    Returns:
        (action_trie, name_trie, id_trie_by_name)
    """
    action_trie = TrieBuilder.build_action_trie(tokenizer, actions)
    name_trie = TrieBuilder.build_name_trie(tokenizer, objects.keys())
    id_tries = TrieBuilder.build_id_tries(tokenizer, objects)
    return action_trie, name_trie, id_tries


def compute_viable_name_nodes(
    name_trie: TokenTrie,
    id_trie_by_name: Dict[str, TokenTrie],
    allowed_objects: Optional[Set[str]],
) -> Optional[Set[int]]:
    """
    Compute set of name trie nodes that lead to allowed objects.

    Used for affordance filtering: only show name tokens that can lead
    to objects satisfying the affordance constraint.

    Args:
        name_trie: Trie of object names
        id_trie_by_name: Per-name ID tries
        allowed_objects: Set of allowed object names, or None for no restriction

    Returns:
        Set of viable node IDs, or None if no restriction
    """
    if allowed_objects is None:
        return None

    allowed_set = {str(n) for n in allowed_objects}
    viable = set()

    def dfs(node: int) -> bool:
        """Returns True if this node can reach an allowed leaf."""
        reaches_allowed = False

        # Check if this is a leaf with allowed label
        if name_trie.is_leaf_node(node):
            label = name_trie.get_leaf_label(node)
            if label is not None and label in allowed_set:
                # Also check that this name has IDs
                if label in id_trie_by_name:
                    reaches_allowed = True

        # Check children
        for child in name_trie.children.get(node, {}).values():
            if dfs(child):
                reaches_allowed = True

        if reaches_allowed:
            viable.add(node)

        return reaches_allowed

    dfs(0)
    return viable


def get_canonical_token_seqs(
    tokenizer,
    actions: Iterable[str],
) -> Dict[str, Tuple[int, ...]]:
    """
    Get canonical token sequences for all actions.

    Used by candidate tracking to build deterministic content witnesses.

    Args:
        tokenizer: HuggingFace tokenizer
        actions: Iterable of action names

    Returns:
        {action_name: (tok1, tok2, ...)} mapping
    """
    result = {}
    for action in actions:
        action_str = str(action).upper()
        seq = canonical_tokenization(tokenizer, action_str)
        if seq:
            result[action_str] = seq
    return result


def get_name_token_seqs(
    tokenizer,
    object_names: Iterable[str],
) -> Dict[str, Tuple[int, ...]]:
    """
    Get canonical token sequences for all object names.

    Args:
        tokenizer: HuggingFace tokenizer
        object_names: Iterable of object names

    Returns:
        {name: (tok1, tok2, ...)} mapping
    """
    result = {}
    for name in object_names:
        name_str = str(name)
        seq = canonical_tokenization(tokenizer, name_str)
        if seq:
            result[name_str] = seq
    return result


def get_id_token_seqs(
    tokenizer,
    objects: Dict[str, List[str]],
) -> Dict[Tuple[str, str], Tuple[int, ...]]:
    """
    Get canonical token sequences for all (name, id) pairs.

    Args:
        tokenizer: HuggingFace tokenizer
        objects: {name: [id1, id2, ...]} mapping

    Returns:
        {(name, id): (tok1, tok2, ...)} mapping
    """
    result = {}
    for name, ids in objects.items():
        name_str = str(name)
        for obj_id in ids:
            id_str = str(obj_id)
            seq = canonical_tokenization(tokenizer, id_str)
            if seq:
                result[(name_str, id_str)] = seq
    return result

"""DFAModel: a compiled token automaton as tensors, for masking LM logits.

Adapted from Ctrl-G (Zhang et al., 2024; see LICENSE).
"""
import torch
import torch.nn as nn


class DFAModel(nn.Module):
    def __init__(self, dfa_graph, vocab_size):
        super().__init__()

        edges = dfa_graph['edges']
        initial_state = dfa_graph['initial_state']
        accept_states = dfa_graph['accept_states']

        state_cnt, edge_cnt = 0, 0
        state2idx, edge2idx = {}, {}

        # pre-process dfa_graph
        for e in edges:
            u, v, _ = e
            for x in [u, v]:
                if x not in state2idx:
                    state2idx[x] = state_cnt
                    state_cnt += 1
            u_idx, v_idx = state2idx[u], state2idx[v]
            if (u_idx, v_idx) not in edge2idx:
                edge2idx[(u_idx, v_idx)] = edge_cnt
                edge_cnt += 1
            else:
                print('ERROR: duplicate edge!')
                exit(1)

        G = {}
        # Convert VE_mask, EV_mask, and T_mask to sparse tensors.
        # For large BEHAVIOR DFAs with 58K states × 116K edges, dense masks would be:
        # - VE_mask: 58K × 116K = 27 GB
        # - EV_mask: 116K × 58K = 27 GB
        # - T_mask: 116K × 128K = 25 GB
        # Total: ~79 GB just for masks!
        #
        # As sparse tensors (each edge connects exactly 1 source and 1 dest state):
        # - VE_mask: 116K non-zeros (one per edge) = ~1.4 MB
        # - EV_mask: 116K non-zeros (one per edge) = ~1.4 MB
        # - T_mask: ~11M non-zeros (~100 tokens/edge) = ~132 MB
        # Total: ~135 MB (580x smaller!)

        ve_mask_indices = []  # (state_idx, edge_idx) for VE connections
        ev_mask_indices = []  # (edge_idx, state_idx) for EV connections
        t_mask_indices = []   # (edge_idx, token_id) for allowed tokens

        E2Src = torch.tensor([0] * edge_cnt)
        E2Dst = torch.tensor([0] * edge_cnt)
        for e in edges:
            u, v, transition = e    # transition should be a bitset of tokens
            u_idx, v_idx = state2idx[u], state2idx[v]
            edge_idx = edge2idx[(u_idx, v_idx)]

            # Build sparse VE_mask and EV_mask indices
            ve_mask_indices.append([u_idx, edge_idx])  # Source state u_idx connects to edge_idx
            ev_mask_indices.append([edge_idx, v_idx])  # Edge edge_idx connects to dest state v_idx

            # Extract allowed token IDs from bitset and add to sparse T_mask indices
            # transition is a numpy bool array where True means token is allowed
            allowed_tokens = transition.nonzero()[0]  # Get indices where transition==True
            for tok in allowed_tokens:
                t_mask_indices.append([edge_idx, int(tok)])

            E2Src[edge_idx] = u_idx
            E2Dst[edge_idx] = v_idx

            if u_idx not in G:
                G[u_idx] = []
            G[u_idx].append((v_idx, transition))

        # Build sparse COO tensors for all three masks
        # Helper function to create sparse tensor from indices list
        def _make_sparse(indices_list, shape):
            if len(indices_list) > 0:
                indices_tensor = torch.LongTensor(indices_list).t()  # Shape: (2, num_nonzeros)
                values = torch.ones(indices_tensor.shape[1], dtype=torch.float32)
                sparse_tensor = torch.sparse_coo_tensor(indices_tensor, values, shape, dtype=torch.float32)
            else:
                # Empty case: create sparse tensor with no non-zero entries
                sparse_tensor = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros(0, dtype=torch.float32),
                    shape,
                    dtype=torch.float32
                )
            return sparse_tensor.coalesce()

        # Build sparse T_mask (critical for memory - 25GB → 132MB)
        T_mask = _make_sparse(t_mask_indices, (edge_cnt, vocab_size))

        # Edges are stored as index tensors instead of dense
        # state-by-edge masks:
        #   (E2Src == state_id)          → outgoing edges of a state
        #   F.one_hot(E2Dst, num_states) → destination states of edges
        # This takes O(num_edges) memory (under 1 MB for the largest
        # BEHAVIOR AS automaton, where dense masks would need ~55 GB).

        # Store sparse T_mask as plain attribute (not Parameter) to avoid densification
        self.T_mask = T_mask
        self.E2Src = nn.Parameter(E2Src, requires_grad=False)
        self.E2Dst = nn.Parameter(E2Dst, requires_grad=False)

        self.G = G
        self.num_states = state_cnt
        # Compiled state id lookups for the two-level processor / block-validity
        # table. `build_graph` emits edges over integer build ids and returns
        # `state_map` (state-machine state -> build id); compose so that
        # `state2idx` is keyed by the tracker's state-machine states.
        self.buildid2idx = state2idx
        smap = dfa_graph.get('state_map', None)
        if smap:
            self.state2idx = {st: state2idx[bid] for st, bid in smap.items() if bid in state2idx}
        else:
            self.state2idx = dict(state2idx)
        self.initial_state = state2idx[initial_state]
        self.accept_states = set([state2idx[x] for x in accept_states])
        # Optional debug helpers
        self.id2state = dfa_graph.get('id2state', None)
        self.vocab_size = vocab_size

    def to(self, *args, **kwargs):
        """
        Override .to() to properly handle sparse T_mask tensor.

        PyTorch's default .to() densifies sparse tensors when moving to GPU via _apply(),
        which would cause 25GB memory allocation instead of ~60MB for large DFAs.
        We manually move each tensor/parameter to preserve T_mask sparsity.
        """
        # Extract device from args/kwargs (handle both .to(device) and .to('cuda:0') signatures)
        device = None
        if args:
            device = args[0]
        elif 'device' in kwargs:
            device = kwargs['device']

        if device is not None:
            # Move sparse T_mask while preserving sparsity
            if hasattr(self, 'T_mask') and self.T_mask.is_sparse:
                self.T_mask = self.T_mask.to(device)

            # Move the dense edge-endpoint tensors
            self.E2Src = nn.Parameter(self.E2Src.to(device), requires_grad=False)
            self.E2Dst = nn.Parameter(self.E2Dst.to(device), requires_grad=False)

        return self

    def next_state(self, state, token):
        for e in self.G.get(state, []):
            v, transition_set = e
            if transition_set[token]:
                return v
        # No valid transition; return None to signal a dead prefix instead of exiting
        return None

    def is_accept(self, state):
        return state in self.accept_states

    def get_valid_tokens(self, state) -> 'torch.BoolTensor':
        """
        Get boolean mask of valid tokens from the given state.

        Args:
            state: DFA state index

        Returns:
            Boolean tensor of shape [vocab_size] where True = valid token
        """
        valid_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for dst, bitset in self.G.get(state, []):
            # bitset is numpy bool array or torch tensor
            if hasattr(bitset, 'nonzero'):
                # numpy array
                valid_indices = bitset.nonzero()[0]
            else:
                valid_indices = bitset.nonzero(as_tuple=True)[0]
            valid_mask[valid_indices] = True
        return valid_mask

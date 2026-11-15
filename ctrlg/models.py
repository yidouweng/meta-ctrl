"""Loading the meta-token HMM and computing its prompt-conditioned initial state."""
import math
from pathlib import Path
from typing import Optional, Tuple

import torch


class SemanticHMM:
    """Meta-token HMM parameters used by the backward DP.

    alpha_exp: [H, H] transition probabilities. beta: [H, V] log emissions,
    padded with -1e9 columns. Trained HMMs also carry a linear map from the
    LM's last hidden state at the end of the prompt to the initial-state logits.
    """

    def __init__(self, alpha_exp, beta, hidden_size, nn_proj=None, nn_bias=None,
                 trained_vocab_size: Optional[int] = None, source_dir: Optional[str] = None):
        self.alpha_exp = alpha_exp
        self.beta = beta
        self.hidden_size = hidden_size
        self.nn_proj = nn_proj
        self.nn_bias = nn_bias
        self.trained_vocab_size = trained_vocab_size
        self.source_dir = source_dir

    def initial_belief(self, prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Log initial distribution over hidden states: prompt-conditioned if the
        HMM has the linear map and an embedding is given, uniform otherwise."""
        if self.nn_proj is not None and self.nn_bias is not None and prompt_emb is not None:
            emb = prompt_emb.to(device=self.nn_proj.device, dtype=self.nn_proj.dtype)
            return torch.log_softmax(self.nn_proj @ emb + self.nn_bias, dim=-1)
        device = self.nn_proj.device if self.nn_proj is not None else self.alpha_exp.device
        H = self.hidden_size
        return torch.full((H,), -math.log(H), device=device, dtype=torch.float32)


def _load_init_map(hmm_dir: Path, hidden_size: int, device: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    path = hmm_dir / "nn_checkpoint.pt"
    if not path.exists():
        return None, None
    sd = torch.load(path, map_location=device, weights_only=True)
    w, b = sd.get("proj.weight"), sd.get("proj.bias")
    if w is None or b is None or int(w.shape[0]) != int(hidden_size):
        return None, None
    return w.float().to(device), b.float().to(device)


def load_semantic_hmm(hmm_dir: str, device: str = "cuda", target_vocab_size: int = 200) -> SemanticHMM:
    """Load `hmm_params.pt` (and `nn_checkpoint.pt` if present) from `hmm_dir`."""
    d = Path(hmm_dir)
    ckpt = torch.load(d / "hmm_params.pt", map_location=device, weights_only=True)
    alpha_exp = torch.softmax(ckpt["log_alpha"].float().to(device), dim=1)
    log_beta = torch.log_softmax(ckpt["log_beta"].float().to(device), dim=1)
    H, V = log_beta.shape
    if V < target_vocab_size:
        pad = torch.full((H, target_vocab_size - V), -1e9, device=device, dtype=log_beta.dtype)
        log_beta = torch.cat([log_beta, pad], dim=1)
    w, b = _load_init_map(d, H, device)
    print(f"  meta-token HMM {d.name}: H={H}, V={V}, prompt-conditioned init={'yes' if w is not None else 'no'}")
    return SemanticHMM(alpha_exp, log_beta, H, nn_proj=w, nn_bias=b, trained_vocab_size=int(V), source_dir=str(d))


def compute_prompt_end_embedding(base_model, prompt_ids, device: str = "cuda") -> torch.Tensor:
    """LM's last-layer hidden state at the last prompt token, the input of the
    initial-state map. `prompt_ids` are the decoder's prompt (ctrlg.prompts),
    the same ids the HMM's training samples were drawn from."""
    ids = torch.tensor([list(prompt_ids)], device=device)
    with torch.no_grad():
        out = base_model(ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0, -1].detach().float()

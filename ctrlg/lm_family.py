"""Base-LM specifics. Llama 3 goes through unchanged; gpt-oss-20b (harmony format)
writes an analysis channel, then `<|channel|>final<|message|>`, the answer and
`<|return|>`. Constraints apply to the answer only (TwoLevelProcessor warmup).
Prompts match the gpt-oss meta-HMM training samples: reasoning effort "low",
"Current date:" pinned to the sampling date."""
from typing import Any, Dict, List

import torch

from . import prompts as P

HARMONY_SIGNPOST = "<|channel|>final<|message|>"
HARMONY_FORCE = "<|end|><|start|>assistant<|channel|>final<|message|>"
HARMONY_WARMUP_BLOCK = ("<|return|>", "<|endoftext|>", "<|call|>")
REASONING_EFFORT = "low"
HARMONY_DATE = "2026-09-07"
HARMONY_DATE_BY_MODULE = {("behavior", "SD"): "2026-09-08"}
WARMUP_MAX_TOKENS = 1024


def is_harmony(name_or_tokenizer) -> bool:
    s = name_or_tokenizer if isinstance(name_or_tokenizer, str) else getattr(name_or_tokenizer, "name_or_path", "")
    s = (s or "").lower()
    return "gpt-oss" in s or "gpt_oss" in s


def load_harmony_lm(base_model: str, device: str):
    """gpt-oss with MXFP4 experts (triton kernels); refuses the ~42 GB bf16 dequantization."""
    from transformers import AutoModelForCausalLM, Mxfp4Config
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": device},
                                                 quantization_config=Mxfp4Config(dequantize=False))
    if getattr(model.config, "quantization_config", None) is not None and \
            getattr(model.config.quantization_config, "dequantize", False):
        raise RuntimeError("gpt-oss was dequantized to bf16; install `kernels` and triton>=3.4 for MXFP4")
    model.eval()
    return model


def prompt_ids(tokenizer, benchmark: str, module: str, llm_prompt: str) -> List[int]:
    if not is_harmony(tokenizer):
        return P.prompt_ids(tokenizer, benchmark, module, llm_prompt)
    date = HARMONY_DATE_BY_MODULE.get((benchmark, module), HARMONY_DATE)
    text = P.chat_text(tokenizer, benchmark, module, llm_prompt, REASONING_EFFORT, date)
    if f"Current date: {date}" not in text or "Reasoning: low" not in text:
        raise RuntimeError("harmony prompt lacks the pinned date or reasoning effort")
    return P.prompt_ids(tokenizer, benchmark, module, llm_prompt, REASONING_EFFORT, date)


def _ids(tokenizer, s: str) -> List[int]:
    return [int(t) for t in tokenizer.encode(s, add_special_tokens=False)]


def eos_id(tokenizer) -> int:
    """`<|return|>` for gpt-oss; for Llama `<|eot_id|>` (else the tokenizer's EOS)."""
    vocab = tokenizer.get_vocab()
    if is_harmony(tokenizer):
        return int(vocab["<|return|>"])
    return int(vocab["<|eot_id|>"]) if "<|eot_id|>" in vocab else int(tokenizer.eos_token_id)


def processor_kwargs(tokenizer) -> Dict[str, Any]:
    if not is_harmony(tokenizer):
        return {}
    vocab = tokenizer.get_vocab()
    return dict(pre_dfa_signpost=_ids(tokenizer, HARMONY_SIGNPOST),
                warmup_force_ids=_ids(tokenizer, HARMONY_FORCE),
                warmup_block_ids=[int(vocab[t]) for t in HARMONY_WARMUP_BLOCK if t in vocab],
                warmup_max_tokens=WARMUP_MAX_TOKENS)


def generation_budget(tokenizer, max_new_tokens: int) -> int:
    return max_new_tokens + WARMUP_MAX_TOKENS + 16 if is_harmony(tokenizer) else max_new_tokens


def generate_kwargs(tokenizer) -> Dict[str, Any]:
    """gpt-oss: stop only on `<|return|>` (its generation config also stops on <|endoftext|>, <|call|>)."""
    if not is_harmony(tokenizer):
        return {}
    return dict(eos_token_id=eos_id(tokenizer))


def prefill_final(tokenizer, prompt_ids: List[int]) -> List[int]:
    """Syntax-mask-only decoding: open the final channel in the prompt."""
    if not is_harmony(tokenizer):
        return list(prompt_ids)
    return list(prompt_ids) + _ids(tokenizer, HARMONY_SIGNPOST)


def answer_text(tokenizer, generated_ids) -> str:
    """The evaluated text: for gpt-oss only what follows the last final-channel signpost."""
    if not is_harmony(tokenizer):
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen = [int(t) for t in generated_ids]
    sp = _ids(tokenizer, HARMONY_SIGNPOST)
    start = 0
    for i in range(len(gen) - len(sp), -1, -1):
        if gen[i:i + len(sp)] == sp:
            start = i + len(sp)
            break
    return tokenizer.decode(gen[start:], skip_special_tokens=True).strip()

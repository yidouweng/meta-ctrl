"""The chat prompt every model input is built from.

The LM during decoding, the meta-token HMM's initial-state embedding and the
HMM's training samples (training/sample.py) all see the same prompt: the
module's system prompt (data/prompts/system_prompts.json) as a
system turn and the benchmark prompt as the user turn, in the LM's chat
template. The training samples were drawn by passing this text to vLLM, which
tokenizes it with special tokens added; `prompt_ids` tokenizes it the same
way, so decoding sees exactly the token ids the samples were drawn from
(for Llama 3 this includes a doubled <|begin_of_text|>).
"""
import json
from typing import List

SYSTEM_PROMPTS = "data/prompts/system_prompts.json"


def system_prompt(benchmark: str, module: str) -> str:
    """System prompt of `benchmark` ("virtualhome" or "behavior") and `module` ("AS" or "SD")."""
    with open(SYSTEM_PROMPTS) as f:
        lines = json.load(f)["modules"][benchmark][module]["system_prompt_lines"]
    return "\n".join(str(x) for x in lines).strip()


def chat_text(tokenizer, benchmark: str, module: str, llm_prompt: str) -> str:
    """The chat-formatted prompt as text, ending with the assistant header."""
    messages = [{"role": "system", "content": system_prompt(benchmark, module)},
                {"role": "user", "content": str(llm_prompt)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_ids(tokenizer, benchmark: str, module: str, llm_prompt: str) -> List[int]:
    """Token ids of `chat_text`, tokenized as vLLM tokenized the sampling prompts."""
    return list(tokenizer(chat_text(tokenizer, benchmark, module, llm_prompt), add_special_tokens=True)["input_ids"])

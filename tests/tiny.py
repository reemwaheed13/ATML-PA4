"""Shared CPU test fixtures: tiny random Qwen2 models and a character-level tokenizer
with a chat template.  No downloads.  Seed 6304."""

import torch

SEED = 6304
VOCAB = 300


def tiny_qwen(seed: int = SEED, vocab: int = VOCAB, layers: int = 2):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval()


class CharTokenizer:
    """ids = ord(ch) % 250 + 2; chat template mimics Qwen's <|im_start|> layout."""
    eos_token_id = 1
    pad_token_id = 1

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        text += "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, return_tensors=None):
        out = {"input_ids": [ord(ch) % 250 + 2 for ch in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i - 2) if 32 <= i - 2 < 127 else "?" for i in ids)

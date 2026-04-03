from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "train_sft_lib.py"
SPEC = importlib.util.spec_from_file_location("train_sft_lib", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_sft_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_sft_lib
SPEC.loader.exec_module(train_sft_lib)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(char) % 97 for char in text]}


def test_extract_prompt_response_for_instruction_schema():
    prompt, response = train_sft_lib.extract_prompt_response(
        {
            "instruction": "Write hello world",
            "input": "python",
            "output": "print('hello world')",
        }
    )
    assert "Instruction" in prompt
    assert "Input" in prompt
    assert response == "print('hello world')"


def test_extract_prompt_response_for_chat_schema():
    prompt, response = train_sft_lib.extract_prompt_response(
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a loop."},
                {"role": "assistant", "content": "for i in range(3): pass"},
            ]
        }
    )
    assert "System" in prompt
    assert "User" in prompt
    assert response == "for i in range(3): pass"


def test_tokenize_supervised_example_masks_prompt_tokens():
    tokenizer = FakeTokenizer()
    encoded = train_sft_lib.tokenize_supervised_example(
        tokenizer,
        prompt="Question",
        response="Answer",
        seq_length=128,
    )
    assert len(encoded["input_ids"]) == 128
    assert len(encoded["labels"]) == 128
    assert encoded["labels"][0] == -100
    assert any(value != -100 for value in encoded["labels"])

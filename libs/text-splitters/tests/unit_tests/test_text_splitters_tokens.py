"""Token-based text splitter tests."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import Tokenizer
from langchain_text_splitters.base import split_text_on_tokens

def test_split_text_on_tokens() -> None:
    """Test splitting by tokens per chunk."""
    text = "foo bar baz 123"

    tokenizer = Tokenizer(
        chunk_overlap=3,
        tokens_per_chunk=7,
        decode=(lambda it: "".join(chr(i) for i in it)),
        encode=(lambda it: [ord(c) for c in it]),
    )
    output = split_text_on_tokens(text=text, tokenizer=tokenizer)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_decode_returns_no_chunks() -> None:
    """Test that when decode returns only empty strings, output is empty, not ['']."""
    text = "foo bar baz 123"

    tokenizer = Tokenizer(
        chunk_overlap=3,
        tokens_per_chunk=7,
        decode=(lambda _: ""),
        encode=(lambda it: [ord(c) for c in it]),
    )
    output = split_text_on_tokens(text=text, tokenizer=tokenizer)
    expected_output: list[Any] = []
    assert output == expected_output



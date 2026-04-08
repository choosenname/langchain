"""JSON text splitter tests."""

from __future__ import annotations

import json
import random
import string
from typing import Any

from langchain_text_splitters.json import RecursiveJsonSplitter

def test_split_json() -> None:
    """Test json text splitter."""
    max_chunk = 800
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk)

    def random_val() -> str:
        return "".join(random.choices(string.ascii_letters, k=random.randint(4, 12)))

    test_data: Any = {
        "val0": random_val(),
        "val1": {f"val1{i}": random_val() for i in range(100)},
    }
    test_data["val1"]["val16"] = {f"val16{i}": random_val() for i in range(100)}

    # uses create_docs and split_text
    docs = splitter.create_documents(texts=[test_data])

    output = [len(doc.page_content) < max_chunk * 1.05 for doc in docs]
    expected_output = [True for doc in docs]
    assert output == expected_output


def test_split_json_with_lists() -> None:
    """Test json text splitter with list conversion."""
    max_chunk = 800
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk)

    def random_val() -> str:
        return "".join(random.choices(string.ascii_letters, k=random.randint(4, 12)))

    test_data: Any = {
        "val0": random_val(),
        "val1": {f"val1{i}": random_val() for i in range(100)},
    }
    test_data["val1"]["val16"] = {f"val16{i}": random_val() for i in range(100)}

    test_data_list: Any = {"testPreprocessing": [test_data]}

    # test text splitter
    texts = splitter.split_text(json_data=test_data)
    texts_list = splitter.split_text(json_data=test_data_list, convert_lists=True)

    assert len(texts_list) >= len(texts)


def test_split_json_many_calls() -> None:
    x = {"a": 1, "b": 2}
    y = {"c": 3, "d": 4}

    splitter = RecursiveJsonSplitter()
    chunk0 = splitter.split_json(x)
    assert chunk0 == [{"a": 1, "b": 2}]

    chunk1 = splitter.split_json(y)
    assert chunk1 == [{"c": 3, "d": 4}]

    # chunk0 is now altered by creating chunk1
    assert chunk0 == [{"a": 1, "b": 2}]

    chunk0_output = [{"a": 1, "b": 2}]
    chunk1_output = [{"c": 3, "d": 4}]

    assert chunk0 == chunk0_output
    assert chunk1 == chunk1_output


def test_split_json_with_empty_dict_values() -> None:
    """Test that empty dicts in JSON values are preserved, not dropped."""
    splitter = RecursiveJsonSplitter(max_chunk_size=300)

    data: dict[str, Any] = {
        "a": "hello",
        "b": {},
        "c": "world",
    }
    chunks = splitter.split_json(data)
    # Recombine all chunks into a single dict
    merged: dict[str, Any] = {}
    for chunk in chunks:
        merged.update(chunk)

    assert merged == {"a": "hello", "b": {}, "c": "world"}


def test_split_json_with_nested_empty_dicts() -> None:
    """Test that nested empty dicts are preserved."""
    splitter = RecursiveJsonSplitter(max_chunk_size=300)

    data: dict[str, Any] = {
        "level1": {
            "level2a": {},
            "level2b": "value",
        }
    }
    chunks = splitter.split_json(data)
    merged: dict[str, Any] = {}
    for chunk in chunks:
        merged.update(chunk)

    assert merged == {"level1": {"level2a": {}, "level2b": "value"}}


def test_split_json_empty_dict_only() -> None:
    """Test splitting a JSON that contains only an empty dict at the top level.

    An empty top-level dict should produce a single empty chunk (or no chunks).
    """
    splitter = RecursiveJsonSplitter(max_chunk_size=300)

    data: dict[str, Any] = {}
    chunks = splitter.split_json(data)
    # With nothing to split, result should be empty list
    assert chunks == []


def test_split_json_mixed_empty_and_nonempty_dicts() -> None:
    """Test a realistic structure mixing empty and non-empty nested dicts."""
    splitter = RecursiveJsonSplitter(max_chunk_size=300)

    data: dict[str, Any] = {
        "config": {},
        "metadata": {"author": "test", "tags": {}},
        "content": "some text",
    }
    chunks = splitter.split_json(data)
    merged: dict[str, Any] = {}
    for chunk in chunks:
        for k, v in chunk.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k].update(v)
            else:
                merged[k] = v

    assert merged["config"] == {}
    assert merged["metadata"] == {"author": "test", "tags": {}}
    assert merged["content"] == "some text"


def test_split_json_empty_dict_value_in_large_payload() -> None:
    """Test that empty dict values survive chunking in a larger payload."""
    max_chunk = 200
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk)

    data: dict[str, Any] = {
        "key0": "x" * 50,
        "empty": {},
        "key1": "y" * 50,
        "nested": {f"k{i}": f"v{i}" for i in range(20)},
    }
    chunks = splitter.split_json(data)

    for chunk in chunks:
        assert len(json.dumps(chunk)) < max_chunk * 1.05

    found_empty = False
    for chunk in chunks:
        if "empty" in chunk and chunk["empty"] == {}:
            found_empty = True
            break
        for v in chunk.values():
            if isinstance(v, dict) and "empty" in v and v["empty"] == {}:
                found_empty = True
                break
    assert found_empty, "Empty dict value was lost during splitting"

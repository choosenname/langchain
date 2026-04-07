from __future__ import annotations

from langchain_codex._types import JsonObject, get_nested_json_list, get_nested_str


def test_nested_json_helpers_follow_the_full_path() -> None:
    payload: JsonObject = {
        "params": {
            "item": {
                "delta": [{"type": "text", "text": "Hello"}],
            },
            "turn": {"id": "turn_123"},
        }
    }

    assert get_nested_json_list(payload, "params", "item", "delta") == [
        {"type": "text", "text": "Hello"}
    ]
    assert get_nested_str(payload, "params", "turn", "id") == "turn_123"

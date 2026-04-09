"""Low-level JSON-like helpers used by protocol modules."""

from __future__ import annotations

from typing import TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]


def as_json_object(value: object) -> JsonObject | None:
    """Return `value` as a JSON-like object when it has string keys."""
    if not isinstance(value, dict):
        return None
    raw_mapping = cast("dict[object, object]", value)
    for key in raw_mapping:
        if not isinstance(key, str):
            return None
    return cast("JsonObject", raw_mapping)


def get_json_object(payload: JsonObject, key: str) -> JsonObject | None:
    """Return a nested JSON object when present."""
    return as_json_object(payload.get(key))


def get_str(payload: JsonObject, key: str) -> str | None:
    """Return a nested string value when present."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def compact_none_values(payload: JsonObject) -> JsonObject:
    """Drop `None` values from a JSON-like dict."""
    return {key: value for key, value in payload.items() if value is not None}

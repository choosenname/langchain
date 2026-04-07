"""Internal typing helpers for Codex JSON-RPC payloads."""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, TypedDict, cast

JsonObject: TypeAlias = dict[str, object]


class TextInputItem(TypedDict):
    """Text input item sent to the Codex app server."""

    type: Literal["text"]
    text: str


class TextReader(Protocol):
    """Readable text stream used by the app-server transport."""

    def readline(self) -> str:
        """Return the next line from the stream."""
        ...


class TextWriter(Protocol):
    """Writable text stream used by the app-server transport."""

    def write(self, data: str) -> int:
        """Write text data to the stream."""
        ...

    def flush(self) -> None:
        """Flush any buffered text to the stream."""
        ...


class AppServerProcess(Protocol):
    """Subprocess protocol required by the transport."""

    stdin: TextWriter
    stdout: TextReader
    stderr: TextReader | None

    def poll(self) -> int | None:
        """Return the current process exit code when available."""
        ...


def as_json_object(value: object) -> JsonObject | None:
    """Return `value` as a JSON-like object when it is a string-keyed dict."""
    if not isinstance(value, dict):
        return None
    raw_mapping = cast("dict[object, object]", value)
    for key in raw_mapping:
        if not isinstance(key, str):
            return None
    return cast("JsonObject", raw_mapping)


def get_json_object(payload: JsonObject, key: str) -> JsonObject | None:
    """Return a nested JSON object from `payload` when present."""
    return as_json_object(payload.get(key))


def get_nested_json_object(payload: JsonObject, *path: str) -> JsonObject | None:
    """Return a nested JSON object from `payload` when the full path exists."""
    current: JsonObject | None = payload
    for key in path:
        if current is None:
            return None
        current = get_json_object(current, key)
    return current


def get_json_list(payload: JsonObject, key: str) -> list[object] | None:
    """Return a nested JSON list from `payload` when present."""
    value = payload.get(key)
    return cast("list[object]", value) if isinstance(value, list) else None


def get_str(payload: JsonObject, key: str) -> str | None:
    """Return a nested string value from `payload` when present."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def get_nested_str(payload: JsonObject, *path: str) -> str | None:
    """Return a nested string from `payload` when the full path exists."""
    if not path:
        return None

    current = payload
    for key in path[:-1]:
        current = get_json_object(current, key)
        if current is None:
            return None
    return get_str(current, path[-1])

"""Parsing for documented Codex input items and thread items."""

from __future__ import annotations

from langchain_codex.protocol.base import JsonObject, get_str
from langchain_codex.types import CodexInputItem, CodexThreadItem

DOCUMENTED_INPUT_ITEM_TYPES = (
    "text",
    "image",
    "localImage",
    "mention",
    "skill",
)

DOCUMENTED_THREAD_ITEM_TYPES = (
    "userMessage",
    "agentMessage",
    "plan",
    "reasoning",
    "commandExecution",
    "fileChange",
    "mcpToolCall",
    "collabToolCall",
    "webSearch",
    "imageView",
    "enteredReviewMode",
    "exitedReviewMode",
    "contextCompaction",
    "compacted",
    "dynamicToolCall",
)


def parse_input_item(payload: JsonObject) -> CodexInputItem:
    """Parse a documented input item."""
    kind = get_str(payload, "type") or "unknown"
    return CodexInputItem(kind=kind, raw=dict(payload))


def parse_thread_item(payload: JsonObject) -> CodexThreadItem:
    """Parse a documented thread item."""
    kind = get_str(payload, "type") or "unknown"
    item_id = get_str(payload, "id")
    return CodexThreadItem(kind=kind, raw=dict(payload), item_id=item_id)

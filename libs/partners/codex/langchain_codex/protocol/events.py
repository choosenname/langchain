"""Typed parsing for Codex app-server notifications."""

from __future__ import annotations

from langchain_codex.protocol.base import (
    JsonObject,
    as_json_object,
    get_json_object,
    get_str,
)
from langchain_codex.protocol.items import parse_thread_item
from langchain_codex.types import CodexEvent

DOCUMENTED_NOTIFICATION_METHODS = (
    "thread/started",
    "thread/status/changed",
    "thread/archived",
    "thread/unarchived",
    "thread/closed",
    "turn/started",
    "turn/completed",
    "turn/diff/updated",
    "turn/plan/updated",
    "thread/tokenUsage/updated",
    "model/rerouted",
    "item/started",
    "item/completed",
    "item/agentMessage/delta",
    "item/plan/delta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/reasoning/textDelta",
    "item/commandExecution/outputDelta",
    "item/fileChange/outputDelta",
    "item/autoApprovalReview/started",
    "item/autoApprovalReview/completed",
    "error",
    "serverRequest/resolved",
    "skills/changed",
    "app/list/updated",
    "fs/changed",
    "mcpServer/startupStatus/updated",
    "mcpServer/oauthLogin/completed",
    "thread/realtime/started",
    "thread/realtime/itemAdded",
    "thread/realtime/transcriptUpdated",
    "thread/realtime/outputAudio/delta",
    "thread/realtime/error",
    "thread/realtime/closed",
    "windowsSandbox/setupCompleted",
    "account/rateLimits/updated",
    "fuzzyFileSearch/sessionUpdated",
    "fuzzyFileSearch/sessionCompleted",
    "command/exec/outputDelta",
)


def parse_notification(message: JsonObject) -> CodexEvent:
    """Parse a notification payload into a typed event container."""
    method = get_str(message, "method") or "unknown"
    params = get_json_object(message, "params") or {}
    raw_item = as_json_object(params.get("item"))
    item = None if raw_item is None else parse_thread_item(raw_item)
    return CodexEvent(method=method, params=dict(params), raw=dict(message), item=item)

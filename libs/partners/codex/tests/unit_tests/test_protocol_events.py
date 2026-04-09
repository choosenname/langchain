from __future__ import annotations

from typing import cast

from langchain_codex.protocol import events as protocol_events


def test_documented_event_registry_tracks_app_server_notifications() -> None:
    expected_methods = {
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
        "item/reasoning/summaryTextDelta",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/textDelta",
        "item/commandExecution/outputDelta",
        "item/fileChange/outputDelta",
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
    }

    assert expected_methods <= set(protocol_events.DOCUMENTED_NOTIFICATION_METHODS)


def test_event_parser_supports_thread_turn_item_and_platform_notifications() -> None:
    notifications = [
        {
            "jsonrpc": "2.0",
            "method": "thread/started",
            "params": {"thread": {"id": "thr_123", "status": {"type": "idle"}}},
        },
        {
            "jsonrpc": "2.0",
            "method": "turn/completed",
            "params": {"turn": {"id": "turn_123", "status": "completed"}},
        },
        {
            "jsonrpc": "2.0",
            "method": "item/started",
            "params": {"item": {"id": "msg_123", "type": "agentMessage", "text": "hello"}},
        },
        {
            "jsonrpc": "2.0",
            "method": "fs/changed",
            "params": {"watchId": "watch_123", "changedPaths": ["/repo/README.md"]},
        },
        {
            "jsonrpc": "2.0",
            "method": "app/list/updated",
            "params": {"data": [{"id": "demo-app", "name": "Demo App"}]},
        },
        {
            "jsonrpc": "2.0",
            "method": "mcpServer/startupStatus/updated",
            "params": {"name": "docs", "status": "ready", "error": None},
        },
        {
            "jsonrpc": "2.0",
            "method": "thread/realtime/started",
            "params": {"threadId": "thr_123", "sessionId": "rt_123"},
        },
        {
            "jsonrpc": "2.0",
            "method": "account/rateLimits/updated",
            "params": {"rateLimits": [{"name": "requests", "remaining": 10}]},
        },
        {
            "jsonrpc": "2.0",
            "method": "serverRequest/resolved",
            "params": {"threadId": "thr_123", "requestId": 99},
        },
    ]

    parsed = [
        protocol_events.parse_notification(cast("dict[str, object]", message))
        for message in notifications
    ]

    assert [event.method for event in parsed] == [
        "thread/started",
        "turn/completed",
        "item/started",
        "fs/changed",
        "app/list/updated",
        "mcpServer/startupStatus/updated",
        "thread/realtime/started",
        "account/rateLimits/updated",
        "serverRequest/resolved",
    ]
    assert parsed[2].item is not None
    assert parsed[2].item.kind == "agentMessage"


def test_unknown_notification_parser_preserves_raw_payload() -> None:
    event = protocol_events.parse_notification(
        {
            "jsonrpc": "2.0",
            "method": "future/event",
            "params": {"value": 1},
        }
    )

    assert event.method == "future/event"
    assert event.params == {"value": 1}

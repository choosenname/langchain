from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from langchain_codex.session import CodexSession, TurnDelta


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.thread_start_calls = 0
        self._thread_id = "thr_1"
        self._notification_handlers: list[Callable[[dict[str, Any]], None]] = []
        self.turn_start_response: dict[str, Any] = {"turn": {"id": "turn_1"}}
        self.stream_text_chunks: list[str] | None = None
        self.completed_turn: dict[str, Any] = {"id": "turn_1", "status": "completed"}
        self.stream_notification_delay: float | None = None
        self.omit_turn_completed = False

    @classmethod
    def with_thread_id(cls, thread_id: str) -> FakeTransport:
        transport = cls()
        transport._thread_id = thread_id
        return transport

    def add_notification_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        self._notification_handlers.append(handler)

        def remove_handler() -> None:
            self._notification_handlers.remove(handler)

        return remove_handler

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "thread/start":
            self.thread_start_calls += 1
            return {"thread": {"id": self._thread_id}}
        if method == "turn/start":
            turn_id = "turn_1"
            if self.stream_text_chunks is not None:
                if self.stream_notification_delay is None:
                    self._emit_stream_notifications(turn_id)
                else:
                    threading.Thread(
                        target=self._emit_stream_notifications,
                        args=(turn_id,),
                        daemon=True,
                    ).start()
                return self.turn_start_response
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "metrics/updated",
                    "params": {"value": 1},
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/started",
                    "params": {"turn": {"id": turn_id}},
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/output",
                    "params": {
                        "turn": {"id": "turn_other"},
                        "event": {"type": "text", "text": "ignore me"},
                    },
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn_other"}},
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/output",
                    "params": {
                        "turn": {"id": turn_id},
                        "event": params["input"][0],
                    },
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {"turn": {"id": turn_id}},
                }
            )
            return self.turn_start_response
        return {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.requests.append((method, params))

    def _emit(self, message: dict[str, Any]) -> None:
        for handler in list(self._notification_handlers):
            handler(message)

    def _emit_stream_notifications(self, turn_id: str) -> None:
        if self.stream_notification_delay is not None:
            time.sleep(self.stream_notification_delay)
        self._emit(
            {
                "jsonrpc": "2.0",
                "method": "item/updated",
                "params": {
                    "turn": {"id": "turn_other"},
                    "item": {
                        "type": "agentMessage",
                        "delta": [{"type": "text", "text": "ignore me"}],
                    },
                },
            }
        )
        if self.stream_notification_delay is not None:
            time.sleep(self.stream_notification_delay)
        self._emit(
            {
                "jsonrpc": "2.0",
                "method": "item/updated",
                "params": {
                    "turn": {"id": turn_id},
                    "item": {
                        "type": "agentMessage",
                        "delta": [{"type": "tool_use", "name": "noop"}],
                    },
                },
            }
        )
        for text in self.stream_text_chunks or []:
            if self.stream_notification_delay is not None:
                time.sleep(self.stream_notification_delay)
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "item/updated",
                    "params": {
                        "turn": {"id": turn_id},
                        "item": {
                            "type": "agentMessage",
                            "delta": [{"type": "text", "text": text}],
                        },
                    },
                }
            )
        if self.stream_notification_delay is not None:
            time.sleep(self.stream_notification_delay)
        if not self.omit_turn_completed:
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {"turn": self.completed_turn},
                }
            )


def test_session_initializes_connection_once() -> None:
    transport = FakeTransport()
    session = CodexSession(transport=transport, model="gpt-5.4")

    session.ensure_started()
    session.ensure_started()

    assert transport.requests[:2] == [
        (
            "initialize",
            {
                "clientInfo": {
                    "name": "langchain_codex",
                    "title": "LangChain Codex",
                    "version": "0.1.0",
                }
            },
        ),
        ("initialized", {}),
    ]
    assert len(transport.requests) == 2


def test_session_defers_thread_creation_until_first_turn() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    session = CodexSession(transport=transport, model="gpt-5.4")

    assert transport.thread_start_calls == 0
    session.ensure_started()
    assert transport.thread_start_calls == 0
    session.run_turn([{"type": "text", "text": "hello"}])
    session.run_turn([{"type": "text", "text": "again"}])

    assert transport.thread_start_calls == 1


def test_session_keeps_existing_notification_handlers_active() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    seen: list[dict[str, Any]] = []
    transport.add_notification_handler(seen.append)
    session = CodexSession(transport=transport, model="gpt-5.4")

    result = session.run_turn([{"type": "text", "text": "hello"}])

    assert result == {
        "thread": {"id": "thr_123"},
        "turn": {"id": "turn_1"},
        "events": [
            {
                "jsonrpc": "2.0",
                "method": "turn/started",
                "params": {"turn": {"id": "turn_1"}},
            },
            {
                "jsonrpc": "2.0",
                "method": "turn/output",
                "params": {
                    "turn": {"id": "turn_1"},
                    "event": {"type": "text", "text": "hello"},
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "turn/completed",
                "params": {"turn": {"id": "turn_1"}},
            },
        ],
    }
    assert [
        (message["method"], message.get("params", {}).get("turn", {}).get("id"))
        for message in seen
    ] == [
        ("metrics/updated", None),
        ("turn/started", "turn_1"),
        ("turn/output", "turn_other"),
        ("turn/completed", "turn_other"),
        ("turn/output", "turn_1"),
        ("turn/completed", "turn_1"),
    ]


def test_session_raises_when_turn_start_missing_turn_id() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.turn_start_response = {}
    session = CodexSession(transport=transport, model="gpt-5.4")

    with pytest.raises(RuntimeError, match="turn/start response missing turn id"):
        session.run_turn([{"type": "text", "text": "hello"}])


def test_stream_turn_yields_item_agent_message_text_deltas_in_order() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.stream_text_chunks = ["Hel", "lo"]
    transport.turn_start_response = {"turn": {"id": "turn_1", "status": "in_progress"}}
    session = CodexSession(transport=transport, model="gpt-5.4")

    deltas = list(session.stream_turn([{"type": "text", "text": "hello"}]))

    assert deltas == [
        TurnDelta(text="Hel"),
        TurnDelta(text="lo"),
        TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_1", "status": "completed"},
            chunk_position="last",
        ),
    ]


@pytest.mark.asyncio
async def test_astream_turn_yields_item_agent_message_text_deltas_in_order() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.stream_text_chunks = ["A", "B"]
    transport.turn_start_response = {"turn": {"id": "turn_1", "status": "in_progress"}}
    session = CodexSession(transport=transport, model="gpt-5.4")

    deltas = [
        delta
        async for delta in session.astream_turn([{"type": "text", "text": "hello"}])
    ]

    assert deltas == [
        TurnDelta(text="A"),
        TurnDelta(text="B"),
        TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_1", "status": "completed"},
            chunk_position="last",
        ),
    ]


@pytest.mark.asyncio
async def test_astream_turn_completes_with_delayed_notifications() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.stream_text_chunks = ["A", "B"]
    transport.stream_notification_delay = 0.01
    transport.turn_start_response = {"turn": {"id": "turn_1", "status": "in_progress"}}
    session = CodexSession(transport=transport, model="gpt-5.4")

    deltas = await asyncio.wait_for(
        _collect_astream_turn(session.astream_turn([{"type": "text", "text": "hello"}])),
        timeout=0.5,
    )

    assert deltas == [
        TurnDelta(text="A"),
        TurnDelta(text="B"),
        TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_1", "status": "completed"},
            chunk_position="last",
        ),
    ]


@pytest.mark.asyncio
async def test_astream_turn_avoids_asyncio_to_thread_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.stream_text_chunks = ["A", "B"]
    transport.stream_notification_delay = 0.01
    transport.turn_start_response = {"turn": {"id": "turn_1", "status": "in_progress"}}
    session = CodexSession(transport=transport, model="gpt-5.4")

    async def fail_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        _ = func
        _ = args
        _ = kwargs
        msg = "asyncio.to_thread bridge should not be used"
        raise AssertionError(msg)

    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)

    deltas = await asyncio.wait_for(
        _collect_astream_turn(session.astream_turn([{"type": "text", "text": "hello"}])),
        timeout=0.5,
    )

    assert deltas == [
        TurnDelta(text="A"),
        TurnDelta(text="B"),
        TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_1", "status": "completed"},
            chunk_position="last",
        ),
    ]


def test_stream_turn_times_out_when_turn_never_completes() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.stream_text_chunks = ["A"]
    transport.omit_turn_completed = True
    transport.turn_start_response = {"turn": {"id": "turn_1", "status": "in_progress"}}
    session = CodexSession(transport=transport, model="gpt-5.4", turn_timeout=0.05)

    with pytest.raises(
        RuntimeError,
        match="Timed out waiting for Codex app-server to complete turn",
    ):
        list(session.stream_turn([{"type": "text", "text": "hello"}]))


@pytest.mark.asyncio
async def test_astream_turn_avoids_polling_sleep_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport.with_thread_id("thr_123")
    transport.stream_text_chunks = ["A", "B"]
    transport.stream_notification_delay = 0.01
    transport.turn_start_response = {"turn": {"id": "turn_1", "status": "in_progress"}}
    session = CodexSession(transport=transport, model="gpt-5.4")

    async def fail_sleep(delay: float, result: Any = None) -> Any:
        _ = delay
        _ = result
        msg = "polling sleep bridge should not be used"
        raise AssertionError(msg)

    monkeypatch.setattr(asyncio, "sleep", fail_sleep)

    deltas = await asyncio.wait_for(
        _collect_astream_turn(session.astream_turn([{"type": "text", "text": "hello"}])),
        timeout=0.5,
    )

    assert deltas == [
        TurnDelta(text="A"),
        TurnDelta(text="B"),
        TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_1", "status": "completed"},
            chunk_position="last",
        ),
    ]


async def _collect_astream_turn(
    stream: Any,
) -> list[TurnDelta]:
    return [delta async for delta in stream]

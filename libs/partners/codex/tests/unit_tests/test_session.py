from __future__ import annotations

from typing import Any, Callable

from langchain_codex.session import CodexSession


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.thread_start_calls = 0
        self._thread_id = "thr_1"
        self._notification_handler: Callable[[dict[str, Any]], None] | None = None

    @classmethod
    def with_thread_id(cls, thread_id: str) -> "FakeTransport":
        transport = cls()
        transport._thread_id = thread_id
        return transport

    def set_notification_handler(
        self,
        handler: Callable[[dict[str, Any]], None] | None,
    ) -> Callable[[dict[str, Any]], None] | None:
        previous_handler = self._notification_handler
        self._notification_handler = handler
        return previous_handler

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "thread/start":
            self.thread_start_calls += 1
            return {"thread": {"id": self._thread_id}}
        if method == "turn/start":
            turn_id = "turn_1"
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
            return {"turn": {"id": turn_id}}
        return {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.requests.append((method, params))

    def _emit(self, message: dict[str, Any]) -> None:
        if self._notification_handler is not None:
            self._notification_handler(message)


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


def test_session_collects_only_active_turn_events() -> None:
    transport = FakeTransport.with_thread_id("thr_123")
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

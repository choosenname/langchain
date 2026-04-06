from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from langchain_codex import ChatCodex


class FakeSession:
    def __init__(self, *, reply_text: str) -> None:
        self.reply_text = reply_text
        self.turn_count = 0
        self.input_items_calls: list[list[dict[str, Any]]] = []

    @classmethod
    def completed(cls, reply_text: str) -> "FakeSession":
        return cls(reply_text=reply_text)

    def run_turn(self, input_items: list[dict[str, Any]]) -> dict[str, Any]:
        self.turn_count += 1
        self.input_items_calls.append(input_items)
        return {
            "thread": {"id": "thr_123"},
            "turn": {"id": "turn_123", "status": "completed"},
            "events": [
                {
                    "jsonrpc": "2.0",
                    "method": "turn/output",
                    "params": {
                        "turn": {"id": "turn_123"},
                        "event": {
                            "type": "text",
                            "text": self.reply_text,
                            "input": input_items,
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn_123"}},
                },
            ],
        }


class SlowSessionFactory:
    def __init__(self) -> None:
        self.call_count = 0
        self.created_sessions: list[FakeSession] = []
        self.first_call_started = threading.Event()
        self.release_first_call = threading.Event()

    def __call__(self) -> FakeSession:
        self.call_count += 1
        session = FakeSession.completed(f"reply-{self.call_count}")
        self.created_sessions.append(session)
        if self.call_count == 1:
            self.first_call_started.set()
            self.release_first_call.wait(timeout=1)
        else:
            time.sleep(0.05)
        return session


def _make_model(
    *,
    fake_session: FakeSession | None = None,
    session_factory: Any | None = None,
) -> ChatCodex:
    model = ChatCodex(model="gpt-5.4")
    if fake_session is not None:
        model._session_factory = lambda: fake_session
    else:
        model._session_factory = session_factory
    model._session_instance = None
    return model


def test_invoke_returns_ai_message_with_codex_metadata() -> None:
    session = FakeSession.completed("Hello from Codex")
    model = _make_model(fake_session=session)

    result = model.invoke("Say hi")

    assert result.content == "Hello from Codex"
    assert result.response_metadata["model_provider"] == "codex"
    assert result.response_metadata["model"] == "gpt-5.4"
    assert result.response_metadata["thread_id"] == "thr_123"
    assert result.response_metadata["turn_id"] == "turn_123"
    assert result.response_metadata["turn_status"] == "completed"
    assert session.input_items_calls == [
        [{"type": "text", "text": "Human: Say hi"}],
    ]


@pytest.mark.asyncio
async def test_ainvoke_reuses_same_session() -> None:
    session = FakeSession.completed("Async reply")
    model = _make_model(fake_session=session)

    await model.ainvoke("one")
    await model.ainvoke("two")

    assert session.turn_count == 2
    assert session.input_items_calls == [
        [{"type": "text", "text": "Human: one"}],
        [{"type": "text", "text": "Human: two"}],
    ]


def test_invoke_drops_unsupported_kwargs_before_session() -> None:
    session = FakeSession.completed("Hello from Codex")
    model = _make_model(fake_session=session)

    result = model.invoke("Say hi", temperature=0.2)

    assert result.content == "Hello from Codex"
    assert session.input_items_calls == [
        [{"type": "text", "text": "Human: Say hi"}],
    ]


@pytest.mark.asyncio
async def test_concurrent_ainvoke_reuses_one_lazy_session() -> None:
    factory = SlowSessionFactory()
    model = _make_model(session_factory=factory)

    first_call = asyncio.create_task(model.ainvoke("one"))
    second_call = asyncio.create_task(model.ainvoke("two"))
    await asyncio.to_thread(factory.first_call_started.wait, 1)
    factory.release_first_call.set()
    await asyncio.gather(first_call, second_call)

    assert factory.call_count == 1
    assert len(factory.created_sessions) == 1
    assert factory.created_sessions[0].input_items_calls == [
        [{"type": "text", "text": "Human: one"}],
        [{"type": "text", "text": "Human: two"}],
    ]

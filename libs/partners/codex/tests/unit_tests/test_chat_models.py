from __future__ import annotations

from typing import Any

import pytest

from langchain_codex import ChatCodex


class FakeSession:
    def __init__(self, *, reply_text: str) -> None:
        self.reply_text = reply_text
        self.start_count = 0
        self.turn_count = 0

    @classmethod
    def completed(cls, reply_text: str) -> "FakeSession":
        return cls(reply_text=reply_text)

    def run_turn(self, input_items: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        if self.turn_count == 0:
            self.start_count += 1
        self.turn_count += 1
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


def _make_model(*, fake_session: FakeSession) -> ChatCodex:
    model = ChatCodex(model="gpt-5.4")
    model._session_factory = lambda: fake_session
    model._session_instance = None
    return model


def test_invoke_returns_ai_message_with_codex_metadata() -> None:
    model = _make_model(fake_session=FakeSession.completed("Hello from Codex"))

    result = model.invoke("Say hi")

    assert result.content == "Hello from Codex"
    assert result.response_metadata["model_provider"] == "codex"
    assert result.response_metadata["model"] == "gpt-5.4"
    assert result.response_metadata["thread_id"] == "thr_123"
    assert result.response_metadata["turn_id"] == "turn_123"
    assert result.response_metadata["turn_status"] == "completed"


@pytest.mark.asyncio
async def test_ainvoke_reuses_same_session() -> None:
    session = FakeSession.completed("Async reply")
    model = _make_model(fake_session=session)

    await model.ainvoke("one")
    await model.ainvoke("two")

    assert session.turn_count == 2
    assert session.start_count == 1

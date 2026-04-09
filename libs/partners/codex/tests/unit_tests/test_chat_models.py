from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_codex import ChatCodex
from langchain_codex.session import TurnDelta
from langchain_codex.types import (
    CodexThreadHandle,
    CodexTurnHandle,
    CodexTurnResult,
)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def run_turn(
        self,
        input_items: list[dict[str, object]],
        *,
        thread_id: str | None = None,
    ) -> CodexTurnResult:
        _ = thread_id
        self.calls.append(input_items)
        return CodexTurnResult(
            thread=CodexThreadHandle(thread_id="thr_123"),
            turn=CodexTurnHandle(turn_id="turn_123", status="completed"),
            events=(),
            output_text="hello",
        )

    def stream_turn(
        self,
        input_items: list[dict[str, object]],
        *,
        thread_id: str | None = None,
    ) -> Iterator[TurnDelta]:
        _ = thread_id
        self.calls.append(input_items)
        yield TurnDelta(text="Hel")
        yield TurnDelta(text="lo")
        yield TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_123", "status": "completed"},
            chunk_position="last",
        )

    async def astream_turn(
        self,
        input_items: list[dict[str, object]],
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[TurnDelta]:
        _ = thread_id
        self.calls.append(input_items)
        yield TurnDelta(text="Hel")
        yield TurnDelta(text="lo")
        yield TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_123", "status": "completed"},
            chunk_position="last",
        )


class FakeClient:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.create_session_calls = 0

    def create_session(self) -> FakeSession:
        self.create_session_calls += 1
        return self.session


def test_chat_codex_invoke_reuses_supplied_session_and_returns_metadata() -> None:
    session = FakeSession()
    model = ChatCodex(model="gpt-5.4", session=session)

    message = model.invoke([HumanMessage(content="hello")])

    assert message.text == "hello"
    metadata = cast("dict[str, object]", message.model_dump()["response_metadata"])
    assert metadata == {
        "model_provider": "codex",
        "model": "gpt-5.4",
        "thread_id": "thr_123",
        "turn_id": "turn_123",
        "turn_status": "completed",
    }
    assert session.calls == [[{"type": "text", "text": "Human: hello"}]]


def test_chat_codex_stream_surfaces_agent_message_text_chunks() -> None:
    session = FakeSession()
    model = ChatCodex(model="gpt-5.4", session=session)

    chunks = list(model.stream([HumanMessage(content="hello")]))

    assert [chunk.text for chunk in chunks] == ["Hel", "lo", ""]
    metadata = cast("dict[str, object]", chunks[-1].model_dump()["response_metadata"])
    assert metadata["thread_id"] == "thr_123"
    assert metadata["turn_id"] == "turn_123"


def test_chat_codex_ainvoke_and_astream_work_with_configured_client() -> None:
    session = FakeSession()
    client = FakeClient(session)
    model = ChatCodex(model="gpt-5.4", client=client)

    async def run() -> tuple[str, list[str]]:
        result = await model.ainvoke(
            [SystemMessage(content="be concise"), HumanMessage(content="hello")]
        )
        stream = [str(chunk.text) async for chunk in model.astream([HumanMessage(content="hello")])]
        return str(result.text), stream

    text, stream_chunks = asyncio.run(run())

    assert text == "hello"
    assert stream_chunks == ["Hel", "lo", ""]
    assert client.create_session_calls == 1
    assert session.calls[0] == [{"type": "text", "text": "System: be concise\nHuman: hello"}]

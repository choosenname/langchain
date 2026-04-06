from __future__ import annotations

import asyncio
import io
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Protocol, cast

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from langchain_codex._types import JsonObject, TextInputItem
from langchain_codex import ChatCodex
from langchain_codex.session import CodexSession, TurnDelta


class SessionLike(Protocol):
    def run_turn(self, input_items: list[TextInputItem]) -> JsonObject:
        ...

    def stream_turn(self, input_items: list[TextInputItem]) -> Iterator[TurnDelta]:
        ...

    def astream_turn(self, input_items: list[TextInputItem]) -> AsyncIterator[TurnDelta]:
        ...


class ChatCodexHarness(ChatCodex):
    def install_session_factory(
        self,
        session_factory: Callable[[], SessionLike] | None,
    ) -> None:
        self._session_factory = session_factory
        self._session_instance = None

    def build_session_for_test(self) -> object:
        return self._build_session()


class FakeSession:
    def __init__(
        self,
        *,
        reply_text: str = "",
        stream_text_chunks: list[str] | None = None,
        stream_completed_turn: JsonObject | None = None,
        stream_thread_id: str = "thr_123",
    ) -> None:
        self.reply_text = reply_text
        self.stream_text_chunks = stream_text_chunks or []
        self.stream_completed_turn = stream_completed_turn or {
            "id": "turn_123",
            "status": "completed",
        }
        self.stream_thread_id = stream_thread_id
        self.turn_count = 0
        self.input_items_calls: list[list[TextInputItem]] = []

    @classmethod
    def completed(cls, reply_text: str) -> FakeSession:
        return cls(reply_text=reply_text)

    @classmethod
    def stream(cls, text_chunks: list[str]) -> FakeSession:
        return cls(stream_text_chunks=text_chunks)

    @classmethod
    def stream_with_completed_turn(
        cls,
        text_chunks: list[str],
        *,
        thread_id: str = "thr_123",
        turn: JsonObject,
    ) -> FakeSession:
        return cls(
            stream_text_chunks=text_chunks,
            stream_completed_turn=turn,
            stream_thread_id=thread_id,
        )

    def run_turn(self, input_items: list[TextInputItem]) -> JsonObject:
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

    def stream_turn(self, input_items: list[TextInputItem]) -> Iterator[TurnDelta]:
        self.turn_count += 1
        self.input_items_calls.append(input_items)
        for text in self.stream_text_chunks:
            yield TurnDelta(text=text)
        yield TurnDelta(
            text="",
            thread_id=self.stream_thread_id,
            turn=self.stream_completed_turn,
            chunk_position="last",
        )

    async def astream_turn(
        self,
        input_items: list[TextInputItem],
    ) -> AsyncIterator[TurnDelta]:
        self.turn_count += 1
        self.input_items_calls.append(input_items)
        for text in self.stream_text_chunks:
            yield TurnDelta(text=text)
        yield TurnDelta(
            text="",
            thread_id=self.stream_thread_id,
            turn=self.stream_completed_turn,
            chunk_position="last",
        )


class DelayedStreamTransport:
    def __init__(self) -> None:
        self._thread_id = "thr_123"
        self._notification_handlers: list[Callable[[JsonObject], None]] = []
        self.turn_start_response: JsonObject = {"turn": {"id": "turn_1", "status": "in_progress"}}
        self.completed_turn: JsonObject = {"id": "turn_1", "status": "completed"}

    def add_notification_handler(
        self,
        on_notification: Callable[[JsonObject], None],
    ) -> Callable[[], None]:
        self._notification_handlers.append(on_notification)

        def remove_handler() -> None:
            self._notification_handlers.remove(on_notification)

        return remove_handler

    def request(self, method: str, params: JsonObject) -> JsonObject:
        if method == "initialize":
            return {}
        if method == "thread/start":
            return {"thread": {"id": self._thread_id}}
        if method == "turn/start":
            threading.Thread(target=self._emit_turn_notifications, daemon=True).start()
            return self.turn_start_response
        return {}

    def notify(self, method: str, params: JsonObject) -> None:
        _ = method
        _ = params

    def _emit_turn_notifications(self) -> None:
        for text in ["A", "B"]:
            time.sleep(0.01)
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "item/updated",
                    "params": {
                        "turn": {"id": "turn_1"},
                        "item": {
                            "type": "agentMessage",
                            "delta": [{"type": "text", "text": text}],
                        },
                    },
                }
            )
        time.sleep(0.01)
        self._emit(
            {
                "jsonrpc": "2.0",
                "method": "turn/completed",
                "params": {"turn": self.completed_turn},
            }
        )

    def _emit(self, message: JsonObject) -> None:
        for handler in list(self._notification_handlers):
            handler(message)


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


class DummyPopen:
    def __init__(self, args: list[str]) -> None:
        self.args = args
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()


def _make_model(
    *,
    fake_session: FakeSession | None = None,
    session_factory: Callable[[], SessionLike] | None = None,
) -> ChatCodexHarness:
    model = ChatCodexHarness(model="gpt-5.4")
    if fake_session is not None:
        model.install_session_factory(lambda: fake_session)
    else:
        model.install_session_factory(session_factory)
    return model


def _message_metadata(message: object) -> dict[str, object]:
    return cast(dict[str, object], getattr(message, "response_metadata"))


def _message_content(message: object) -> str:
    content = getattr(message, "content")
    assert isinstance(content, str)
    return content


def _patch_missing_binary(binary: str) -> str | None:
    _ = binary
    return None


def test_invoke_returns_ai_message_with_codex_metadata() -> None:
    session = FakeSession.completed("Hello from Codex")
    model = _make_model(fake_session=session)

    result = model.invoke("Say hi")
    metadata = _message_metadata(result)

    assert _message_content(result) == "Hello from Codex"
    assert metadata["model_provider"] == "codex"
    assert metadata["model"] == "gpt-5.4"
    assert metadata["thread_id"] == "thr_123"
    assert metadata["turn_id"] == "turn_123"
    assert metadata["turn_status"] == "completed"
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

    assert _message_content(result) == "Hello from Codex"
    assert session.input_items_calls == [
        [{"type": "text", "text": "Human: Say hi"}],
    ]


def test_missing_binary_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain_codex.chat_models.shutil.which", _patch_missing_binary)
    model = ChatCodex(model="gpt-5.4", codex_binary="missing-codex")

    with pytest.raises(RuntimeError, match=r"missing-codex.*PATH"):
        model.invoke("Say hi")


def test_build_session_uses_codex_command_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[list[str]] = []

    def fake_which(binary: str) -> str:
        assert binary == "ai-creds"
        return f"/usr/bin/{binary}"

    def fake_popen(
        args: list[str],
        *,
        stdin: Any,
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> DummyPopen:
        _ = stdin
        _ = stdout
        _ = stderr
        assert text is True
        popen_calls.append(args)
        return DummyPopen(args)

    monkeypatch.setattr("langchain_codex.chat_models.shutil.which", fake_which)
    monkeypatch.setattr("langchain_codex.chat_models.subprocess.Popen", fake_popen)
    def fake_transport(**kwargs: object) -> dict[str, object]:
        return kwargs

    def fake_session(**kwargs: object) -> dict[str, object]:
        return kwargs

    monkeypatch.setattr("langchain_codex.chat_models.CodexAppServerTransport", fake_transport)
    monkeypatch.setattr("langchain_codex.chat_models.CodexSession", fake_session)

    model = ChatCodexHarness(model="gpt-5.4", codex_command="ai-creds run codex")

    model.build_session_for_test()

    assert popen_calls == [["ai-creds", "run", "codex", "app-server"]]


def test_build_session_uses_default_codex_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[list[str]] = []

    def fake_which(binary: str) -> str:
        assert binary == "codex"
        return f"/usr/bin/{binary}"

    def fake_popen(
        args: list[str],
        *,
        stdin: Any,
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> DummyPopen:
        _ = stdin
        _ = stdout
        _ = stderr
        assert text is True
        popen_calls.append(args)
        return DummyPopen(args)

    monkeypatch.setattr("langchain_codex.chat_models.shutil.which", fake_which)
    monkeypatch.setattr("langchain_codex.chat_models.subprocess.Popen", fake_popen)
    def fake_transport(**kwargs: object) -> dict[str, object]:
        return kwargs

    def fake_session(**kwargs: object) -> dict[str, object]:
        return kwargs

    monkeypatch.setattr("langchain_codex.chat_models.CodexAppServerTransport", fake_transport)
    monkeypatch.setattr("langchain_codex.chat_models.CodexSession", fake_session)

    model = ChatCodexHarness(model="gpt-5.4")

    model.build_session_for_test()

    assert popen_calls == [["codex", "app-server"]]


def test_build_session_rejects_empty_codex_command() -> None:
    model = ChatCodexHarness(model="gpt-5.4", codex_command="")

    with pytest.raises(RuntimeError, match="codex_command"):
        model.build_session_for_test()


def test_build_session_rejects_invalid_codex_command() -> None:
    model = ChatCodexHarness(model="gpt-5.4", codex_command='"ai-creds run codex')

    with pytest.raises(RuntimeError, match="Invalid codex_command"):
        model.build_session_for_test()


def test_invoke_rejects_unsupported_message_content() -> None:
    model = _make_model(fake_session=FakeSession.completed("unused"))

    with pytest.raises(RuntimeError, match="only supports string message content"):
        model.invoke([HumanMessage(content=[{"type": "text", "text": "hi"}])])


def test_invoke_rejects_unsupported_message_type() -> None:
    model = _make_model(fake_session=FakeSession.completed("unused"))

    with pytest.raises(RuntimeError, match="does not support tool messages"):
        model.invoke([ToolMessage(content="tool output", tool_call_id="call_123")])


def test_stream_yields_text_chunks_in_order() -> None:
    model = _make_model(fake_session=FakeSession.stream(["Hel", "lo"]))

    chunks = list(model.stream("Say hello"))

    assert "".join(chunk.text for chunk in chunks) == "Hello"
    assert chunks[-1].chunk_position == "last"


def test_stream_surfaces_authoritative_completed_turn_state() -> None:
    model = _make_model(
        fake_session=FakeSession.stream_with_completed_turn(
            ["draft"],
            turn={"id": "turn_123", "status": "completed"},
        )
    )

    chunks = list(model.stream("Say hello"))
    metadata = _message_metadata(chunks[-1])

    assert "".join(chunk.text for chunk in chunks) == "draft"
    assert metadata["thread_id"] == "thr_123"
    assert metadata["turn_id"] == "turn_123"
    assert metadata["turn_status"] == "completed"
    assert chunks[-1].chunk_position == "last"


@pytest.mark.asyncio
async def test_astream_yields_text_chunks_in_order() -> None:
    model = _make_model(fake_session=FakeSession.stream(["A", "B"]))

    chunks = [chunk async for chunk in model.astream("letters")]

    assert "".join(chunk.text for chunk in chunks) == "AB"
    assert chunks[-1].chunk_position == "last"


@pytest.mark.asyncio
async def test_astream_surfaces_authoritative_completed_turn_state() -> None:
    model = _make_model(
        fake_session=FakeSession.stream_with_completed_turn(
            ["draft"],
            turn={"id": "turn_123", "status": "completed"},
        )
    )

    chunks = [chunk async for chunk in model.astream("letters")]
    metadata = _message_metadata(chunks[-1])

    assert "".join(chunk.text for chunk in chunks) == "draft"
    assert metadata["thread_id"] == "thr_123"
    assert metadata["turn_id"] == "turn_123"
    assert metadata["turn_status"] == "completed"
    assert chunks[-1].chunk_position == "last"


@pytest.mark.asyncio
async def test_astream_uses_real_codex_session_with_fake_transport() -> None:
    model = ChatCodexHarness(model="gpt-5.4")
    model.install_session_factory(lambda: CodexSession(
        transport=DelayedStreamTransport(),
        model="gpt-5.4",
    ))

    chunks = await asyncio.wait_for(
        _collect_astream_chunks(model.astream("letters")),
        timeout=0.5,
    )
    metadata = _message_metadata(chunks[-1])

    assert "".join(chunk.text for chunk in chunks) == "AB"
    assert metadata["thread_id"] == "thr_123"
    assert metadata["turn_id"] == "turn_1"
    assert metadata["turn_status"] == "completed"
    assert chunks[-1].chunk_position == "last"


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


async def _collect_astream_chunks(
    stream: AsyncIterator[AIMessageChunk],
) -> list[AIMessageChunk]:
    return [chunk async for chunk in stream]

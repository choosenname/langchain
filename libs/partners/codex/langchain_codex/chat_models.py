"""Codex chat models."""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Protocol

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_codex.errors import CodexError, CodexInputError
from langchain_codex.session import CodexSession, TurnDelta
from langchain_codex.transport import CodexAppServerTransport


class _CodexSessionLike(Protocol):
    def run_turn(self, input_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Run one app-server turn."""
        ...

    def stream_turn(self, input_items: list[dict[str, Any]]) -> Iterator[TurnDelta]:
        """Stream text deltas for one app-server turn."""
        ...

    def astream_turn(
        self,
        input_items: list[dict[str, Any]],
    ) -> AsyncIterator[TurnDelta]:
        """Asynchronously stream text deltas for one app-server turn."""
        ...


def _get_nested_string(payload: dict[str, Any], *path: str) -> str | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) else None


def _extract_turn_text(turn: dict[str, Any]) -> str:
    text_chunks: list[str] = []
    events = turn.get("events")
    if not isinstance(events, list):
        return ""

    for message in events:
        if not isinstance(message, dict):
            continue
        params = message.get("params")
        if not isinstance(params, dict):
            continue
        event = params.get("event")
        if not isinstance(event, dict):
            continue
        if event.get("type") != "text":
            continue
        text = event.get("text")
        if isinstance(text, str):
            text_chunks.append(text)

    return "".join(text_chunks)


def _response_metadata_from_delta(
    delta: TurnDelta,
    *,
    model_name: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_provider": "codex",
        "model": model_name,
    }
    thread_id = getattr(delta, "thread_id", None)
    if thread_id is not None:
        metadata["thread_id"] = thread_id
    turn = getattr(delta, "turn", None)
    if turn is not None:
        turn_id = _get_nested_string({"turn": turn}, "turn", "id")
        if turn_id is not None:
            metadata["turn_id"] = turn_id
        turn_status = _get_nested_string(
            {"turn": turn},
            "turn",
            "status",
        )
        if turn_status is not None:
            metadata["turn_status"] = turn_status
    return metadata


class ChatCodex(BaseChatModel):
    """LangChain chat model backed by a local Codex app-server session."""

    model_name: str = Field(alias="model")
    codex_binary: str = "codex"
    request_timeout: float | None = 30.0
    turn_timeout: float | None = 60.0

    model_config = ConfigDict(populate_by_name=True)

    _process: subprocess.Popen[str] | None = PrivateAttr(default=None)
    _session_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _session_factory: Callable[[], _CodexSessionLike] | None = PrivateAttr(default=None)
    _session_instance: _CodexSessionLike | None = PrivateAttr(default=None)

    def _session(self) -> _CodexSessionLike:
        if self._session_instance is not None:
            return self._session_instance
        with self._session_lock:
            if self._session_instance is None:
                if self._session_factory is not None:
                    self._session_instance = self._session_factory()
                else:
                    self._session_instance = self._build_session()
        return self._session_instance

    def _build_session(self) -> CodexSession:
        if shutil.which(self.codex_binary) is None:
            msg = (
                "Unable to launch Codex app-server because "
                f"{self.codex_binary!r} is not available on PATH."
            )
            raise CodexError(msg)

        process = subprocess.Popen(
            [self.codex_binary, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if process.stdin is None or process.stdout is None:
            msg = "Codex app-server process did not expose stdio pipes."
            raise CodexError(msg)
        self._process = process
        transport = CodexAppServerTransport(
            process=process,
            request_timeout=self.request_timeout,
        )
        return CodexSession(
            transport=transport,
            model=self.model_name,
            turn_timeout=self.turn_timeout,
        )

    def _to_input_items(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        rendered_messages: list[str] = []
        for message in messages:
            if not isinstance(message.content, str):
                msg = "ChatCodex only supports string message content."
                raise CodexInputError(msg)

            if message.type == "human":
                role = "Human"
            elif message.type == "ai":
                role = "AI"
            elif message.type == "system":
                role = "System"
            else:
                msg = f"ChatCodex does not support {message.type} messages."
                raise CodexInputError(msg)

            rendered_messages.append(f"{role}: {message.content}")

        return [{"type": "text", "text": "\n".join(rendered_messages)}]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = stop
        _ = run_manager
        _ = kwargs
        return self._create_chat_result(messages)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = stop
        _ = run_manager
        _ = kwargs
        return self._create_chat_result(messages)

    def _create_chat_result(self, messages: list[BaseMessage]) -> ChatResult:
        turn = self._session().run_turn(self._to_input_items(messages))
        message = AIMessage(
            content=_extract_turn_text(turn),
            response_metadata={
                "model_provider": "codex",
                "model": self.model_name,
                "thread_id": _get_nested_string(turn, "thread", "id"),
                "turn_id": _get_nested_string(turn, "turn", "id"),
                "turn_status": _get_nested_string(turn, "turn", "status"),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        _ = stop
        _ = kwargs
        for delta in self._session().stream_turn(self._to_input_items(messages)):
            message_chunk = AIMessageChunk(
                content=delta.text,
                response_metadata=_response_metadata_from_delta(
                    delta,
                    model_name=self.model_name,
                ),
                chunk_position=getattr(delta, "chunk_position", None),
            )
            generation_chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager is not None and delta.text:
                run_manager.on_llm_new_token(delta.text, chunk=generation_chunk)
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        _ = stop
        _ = kwargs
        async for delta in self._session().astream_turn(self._to_input_items(messages)):
            message_chunk = AIMessageChunk(
                content=delta.text,
                response_metadata=_response_metadata_from_delta(
                    delta,
                    model_name=self.model_name,
                ),
                chunk_position=getattr(delta, "chunk_position", None),
            )
            generation_chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager is not None and delta.text:
                await run_manager.on_llm_new_token(delta.text, chunk=generation_chunk)
            yield generation_chunk

    @property
    def _llm_type(self) -> str:
        return "codex-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "codex_binary": self.codex_binary,
            "request_timeout": self.request_timeout,
            "turn_timeout": self.turn_timeout,
        }

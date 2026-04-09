"""LangChain chat-model adapter for the Codex provider."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_codex.client import CodexClient
from langchain_codex.errors import CodexInputError
from langchain_codex.types import CodexClientConfig

if TYPE_CHECKING:
    from langchain_codex.session import TurnDelta
    from langchain_codex.types import CodexTurnResult


def _message_role_name(message: BaseMessage) -> str:
    if message.type == "human":
        return "Human"
    if message.type == "ai":
        return "AI"
    if message.type == "system":
        return "System"
    msg = f"ChatCodex does not support {message.type} messages."
    raise CodexInputError(msg)


def _render_message(message: BaseMessage) -> str:
    content = cast("object", cast("Any", message).content)
    if not isinstance(content, str):
        msg = "ChatCodex only supports string message content."
        raise CodexInputError(msg)
    return f"{_message_role_name(message)}: {content}"


class ChatCodex(BaseChatModel):
    """Thin LangChain adapter over `CodexSession`."""

    model_name: str = Field(alias="model")
    launch_command: tuple[str, ...] | list[str] = ("codex", "app-server")
    cwd: str | None = None
    approval_policy: str | None = "on-request"
    sandbox_policy: dict[str, object] | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    request_timeout: float | None = None
    turn_timeout: float | None = None
    client: Any | None = None
    session: Any | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    _session_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _session_instance: Any | None = PrivateAttr(default=None)

    def _session(self) -> Any:
        if self.session is not None:
            return self.session
        if self._session_instance is not None:
            return self._session_instance
        with self._session_lock:
            if self._session_instance is None:
                client = self.client if self.client is not None else self._build_client()
                self._session_instance = client.create_session()
        return self._session_instance

    def _build_client(self) -> CodexClient:
        config = CodexClientConfig(
            launch_command=tuple(self.launch_command),
            model=self.model_name,
            cwd=self.cwd,
            approval_policy=self.approval_policy,
            sandbox_policy=self.sandbox_policy,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            request_timeout=self.request_timeout,
            turn_timeout=self.turn_timeout,
        )
        return CodexClient(config)

    def _to_input_items(self, messages: list[BaseMessage]) -> list[dict[str, object]]:
        rendered_messages = [_render_message(message) for message in messages]
        return [{"type": "text", "text": "\n".join(rendered_messages)}]

    def _response_metadata(self, result: CodexTurnResult) -> dict[str, object]:
        return {
            "model_provider": "codex",
            "model": self.model_name,
            "thread_id": result.thread.thread_id,
            "turn_id": result.turn.turn_id,
            "turn_status": result.turn.status,
        }

    def _response_metadata_from_delta(self, delta: TurnDelta) -> dict[str, object]:
        metadata: dict[str, object] = {
            "model_provider": "codex",
            "model": self.model_name,
        }
        if delta.thread_id is not None:
            metadata["thread_id"] = delta.thread_id
        if delta.turn is not None:
            turn_id = delta.turn.get("id")
            turn_status = delta.turn.get("status")
            if isinstance(turn_id, str):
                metadata["turn_id"] = turn_id
            if isinstance(turn_status, str):
                metadata["turn_status"] = turn_status
        return metadata

    def _generation_chunk_from_delta(self, delta: TurnDelta) -> ChatGenerationChunk:
        message_chunk = AIMessageChunk(
            content=delta.text,
            response_metadata=self._response_metadata_from_delta(delta),
            chunk_position=delta.chunk_position,
        )
        return ChatGenerationChunk(message=message_chunk)

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
        result = self._session().run_turn(self._to_input_items(messages))
        message = AIMessage(
            content=result.output_text,
            response_metadata=self._response_metadata(result),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

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
        return self._generate(messages)

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
            generation_chunk = self._generation_chunk_from_delta(delta)
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
            generation_chunk = self._generation_chunk_from_delta(delta)
            if run_manager is not None and delta.text:
                await run_manager.on_llm_new_token(delta.text, chunk=generation_chunk)
            yield generation_chunk

    @property
    def _llm_type(self) -> str:
        return "codex-chat"

    @property
    def _identifying_params(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "launch_command": tuple(self.launch_command),
            "cwd": self.cwd,
            "approval_policy": self.approval_policy,
            "request_timeout": self.request_timeout,
            "turn_timeout": self.turn_timeout,
        }

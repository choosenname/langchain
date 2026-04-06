"""Codex chat models."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Protocol, cast

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

from langchain_codex._types import (
    AppServerProcess,
    JsonObject,
    TextInputItem,
    as_json_object,
    get_json_list,
    get_json_object,
    get_str,
)
from langchain_codex.errors import CodexError, CodexInputError
from langchain_codex.session import CodexSession, TurnDelta
from langchain_codex.transport import CodexAppServerTransport


class _CodexSessionLike(Protocol):
    def run_turn(self, input_items: list[TextInputItem]) -> JsonObject:
        """Run one app-server turn."""
        ...

    def stream_turn(self, input_items: list[TextInputItem]) -> Iterator[TurnDelta]:
        """Stream text deltas for one app-server turn."""
        ...

    def astream_turn(
        self,
        input_items: list[TextInputItem],
    ) -> AsyncIterator[TurnDelta]:
        """Asynchronously stream text deltas for one app-server turn."""
        ...


def _get_nested_string(payload: JsonObject, *path: str) -> str | None:
    value: object = payload
    for key in path:
        nested = as_json_object(value)
        if nested is None:
            return None
        value = nested.get(key)
    return value if isinstance(value, str) else None


def _extract_turn_text(turn: JsonObject) -> str:
    text_chunks: list[str] = []
    events = get_json_list(turn, "events")
    if events is None:
        return ""

    for raw_message in events:
        message = as_json_object(raw_message)
        if message is None:
            continue
        params = get_json_object(message, "params")
        if params is None:
            continue
        event = get_json_object(params, "event")
        if event is None:
            continue
        if get_str(event, "type") != "text":
            continue
        text = get_str(event, "text")
        if text is not None:
            text_chunks.append(text)

    return "".join(text_chunks)


def _response_metadata_from_delta(
    delta: TurnDelta,
    *,
    model_name: str,
) -> JsonObject:
    metadata: JsonObject = {
        "model_provider": "codex",
        "model": model_name,
    }
    thread_id = delta.thread_id
    if thread_id is not None:
        metadata["thread_id"] = thread_id
    turn = delta.turn
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
    codex_command: str | None = None
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
        command = self._app_server_command()
        executable = command[0]
        if shutil.which(executable) is None:
            msg = (
                "Unable to launch Codex app-server because "
                f"{executable!r} is not available on PATH."
            )
            raise CodexError(msg)

        process = subprocess.Popen(
            command,
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
            process=cast(AppServerProcess, process),
            request_timeout=self.request_timeout,
        )
        return CodexSession(
            transport=transport,
            model=self.model_name,
            turn_timeout=self.turn_timeout,
        )

    def _app_server_command(self) -> list[str]:
        if self.codex_command is None:
            return [self.codex_binary, "app-server"]

        try:
            command = shlex.split(self.codex_command)
        except ValueError as err:
            msg = f"Invalid codex_command: {err}"
            raise CodexError(msg) from err

        if not command:
            msg = "Invalid codex_command: command must not be empty."
            raise CodexError(msg)

        return [*command, "app-server"]

    def _to_input_items(self, messages: list[BaseMessage]) -> list[TextInputItem]:
        rendered_messages: list[str] = []
        for message in messages:
            content = getattr(message, "content")
            if not isinstance(content, str):
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

            rendered_messages.append(f"{role}: {content}")

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
                chunk_position=delta.chunk_position,
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
                chunk_position=delta.chunk_position,
            )
            generation_chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager is not None and delta.text:
                await run_manager.on_llm_new_token(delta.text, chunk=generation_chunk)
            yield generation_chunk

    @property
    def _llm_type(self) -> str:
        return "codex-chat"

    @property
    def _identifying_params(self) -> JsonObject:
        return {
            "model_name": self.model_name,
            "codex_binary": self.codex_binary,
            "codex_command": self.codex_command,
            "request_timeout": self.request_timeout,
            "turn_timeout": self.turn_timeout,
        }

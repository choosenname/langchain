"""Codex chat models."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Literal, Protocol, cast

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
    get_nested_json_object,
    get_nested_str,
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

def _iter_turn_messages(turn: JsonObject) -> Iterator[JsonObject]:
    events = get_json_list(turn, "events")
    if events is None:
        return

    for raw_message in events:
        message = as_json_object(raw_message)
        if message is not None:
            yield message


def _completed_agent_message_text(message: JsonObject) -> str | None:
    if message.get("method") != "item/completed":
        return None

    item = get_nested_json_object(message, "params", "item")
    if item is None or get_str(item, "type") != "agentMessage":
        return None
    return get_str(item, "text")


def _agent_message_delta_text(message: JsonObject) -> str | None:
    if message.get("method") == "item/agentMessage/delta":
        params = get_json_object(message, "params")
        if params is None:
            return None
        return get_str(params, "delta")

    item = get_nested_json_object(message, "params", "item")
    if item is None or get_str(item, "type") != "agentMessage":
        return None

    delta_blocks = get_json_list(item, "delta")
    if delta_blocks is None:
        return None

    text_parts: list[str] = []
    for raw_block in delta_blocks:
        block = as_json_object(raw_block)
        if block is None or get_str(block, "type") != "text":
            continue
        text = get_str(block, "text")
        if text is not None:
            text_parts.append(text)
    if not text_parts:
        return None
    return "".join(text_parts)


def _legacy_turn_output_text(message: JsonObject) -> str | None:
    event = get_nested_json_object(message, "params", "event")
    if event is None or get_str(event, "type") != "text":
        return None
    return get_str(event, "text")


def _extract_turn_text(turn: JsonObject) -> str:
    completed_agent_messages: list[str] = []
    item_deltas: list[str] = []
    text_chunks: list[str] = []
    for message in _iter_turn_messages(turn):
        completed_message = _completed_agent_message_text(message)
        if completed_message is not None:
            completed_agent_messages.append(completed_message)

        delta_text = _agent_message_delta_text(message)
        if delta_text is not None:
            item_deltas.append(delta_text)

        turn_output_text = _legacy_turn_output_text(message)
        if turn_output_text is not None:
            text_chunks.append(turn_output_text)

    if completed_agent_messages:
        return completed_agent_messages[-1]
    if item_deltas:
        return "".join(item_deltas)
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
        turn_id = get_nested_str({"turn": turn}, "turn", "id")
        if turn_id is not None:
            metadata["turn_id"] = turn_id
        turn_status = get_nested_str({"turn": turn}, "turn", "status")
        if turn_status is not None:
            metadata["turn_status"] = turn_status
    return metadata


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
    typed_message = cast("Any", message)
    content = cast("object", typed_message.content)
    if not isinstance(content, str):
        msg = "ChatCodex only supports string message content."
        raise CodexInputError(msg)
    return f"{_message_role_name(message)}: {content}"


class ChatCodex(BaseChatModel):
    """LangChain chat model backed by a local Codex app-server session."""

    model_name: str = Field(alias="model")
    codex_binary: str = "codex"
    codex_command: str | None = None
    request_timeout: float | None = None
    turn_timeout: float | None = None
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = (
        "on-request"
    )
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
        "workspace-write"
    )

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
        if shutil.which(command[0]) is None:
            msg = (
                "Unable to launch Codex app-server because "
                f"{command[0]!r} is not available on PATH."
            )
            raise CodexError(msg)

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            msg = "Codex app-server process did not expose stdio pipes."
            raise CodexError(msg)
        self._process = process
        transport = CodexAppServerTransport(
            process=cast("AppServerProcess", process),
            request_timeout=self.request_timeout,
        )
        return CodexSession(
            transport=transport,
            model=self.model_name,
            approval_policy=self.approval_policy,
            sandbox=self.sandbox,
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
        rendered_messages = [_render_message(message) for message in messages]
        return [{"type": "text", "text": "\n".join(rendered_messages)}]

    def _chat_result_metadata(self, turn: JsonObject) -> JsonObject:
        return {
            "model_provider": "codex",
            "model": self.model_name,
            "thread_id": get_nested_str(turn, "thread", "id"),
            "turn_id": get_nested_str(turn, "turn", "id"),
            "turn_status": get_nested_str(turn, "turn", "status"),
        }

    def _generation_chunk_from_delta(self, delta: TurnDelta) -> ChatGenerationChunk:
        message_chunk = AIMessageChunk(
            content=delta.text,
            response_metadata=_response_metadata_from_delta(
                delta,
                model_name=self.model_name,
            ),
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
            response_metadata=self._chat_result_metadata(turn),
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
    def _identifying_params(self) -> JsonObject:
        return {
            "model_name": self.model_name,
            "codex_binary": self.codex_binary,
            "codex_command": self.codex_command,
            "request_timeout": self.request_timeout,
            "turn_timeout": self.turn_timeout,
            "approval_policy": self.approval_policy,
            "sandbox": self.sandbox,
        }

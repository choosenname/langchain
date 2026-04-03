"""Codex chat models."""

from __future__ import annotations

import subprocess
from typing import Any, Callable, Protocol

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, get_buffer_string
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_codex.session import CodexSession
from langchain_codex.transport import CodexAppServerTransport


class _CodexSessionLike(Protocol):
    def run_turn(
        self, input_items: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Run one app-server turn."""


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


class ChatCodex(BaseChatModel):
    """LangChain chat model backed by a local Codex app-server session."""

    model_name: str = Field(alias="model")
    codex_binary: str = "codex"

    model_config = ConfigDict(populate_by_name=True)

    _process: subprocess.Popen[str] | None = PrivateAttr(default=None)
    _session_factory: Callable[[], _CodexSessionLike] | None = PrivateAttr(default=None)
    _session_instance: _CodexSessionLike | None = PrivateAttr(default=None)

    def _session(self) -> _CodexSessionLike:
        if self._session_instance is None:
            if self._session_factory is not None:
                self._session_instance = self._session_factory()
            else:
                self._session_instance = self._build_session()
        return self._session_instance

    def _build_session(self) -> CodexSession:
        process = subprocess.Popen(
            [self.codex_binary, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if process.stdin is None or process.stdout is None:
            msg = "Codex app-server process did not expose stdio pipes."
            raise RuntimeError(msg)
        self._process = process
        transport = CodexAppServerTransport(process=process)
        return CodexSession(transport=transport, model=self.model_name)

    def _to_input_items(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        return [{"type": "text", "text": get_buffer_string(messages)}]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = stop
        _ = run_manager
        turn = self._session().run_turn(self._to_input_items(messages), **kwargs)
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

    @property
    def _llm_type(self) -> str:
        return "codex-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "codex_binary": self.codex_binary,
        }

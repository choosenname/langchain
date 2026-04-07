"""Session lifecycle management for the Codex app server."""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from langchain_codex._types import (
    JsonObject,
    TextInputItem,
    as_json_object,
    get_json_list,
    get_json_object,
    get_nested_json_list,
    get_nested_json_object,
    get_str,
)
from langchain_codex.errors import CodexError


def _empty_notifications() -> list[JsonObject]:
    """Return an empty typed notification buffer."""
    return []


@dataclass(frozen=True)
class TurnDelta:
    """Incremental text emitted while a turn is running."""

    text: str
    thread_id: str | None = None
    turn: JsonObject | None = None
    chunk_position: Literal["last"] | None = None


@dataclass
class _ActiveTurn:
    """State required while consuming notifications for a single turn."""

    thread_id: str
    turn: JsonObject
    turn_id: str
    notification_condition: threading.Condition = field(default_factory=threading.Condition)
    buffered_notifications: list[JsonObject] = field(default_factory=_empty_notifications)
    remove_notification_handler: Callable[[], None] = field(default=lambda: None)
    latest_thread_status: JsonObject | None = None

    def collect_notification(self, message: JsonObject) -> None:
        """Buffer a notification and wake any waiting turn consumer."""
        with self.notification_condition:
            self._update_thread_status(message)
            self.buffered_notifications.append(message)
            self.notification_condition.notify_all()

    def _update_thread_status(self, message: JsonObject) -> None:
        if message.get("method") != "thread/status/changed":
            return

        params = get_json_object(message, "params")
        if params is None or get_str(params, "threadId") != self.thread_id:
            return

        status = get_json_object(params, "status")
        if status is not None:
            self.latest_thread_status = status


class _CodexTransport(Protocol):
    """Protocol for the transport methods used by `CodexSession`."""

    def request(self, method: str, params: JsonObject) -> JsonObject:
        """Send a request and return the JSON-RPC result payload."""
        ...

    def notify(self, method: str, params: JsonObject) -> None:
        """Send a notification."""
        ...

    def add_notification_handler(
        self,
        on_notification: Callable[[JsonObject], None],
    ) -> Callable[[], None]:
        """Install an additional notification handler and return a remover."""
        ...

    def diagnostics(self) -> str | None:
        """Return recent transport diagnostics when available."""
        ...


class CodexSession:
    """Own a single app-server connection and a persistent thread."""

    _CLIENT_INFO: dict[str, str] = {
        "name": "langchain_codex",
        "title": "LangChain Codex",
        "version": "0.1.0",
    }

    def __init__(
        self,
        *,
        transport: _CodexTransport,
        model: str,
        approval_policy: str | None = "on-request",
        sandbox: str | None = "workspace-write",
        turn_timeout: float | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            transport: App-server transport used for JSON-RPC communication.
            model: Model name used when creating the persistent thread.
            approval_policy: Approval policy passed to `thread/start` when creating
                the Codex thread.
            sandbox: Sandbox mode passed to `thread/start` when creating the Codex
                thread.
            turn_timeout: Maximum seconds to wait for turn completion events.
        """
        self._transport = transport
        self.model = model
        self._approval_policy = approval_policy
        self._sandbox = sandbox
        self._turn_timeout = turn_timeout
        self._started = False
        self._thread_id: str | None = None

    def ensure_started(self) -> None:
        """Perform the app-server handshake once."""
        if self._started:
            return

        self._transport.request("initialize", {"clientInfo": self._CLIENT_INFO})
        self._transport.notify("initialized", {})
        self._started = True

    def run_turn(self, input_items: list[TextInputItem]) -> JsonObject:
        """Run a turn and collect turn-scoped events.

        Args:
            input_items: Turn input payload to send to the app server.

        Returns:
            A structured turn result containing the thread, turn, and raw events.
        """
        active_turn = self._start_turn(input_items)
        events = list(self._iter_turn_notifications(active_turn))

        return {
            "thread": {"id": active_turn.thread_id},
            "turn": active_turn.turn,
            "events": events,
        }

    def stream_turn(self, input_items: list[TextInputItem]) -> Iterator[TurnDelta]:
        """Yield text deltas for a turn until the server confirms completion.

        Args:
            input_items: Turn input payload to send to the app server.

        Yields:
            Text deltas emitted from `item/agentMessage/delta` notifications.
        """
        active_turn = self._start_turn(input_items)
        for message in self._iter_turn_notifications(active_turn):
            text = self._extract_text_delta(message)
            if text is not None:
                yield TurnDelta(text=text)
        yield TurnDelta(
            text="",
            thread_id=active_turn.thread_id,
            turn=active_turn.turn,
            chunk_position="last",
        )

    async def astream_turn(
        self,
        input_items: list[TextInputItem],
    ) -> AsyncIterator[TurnDelta]:
        """Asynchronously yield text deltas for a turn."""
        loop = asyncio.get_running_loop()
        result_queue: queue.Queue[TurnDelta | Exception | None] = queue.Queue()
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)

        def submit_result(item: TurnDelta | Exception | None) -> None:
            result_queue.put(item)
            with contextlib.suppress(OSError):
                os.write(write_fd, b"\0")

        def stream_in_background() -> None:
            try:
                for delta in self.stream_turn(input_items):
                    submit_result(delta)
            except Exception as exc:
                submit_result(exc)
            else:
                submit_result(None)

        worker = threading.Thread(target=stream_in_background, daemon=True)
        worker.start()

        try:
            while True:
                try:
                    item = result_queue.get_nowait()
                except queue.Empty:
                    wakeup = loop.create_future()

                    def on_ready(
                        ready_future: asyncio.Future[None] = wakeup,
                    ) -> None:
                        if not ready_future.done():
                            ready_future.set_result(None)

                    loop.add_reader(read_fd, on_ready)
                    try:
                        await wakeup
                    finally:
                        loop.remove_reader(read_fd)
                    with contextlib.suppress(OSError):
                        os.read(read_fd, 4096)
                    continue
                while True:
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
                    try:
                        item = result_queue.get_nowait()
                    except queue.Empty:
                        break
        finally:
            with contextlib.suppress(OSError):
                os.close(read_fd)
            with contextlib.suppress(OSError):
                os.close(write_fd)

    def _ensure_thread(self) -> str:
        """Create the app-server thread lazily and reuse it thereafter."""
        if self._thread_id is None:
            result = self._transport.request("thread/start", self._thread_start_params())
            thread = get_json_object(result, "thread")
            if thread is None:
                msg = "thread/start response missing thread id"
                raise CodexError(msg)
            thread_id = get_str(thread, "id")
            if thread_id is None or not thread_id:
                msg = "thread/start response missing thread id"
                raise CodexError(msg)
            self._thread_id = thread_id
        return self._thread_id

    def _start_turn(self, input_items: list[TextInputItem]) -> _ActiveTurn:
        """Start a turn and install a scoped notification buffer."""
        self.ensure_started()
        thread_id = self._ensure_thread()
        active_turn = _ActiveTurn(
            thread_id=thread_id,
            turn={},
            turn_id="",
        )
        active_turn.remove_notification_handler = self._transport.add_notification_handler(
            active_turn.collect_notification
        )
        try:
            result = self._transport.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": input_items,
                },
            )
            turn = self._require_turn(result, error_message="turn/start response missing turn id")
        except Exception:
            active_turn.remove_notification_handler()
            raise
        active_turn.turn = turn
        turn_id = get_str(turn, "id")
        if turn_id is None:
            msg = "turn/start response missing turn id"
            raise CodexError(msg)
        active_turn.turn_id = turn_id
        return active_turn

    def _iter_turn_notifications(
        self,
        active_turn: _ActiveTurn,
    ) -> Iterator[JsonObject]:
        """Yield notifications for the active turn until completion."""
        turn_completed = False
        deadline = (
            None
            if self._turn_timeout is None
            else time.monotonic() + self._turn_timeout
        )
        try:
            while not turn_completed:
                pending_notifications = self._wait_for_turn_notifications(
                    active_turn,
                    deadline=deadline,
                )

                for message in pending_notifications:
                    if not self._is_turn_notification_for(message, active_turn.turn_id):
                        continue

                    turn_completed = self._handle_turn_notification(
                        active_turn,
                        message,
                        turn_completed=turn_completed,
                    )
                    yield message
        finally:
            active_turn.remove_notification_handler()

    def _wait_for_turn_notifications(
        self,
        active_turn: _ActiveTurn,
        *,
        deadline: float | None,
    ) -> list[JsonObject]:
        with active_turn.notification_condition:
            while not active_turn.buffered_notifications:
                self._wait_for_next_notification(active_turn, deadline=deadline)
            pending_notifications = active_turn.buffered_notifications[:]
            active_turn.buffered_notifications.clear()
        return pending_notifications

    def _wait_for_next_notification(
        self,
        active_turn: _ActiveTurn,
        *,
        deadline: float | None,
    ) -> None:
        remaining = self._remaining_turn_time(deadline)
        if remaining == 0.0:
            msg = self._turn_timeout_message(active_turn)
            raise CodexError(msg)

        active_turn.notification_condition.wait(timeout=remaining)
        if active_turn.buffered_notifications:
            return
        updated_remaining = self._remaining_turn_time(deadline)
        if updated_remaining is not None and updated_remaining <= 0:
            msg = self._turn_timeout_message(active_turn)
            raise CodexError(msg)

    @staticmethod
    def _remaining_turn_time(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _handle_turn_notification(
        self,
        active_turn: _ActiveTurn,
        message: JsonObject,
        *,
        turn_completed: bool,
    ) -> bool:
        if message.get("method") != "turn/completed":
            return turn_completed

        completed_turn = self._extract_turn_from_notification(message)
        if completed_turn is not None:
            active_turn.turn = completed_turn
        return True

    def _turn_timeout_message(self, active_turn: _ActiveTurn) -> str:
        msg = (
            "Timed out waiting for Codex app-server to complete turn "
            f"{active_turn.turn_id!r}."
        )
        thread_status = self._thread_status_summary(active_turn.latest_thread_status)
        if thread_status is not None:
            msg += f" Last thread status: {thread_status}."
        diagnostics = self._transport.diagnostics()
        if diagnostics is None:
            return msg
        return f"{msg}\n\n{diagnostics}"

    @staticmethod
    def _thread_status_summary(status: JsonObject | None) -> str | None:
        if status is None:
            return None

        status_type = get_str(status, "type")
        if status_type != "active":
            return status_type

        active_flags = get_json_list(status, "activeFlags")
        if not active_flags:
            return status_type
        rendered_flags = [flag for flag in active_flags if isinstance(flag, str)]
        if not rendered_flags:
            return status_type
        return f"{status_type} ({', '.join(rendered_flags)})"

    @staticmethod
    def _is_turn_notification_for(message: JsonObject, turn_id: str) -> bool:
        """Return `True` when a notification belongs to the active turn."""
        notification_turn_id = CodexSession._notification_turn_id(message)
        return notification_turn_id == turn_id

    @staticmethod
    def _notification_turn_id(message: JsonObject) -> str | None:
        params = get_json_object(message, "params")
        if params is None:
            return None

        direct_turn_id = get_str(params, "turnId")
        if direct_turn_id is not None:
            return direct_turn_id

        turn = get_nested_json_object(params, "turn")
        if turn is None:
            return None
        return get_str(turn, "id")

    @staticmethod
    def _require_turn(result: JsonObject, *, error_message: str) -> JsonObject:
        """Return the active turn payload or raise if the response is malformed."""
        turn = get_json_object(result, "turn")
        if turn is None:
            raise CodexError(error_message)

        turn_id = get_str(turn, "id")
        if turn_id is None or not turn_id:
            raise CodexError(error_message)

        return turn

    @staticmethod
    def _extract_turn_from_notification(
        message: JsonObject,
    ) -> JsonObject | None:
        """Return turn metadata from a notification when present."""
        turn = get_nested_json_object(message, "params", "turn")
        if turn is None:
            return None

        turn_id = get_str(turn, "id")
        if turn_id is None or not turn_id:
            return None

        return turn

    @staticmethod
    def _extract_text_delta(message: JsonObject) -> str | None:
        """Return streamed text from an `agentMessage` delta notification."""
        if message.get("method") == "item/agentMessage/delta":
            params = get_json_object(message, "params")
            if params is None:
                return None
            return get_str(params, "delta")

        item = get_nested_json_object(message, "params", "item")
        if item is None:
            return None
        if get_str(item, "type") != "agentMessage":
            return None

        return CodexSession._text_from_delta_blocks(get_nested_json_list(item, "delta"))

    @staticmethod
    def _text_from_delta_blocks(delta_blocks: list[object] | None) -> str | None:
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

    def _thread_start_params(self) -> JsonObject:
        params: JsonObject = {"model": self.model}
        if self._approval_policy is not None:
            params["approvalPolicy"] = self._approval_policy
        if self._sandbox is not None:
            params["sandbox"] = self._sandbox
        return params

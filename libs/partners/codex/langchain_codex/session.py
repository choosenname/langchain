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
from typing import Any, Literal, Protocol

from langchain_codex.errors import CodexError


@dataclass(frozen=True)
class TurnDelta:
    """Incremental text emitted while a turn is running."""

    text: str
    thread_id: str | None = None
    turn: dict[str, Any] | None = None
    chunk_position: Literal["last"] | None = None


@dataclass
class _ActiveTurn:
    """State required while consuming notifications for a single turn."""

    thread_id: str
    turn: dict[str, Any]
    turn_id: str
    notification_condition: threading.Condition = field(default_factory=threading.Condition)
    buffered_notifications: list[dict[str, Any]] = field(default_factory=list)
    remove_notification_handler: Callable[[], None] = field(default=lambda: None)

    def collect_notification(self, message: dict[str, Any]) -> None:
        """Buffer a notification and wake any waiting turn consumer."""
        with self.notification_condition:
            self.buffered_notifications.append(message)
            self.notification_condition.notify_all()


class _CodexTransport(Protocol):
    """Protocol for the transport methods used by `CodexSession`."""

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return the JSON-RPC result payload."""

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification."""

    def add_notification_handler(
        self,
        on_notification: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        """Install an additional notification handler and return a remover."""


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
        turn_timeout: float | None = 60.0,
    ) -> None:
        """Initialize the session.

        Args:
            transport: App-server transport used for JSON-RPC communication.
            model: Model name used when creating the persistent thread.
            turn_timeout: Maximum seconds to wait for turn completion events.
        """
        self._transport = transport
        self.model = model
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

    def run_turn(self, input_items: list[dict[str, Any]]) -> dict[str, Any]:
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

    def stream_turn(self, input_items: list[dict[str, Any]]) -> Iterator[TurnDelta]:
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
        input_items: list[dict[str, Any]],
    ) -> AsyncIterator[TurnDelta]:
        """Asynchronously yield text deltas for a turn."""
        done = object()
        loop = asyncio.get_running_loop()
        result_queue: queue.Queue[TurnDelta | Exception | object] = queue.Queue()
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)

        def submit_result(item: TurnDelta | Exception | object) -> None:
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
                submit_result(done)

        worker = threading.Thread(target=stream_in_background, daemon=True)
        worker.start()

        try:
            while True:
                try:
                    item = result_queue.get_nowait()
                except queue.Empty:
                    wakeup = loop.create_future()

                    def on_ready() -> None:
                        if not wakeup.done():
                            wakeup.set_result(None)

                    loop.add_reader(read_fd, on_ready)
                    try:
                        await wakeup
                    finally:
                        loop.remove_reader(read_fd)
                    with contextlib.suppress(OSError):
                        os.read(read_fd, 4096)
                    continue
                while True:
                    if item is done:
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
            result = self._transport.request("thread/start", {"model": self.model})
            self._thread_id = result["thread"]["id"]
        return self._thread_id

    def _start_turn(self, input_items: list[dict[str, Any]]) -> _ActiveTurn:
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
            turn = self._extract_turn(result)
        except Exception:
            active_turn.remove_notification_handler()
            raise
        active_turn.turn = turn
        active_turn.turn_id = turn["id"]
        return active_turn

    def _iter_turn_notifications(
        self,
        active_turn: _ActiveTurn,
    ) -> Iterator[dict[str, Any]]:
        """Yield notifications for the active turn until completion."""
        turn_completed = False
        deadline = (
            None
            if self._turn_timeout is None
            else time.monotonic() + self._turn_timeout
        )
        try:
            while not turn_completed:
                with active_turn.notification_condition:
                    while not active_turn.buffered_notifications and not turn_completed:
                        remaining = (
                            None
                            if deadline is None
                            else max(0.0, deadline - time.monotonic())
                        )
                        if remaining == 0.0:
                            msg = (
                                "Timed out waiting for Codex app-server to complete "
                                f"turn {active_turn.turn_id!r}."
                            )
                            raise CodexError(msg)
                        active_turn.notification_condition.wait(timeout=remaining)
                        if not active_turn.buffered_notifications and deadline is not None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                msg = (
                                    "Timed out waiting for Codex app-server to complete "
                                    f"turn {active_turn.turn_id!r}."
                                )
                                raise CodexError(msg)
                    pending_notifications = active_turn.buffered_notifications[:]
                    active_turn.buffered_notifications.clear()

                for message in pending_notifications:
                    if not self._is_turn_notification_for(message, active_turn.turn_id):
                        continue

                    if message.get("method") == "turn/completed":
                        completed_turn = self._extract_turn_from_notification(message)
                        if completed_turn is not None:
                            active_turn.turn = completed_turn
                        turn_completed = True
                    yield message
        finally:
            active_turn.remove_notification_handler()

    @staticmethod
    def _is_turn_notification_for(message: dict[str, Any], turn_id: str) -> bool:
        """Return `True` when a notification belongs to the active turn."""
        params = message.get("params")
        if not isinstance(params, dict):
            return False

        turn = params.get("turn")
        if not isinstance(turn, dict):
            return False

        return turn.get("id") == turn_id

    @staticmethod
    def _extract_turn(result: dict[str, Any]) -> dict[str, Any]:
        """Return the active turn payload or raise if the response is malformed."""
        turn = result.get("turn")
        if not isinstance(turn, dict):
            msg = "turn/start response missing turn id"
            raise CodexError(msg)

        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            msg = "turn/start response missing turn id"
            raise CodexError(msg)

        return turn

    @staticmethod
    def _extract_turn_from_notification(
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return turn metadata from a notification when present."""
        params = message.get("params")
        if not isinstance(params, dict):
            return None

        turn = params.get("turn")
        if not isinstance(turn, dict):
            return None

        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            return None

        return turn

    @staticmethod
    def _extract_text_delta(message: dict[str, Any]) -> str | None:
        """Return streamed text from an `agentMessage` delta notification."""
        params = message.get("params")
        if not isinstance(params, dict):
            return None

        item = params.get("item")
        if not isinstance(item, dict) or item.get("type") != "agentMessage":
            return None

        delta = item.get("delta")
        if not isinstance(delta, list):
            return None

        text_parts = [
            text
            for block in delta
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(text := block.get("text"), str)
        ]
        if not text_parts:
            return None
        return "".join(text_parts)

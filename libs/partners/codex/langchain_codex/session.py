"""Provider-native session workflows for Codex app-server."""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from langchain_codex.errors import CodexError
from langchain_codex.protocol.events import parse_notification
from langchain_codex.protocol.requests import (
    build_thread_start_params,
    build_turn_start_params,
)
from langchain_codex.types import (
    CodexApprovalDecision,
    CodexApprovalRequest,
    CodexClientConfig,
    CodexEvent,
    CodexThreadHandle,
    CodexTurnHandle,
    CodexTurnResult,
    JsonObject,
)

if TYPE_CHECKING:
    from langchain_codex.transport.base import CodexTransport

NotificationHandler = Callable[[dict[str, object]], None]
ServerRequestHandler = Callable[[object], object | None]


@dataclass(frozen=True)
class TurnDelta:
    """Incremental text emitted while a turn is running."""

    text: str
    thread_id: str | None = None
    turn: JsonObject | None = None
    chunk_position: Literal["last"] | None = None


def _empty_notifications() -> list[dict[str, object]]:
    return []


@dataclass
class _ActiveTurn:
    thread_id: str
    turn_id: str
    turn: JsonObject
    notification_condition: threading.Condition = field(default_factory=threading.Condition)
    buffered_notifications: list[dict[str, object]] = field(default_factory=_empty_notifications)
    remove_notification_handler: Callable[[], None] = field(default=lambda: None)

    def collect(self, notification: dict[str, object]) -> None:
        with self.notification_condition:
            self.buffered_notifications.append(notification)
            self.notification_condition.notify_all()


class CodexSession:
    """Manage thread, turn, filesystem, and realtime workflows on one connection."""

    _CLIENT_INFO: dict[str, str] = {
        "name": "langchain_codex",
        "title": "LangChain Codex",
        "version": "0.1.0",
    }

    def __init__(
        self,
        *,
        transport: CodexTransport,
        config: CodexClientConfig,
        approval_handler: Callable[[CodexApprovalRequest], CodexApprovalDecision] | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            transport: JSON-RPC transport implementation.
            config: Shared client/session configuration.
            approval_handler: Optional blocking handler for server requests.
            thread_id: Initial thread id to bind the session to.
        """
        self._transport = transport
        self.config = config
        self._approval_handler = approval_handler
        self._started = False
        self._current_thread_id = thread_id
        self._known_threads: dict[str, CodexThreadHandle] = {}
        self._turn_timeout = config.turn_timeout
        self._server_request_remove = self._add_server_request_handler()

    def ensure_started(self) -> None:
        """Perform the app-server handshake once."""
        if self._started:
            return
        self._request(
            "initialize",
            {"clientInfo": dict(self._CLIENT_INFO)},
        )
        self._notify("initialized", {})
        self._started = True

    def start_thread(self) -> CodexThreadHandle:
        """Create and select a new thread."""
        self.ensure_started()
        response = self._request("thread/start", self._thread_start_params())
        thread = self._thread_from_response(response)
        self._remember_thread(thread)
        return thread

    def resume_thread(self, thread_id: str) -> CodexThreadHandle:
        """Resume an existing thread and select it."""
        self.ensure_started()
        response = self._request("thread/resume", {"threadId": thread_id})
        thread = self._thread_from_response(response)
        self._remember_thread(thread)
        return thread

    def fork_thread(self, thread_id: str, *, ephemeral: bool = False) -> CodexThreadHandle:
        """Fork an existing thread."""
        self.ensure_started()
        response = self._request(
            "thread/fork",
            {"threadId": thread_id, "ephemeral": ephemeral},
        )
        thread = self._thread_from_response(response)
        self._remember_thread(thread)
        return thread

    def list_threads(self) -> list[CodexThreadHandle]:
        """List persisted threads."""
        self.ensure_started()
        response = self._request("thread/list", {})
        data = response.get("data")
        if not isinstance(data, list):
            return []
        items = cast("list[object]", data)
        threads: list[CodexThreadHandle] = []
        for item in items:
            payload = self._as_json_object(item)
            if payload is None:
                continue
            threads.append(self._thread_from_payload(payload))
        return threads

    def list_loaded_threads(self) -> list[str]:
        """List loaded thread ids."""
        self.ensure_started()
        response = self._request("thread/loaded/list", {})
        data = response.get("data")
        if not isinstance(data, list):
            return []
        items = cast("list[object]", data)
        return [item for item in items if isinstance(item, str)]

    def read_thread(self, thread_id: str) -> CodexThreadHandle:
        """Read a thread without resuming it."""
        self.ensure_started()
        response = self._request("thread/read", {"threadId": thread_id})
        thread = self._thread_from_response(response)
        self._known_threads.setdefault(thread.thread_id, thread)
        return thread

    def rollback_thread(self, thread_id: str, *, turns: int) -> CodexThreadHandle:
        """Rollback a thread's context."""
        self.ensure_started()
        response = self._request(
            "thread/rollback",
            {"threadId": thread_id, "turns": turns},
        )
        thread = self._thread_from_response(response)
        self._remember_thread(thread)
        return thread

    def list_known_threads(self) -> list[CodexThreadHandle]:
        """Return in-memory known threads."""
        return list(self._known_threads.values())

    def steer_turn(
        self,
        thread_id: str,
        input_items: Sequence[JsonObject],
        *,
        expected_turn_id: str,
    ) -> str:
        """Append input to the active turn."""
        self.ensure_started()
        response = self._request(
            "turn/steer",
            {
                "threadId": thread_id,
                "input": list(input_items),
                "expectedTurnId": expected_turn_id,
            },
        )
        turn_id = response.get("turnId")
        if not isinstance(turn_id, str):
            msg = "turn/steer response missing turn id"
            raise CodexError(msg)
        return turn_id

    def interrupt_turn(self, thread_id: str, turn_id: str) -> None:
        """Interrupt an active turn."""
        self.ensure_started()
        self._request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})

    def start_review(
        self,
        thread_id: str,
        *,
        target: JsonObject,
        delivery: str | None = None,
    ) -> CodexTurnResult:
        """Start a review turn."""
        self.ensure_started()
        params: JsonObject = {"threadId": thread_id, "target": target}
        if delivery is not None:
            params["delivery"] = delivery
        response = self._request("review/start", params)
        thread = self._known_threads.get(thread_id) or CodexThreadHandle(thread_id=thread_id)
        turn = self._turn_from_response(response)
        return CodexTurnResult(thread=thread, turn=turn, events=())

    def run_turn(
        self,
        input_items: Sequence[JsonObject],
        *,
        thread_id: str | None = None,
    ) -> CodexTurnResult:
        """Run a turn and collect notifications until completion."""
        active_turn = self._start_turn(input_items, thread_id=thread_id)
        notifications = self._iter_turn_notifications(active_turn)
        events = tuple(parse_notification(message) for message in notifications)
        output_text = self._output_text_from_events(events)
        thread = self._known_threads.get(active_turn.thread_id)
        if thread is None:
            thread = CodexThreadHandle(thread_id=active_turn.thread_id)
        turn_status = self._turn_status_from_notifications(events)
        if turn_status is None:
            turn_status = self._get_str(active_turn.turn, "status")
        turn = CodexTurnHandle(
            turn_id=active_turn.turn_id,
            status=turn_status,
            raw=dict(active_turn.turn),
        )
        return CodexTurnResult(thread=thread, turn=turn, events=events, output_text=output_text)

    def stream_turn(
        self,
        input_items: Sequence[JsonObject],
        *,
        thread_id: str | None = None,
    ) -> Iterator[TurnDelta]:
        """Stream text deltas for a turn."""
        active_turn = self._start_turn(input_items, thread_id=thread_id)
        for message in self._iter_turn_notifications(active_turn):
            text = self._text_delta_from_notification(message)
            if text is not None:
                yield TurnDelta(text=text)
        yield TurnDelta(
            text="",
            thread_id=active_turn.thread_id,
            turn=dict(active_turn.turn),
            chunk_position="last",
        )

    async def astream_turn(
        self,
        input_items: Sequence[JsonObject],
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[TurnDelta]:
        """Asynchronously stream text deltas for a turn."""
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
                for delta in self.stream_turn(input_items, thread_id=thread_id):
                    submit_result(delta)
            except Exception as exc:  # pragma: no cover - surfaced to caller
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

                    def on_ready(ready_future: asyncio.Future[None] = wakeup) -> None:
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

    def read_file(self, path: str) -> str:
        """Read a file via `fs/readFile`."""
        self.ensure_started()
        response = self._request("fs/readFile", {"path": path})
        data = response.get("dataBase64")
        if not isinstance(data, str):
            msg = "fs/readFile response missing dataBase64"
            raise CodexError(msg)
        return data

    def write_file(self, path: str, *, data_base64: str) -> None:
        """Write a file via `fs/writeFile`."""
        self.ensure_started()
        self._request("fs/writeFile", {"path": path, "dataBase64": data_base64})

    def create_directory(self, path: str, *, recursive: bool = True) -> None:
        """Create a directory via `fs/createDirectory`."""
        self.ensure_started()
        self._request("fs/createDirectory", {"path": path, "recursive": recursive})

    def get_metadata(self, path: str) -> JsonObject:
        """Fetch filesystem metadata."""
        self.ensure_started()
        response = self._request("fs/getMetadata", {"path": path})
        return dict(response)

    def read_directory(self, path: str) -> list[JsonObject]:
        """Read a directory listing."""
        self.ensure_started()
        response = self._request("fs/readDirectory", {"path": path})
        entries = response.get("entries")
        if isinstance(entries, list):
            raw_entries = cast("list[object]", entries)
            return [cast("JsonObject", entry) for entry in raw_entries if isinstance(entry, dict)]
        data = response.get("data")
        if isinstance(data, list):
            raw_entries = cast("list[object]", data)
            return [cast("JsonObject", entry) for entry in raw_entries if isinstance(entry, dict)]
        return []

    def remove_path(self, path: str, *, recursive: bool = True, force: bool = True) -> None:
        """Remove a filesystem path."""
        self.ensure_started()
        self._request(
            "fs/remove",
            {"path": path, "recursive": recursive, "force": force},
        )

    def copy_path(
        self,
        source_path: str,
        destination_path: str,
        *,
        recursive: bool | None = None,
    ) -> None:
        """Copy a filesystem path."""
        self.ensure_started()
        params: JsonObject = {
            "sourcePath": source_path,
            "destinationPath": destination_path,
        }
        if recursive is not None:
            params["recursive"] = recursive
        self._request("fs/copy", params)

    def watch_path(self, watch_id: str, path: str) -> str:
        """Watch a filesystem path."""
        self.ensure_started()
        response = self._request("fs/watch", {"watchId": watch_id, "path": path})
        watched_path = response.get("path")
        if not isinstance(watched_path, str):
            msg = "fs/watch response missing path"
            raise CodexError(msg)
        return watched_path

    def unwatch_path(self, watch_id: str) -> None:
        """Stop watching a filesystem path."""
        self.ensure_started()
        self._request("fs/unwatch", {"watchId": watch_id})

    def start_realtime(self, thread_id: str, *, prompt: str | None = None) -> None:
        """Start realtime for a thread."""
        self.ensure_started()
        params: JsonObject = {"threadId": thread_id}
        if prompt is not None:
            params["prompt"] = prompt
        self._request("thread/realtime/start", params)

    def append_realtime_audio(self, thread_id: str, *, audio: JsonObject) -> None:
        """Append audio to a realtime session."""
        self.ensure_started()
        self._request("thread/realtime/appendAudio", {"threadId": thread_id, "audio": audio})

    def append_realtime_text(self, thread_id: str, *, text: str) -> None:
        """Append text to a realtime session."""
        self.ensure_started()
        self._request("thread/realtime/appendText", {"threadId": thread_id, "text": text})

    def stop_realtime(self, thread_id: str) -> None:
        """Stop a realtime session."""
        self.ensure_started()
        self._request("thread/realtime/stop", {"threadId": thread_id})

    def _start_turn(
        self,
        input_items: Sequence[JsonObject],
        *,
        thread_id: str | None,
    ) -> _ActiveTurn:
        self.ensure_started()
        selected_thread_id = thread_id or self._current_thread_id
        if selected_thread_id is None:
            selected_thread_id = self.start_thread().thread_id
        active_turn = _ActiveTurn(thread_id=selected_thread_id, turn_id="", turn={})
        active_turn.remove_notification_handler = self._add_notification_handler(
            active_turn.collect
        )
        try:
            response = self._request(
                "turn/start",
                build_turn_start_params(
                    thread_id=selected_thread_id,
                    input_items=list(input_items),
                    cwd=self.config.cwd,
                    approval_policy=self.config.approval_policy,
                    sandbox_policy=self.config.sandbox_policy,
                    model=self.config.model,
                    effort=self.config.reasoning_effort,
                    summary=self.config.reasoning_summary,
                ),
            )
        except Exception:
            active_turn.remove_notification_handler()
            raise
        turn = self._turn_payload_from_response(response)
        active_turn.turn = turn
        turn_id = self._get_str(turn, "id")
        if turn_id is None:
            msg = "turn/start response missing turn id"
            raise CodexError(msg)
        active_turn.turn_id = turn_id
        return active_turn

    def _iter_turn_notifications(self, active_turn: _ActiveTurn) -> Iterator[dict[str, object]]:
        deadline = None if self._turn_timeout is None else time.monotonic() + self._turn_timeout
        try:
            while True:
                pending = self._wait_for_notifications(active_turn, deadline=deadline)
                for message in pending:
                    if not self._notification_matches_turn(message, active_turn.turn_id):
                        continue
                    if self._is_turn_completed(message):
                        completed_turn = self._turn_from_notification(message)
                        if completed_turn is not None:
                            active_turn.turn = completed_turn
                        yield message
                        return
                    yield message
        finally:
            active_turn.remove_notification_handler()

    def _wait_for_notifications(
        self,
        active_turn: _ActiveTurn,
        *,
        deadline: float | None,
    ) -> list[dict[str, object]]:
        with active_turn.notification_condition:
            while not active_turn.buffered_notifications:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    msg = f"Timed out waiting for Codex turn {active_turn.turn_id!r}."
                    raise CodexError(msg)
                active_turn.notification_condition.wait(timeout=remaining)
            buffered = list(active_turn.buffered_notifications)
            active_turn.buffered_notifications.clear()
            return buffered

    def _notification_matches_turn(
        self,
        notification: dict[str, object],
        turn_id: str,
    ) -> bool:
        params = notification.get("params")
        params_object = self._as_json_object(params)
        if params_object is None:
            return False
        direct_turn_id = params_object.get("turnId")
        if isinstance(direct_turn_id, str):
            return direct_turn_id == turn_id
        turn = self._as_json_object(params_object.get("turn"))
        if turn is not None:
            nested_turn_id = turn.get("id")
            return isinstance(nested_turn_id, str) and nested_turn_id == turn_id
        return False

    @staticmethod
    def _is_turn_completed(notification: dict[str, object]) -> bool:
        return notification.get("method") == "turn/completed"

    @staticmethod
    def _turn_from_notification(notification: dict[str, object]) -> JsonObject | None:
        params = CodexSession._as_json_object(notification.get("params"))
        if params is None:
            return None
        turn = CodexSession._as_json_object(params.get("turn"))
        if turn is None:
            return None
        return dict(turn)

    @staticmethod
    def _text_delta_from_notification(notification: dict[str, object]) -> str | None:
        if notification.get("method") != "item/agentMessage/delta":
            return None
        params = CodexSession._as_json_object(notification.get("params"))
        if params is None:
            return None
        delta = params.get("delta")
        return delta if isinstance(delta, str) else None

    @staticmethod
    def _output_text_from_events(events: Sequence[CodexEvent]) -> str:
        final_messages = [
            event.item.raw.get("text")
            for event in events
            if event.item is not None and event.item.kind == "agentMessage"
        ]
        final_text = [text for text in final_messages if isinstance(text, str)]
        if final_text:
            return final_text[-1]
        delta_text = [
            event.params.get("delta")
            for event in events
            if event.method == "item/agentMessage/delta"
        ]
        return "".join(text for text in delta_text if isinstance(text, str))

    @staticmethod
    def _turn_status_from_notifications(events: Sequence[CodexEvent]) -> str | None:
        for event in reversed(events):
            if event.method != "turn/completed":
                continue
            turn = CodexSession._as_json_object(event.params.get("turn"))
            if turn is None:
                continue
            status = turn.get("status")
            if isinstance(status, str):
                return status
        return None

    def _thread_start_params(self) -> JsonObject:
        return build_thread_start_params(
            model=self.config.model,
            cwd=self.config.cwd,
            approval_policy=self.config.approval_policy,
            sandbox_policy=self.config.sandbox_policy,
            personality=self.config.personality,
            service_name=self.config.service_name,
            mcp_servers=self.config.mcp_servers,
            include_default_mcp_config=self.config.include_default_mcp_config,
            experimental_api=self.config.experimental_api or None,
        )

    def _thread_from_response(self, response: JsonObject) -> CodexThreadHandle:
        thread_payload = self._as_json_object(response.get("thread"))
        if thread_payload is None:
            msg = "response missing thread payload"
            raise CodexError(msg)
        return self._thread_from_payload(thread_payload)

    def _thread_from_payload(self, payload: JsonObject) -> CodexThreadHandle:
        thread_id = self._get_str(payload, "id")
        if thread_id is None:
            msg = "thread payload missing id"
            raise CodexError(msg)
        status = payload.get("status")
        ephemeral = payload.get("ephemeral")
        return CodexThreadHandle(
            thread_id=thread_id,
            name=self._get_str(payload, "name"),
            status=cast("JsonObject", status) if isinstance(status, dict) else None,
            ephemeral=ephemeral if isinstance(ephemeral, bool) else None,
            preview=self._get_str(payload, "preview"),
            raw=dict(payload),
        )

    def _turn_from_response(self, response: JsonObject) -> CodexTurnHandle:
        turn_payload = self._turn_payload_from_response(response)
        turn_id = self._get_str(turn_payload, "id")
        if turn_id is None:
            msg = "response missing turn id"
            raise CodexError(msg)
        return CodexTurnHandle(
            turn_id=turn_id,
            status=self._get_str(turn_payload, "status"),
            raw=dict(turn_payload),
        )

    @staticmethod
    def _turn_payload_from_response(response: JsonObject) -> JsonObject:
        turn_payload = response.get("turn")
        if not isinstance(turn_payload, dict):
            msg = "response missing turn payload"
            raise CodexError(msg)
        return dict(cast("JsonObject", turn_payload))

    def _remember_thread(self, thread: CodexThreadHandle) -> None:
        self._known_threads[thread.thread_id] = thread
        self._current_thread_id = thread.thread_id

    def _handle_server_request(self, raw_request: object) -> object | None:
        request = self._as_json_object(raw_request)
        if request is None:
            return None
        request_id = request.get("id")
        method = request.get("method")
        params = self._as_json_object(request.get("params"))
        if not isinstance(request_id, int) or not isinstance(method, str):
            return None
        if params is None:
            return None
        approval_request = CodexApprovalRequest(
            request_id=request_id,
            method=method,
            params=dict(params),
            thread_id=self._get_str(params, "threadId"),
            turn_id=self._get_str(params, "turnId"),
        )
        if self._approval_handler is None:
            decision = CodexApprovalDecision(decision="cancel")
        else:
            decision = self._approval_handler(approval_request)
        return {"decision": decision.decision}

    def _request(self, method: str, params: JsonObject) -> JsonObject:
        response = self._transport.request(method, params)
        return dict(response)

    def _notify(self, method: str, params: JsonObject) -> None:
        self._transport.notify(method, params)

    def _add_notification_handler(self, handler: NotificationHandler) -> Callable[[], None]:
        return self._transport.add_notification_handler(handler)

    def _add_server_request_handler(self) -> Callable[[], None]:
        return self._transport.add_server_request_handler(self._handle_server_request)

    @staticmethod
    def _get_str(payload: dict[str, object], key: str) -> str | None:
        value = payload.get(key)
        return value if isinstance(value, str) else None

    @staticmethod
    def _as_json_object(value: object) -> JsonObject | None:
        if not isinstance(value, dict):
            return None
        return cast("JsonObject", value)

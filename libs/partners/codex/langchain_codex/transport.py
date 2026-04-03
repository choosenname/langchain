"""JSON-RPC transport for the Codex app server."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable


class CodexAppServerTransport:
    """Manage newline-delimited JSON-RPC process I/O."""

    def __init__(
        self,
        *,
        process: Any,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the transport.

        Args:
            process: Subprocess-like object with `stdin` and `stdout` file handles.
            on_notification: Optional callback invoked for JSON-RPC notifications.
        """
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._on_notification = on_notification
        self._next_id = 1
        self._lock = threading.Lock()
        self._response_events: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start reading the process stdout in the background."""
        with self._lock:
            if self._reader_thread is not None and self._reader_thread.is_alive():
                return

            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True,
            )
            self._reader_thread.start()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the matching result payload."""
        request_id = self._next_request_id()
        response_event = threading.Event()

        with self._lock:
            cached_response = self._responses.pop(request_id, None)
            if cached_response is not None:
                return cached_response["result"]
            self._response_events[request_id] = response_event

        self.start()
        self._write({"id": request_id, "method": method, "params": params})
        response_event.wait()

        with self._lock:
            response = self._responses.pop(request_id)
            self._response_events.pop(request_id, None)

        return response["result"]

    def _next_request_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
        return request_id

    def _write(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        self._stdin.write(f"{payload}\n")
        self._stdin.flush()

    def _reader_loop(self) -> None:
        for line in iter(self._stdout.readline, ""):
            if not line:
                break
            message = json.loads(line)
            if "id" in message:
                self._deliver_response(message)
            else:
                self._on_notification_message(message)

    def _deliver_response(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        with self._lock:
            response_event = self._response_events.get(request_id)
            self._responses[request_id] = message

        if response_event is not None:
            response_event.set()

    def _on_notification_message(self, message: dict[str, Any]) -> None:
        if self._on_notification is not None:
            self._on_notification(message)

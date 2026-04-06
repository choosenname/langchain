"""JSON-RPC transport for the Codex app server."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from langchain_codex._types import (
    AppServerProcess,
    JsonObject,
    as_json_object,
    get_json_object,
    get_str,
)
from langchain_codex.errors import CodexTransportError


class CodexAppServerTransport:
    """Manage newline-delimited JSON-RPC process I/O."""

    def __init__(
        self,
        *,
        process: AppServerProcess,
        on_notification: Callable[[JsonObject], None] | None = None,
        request_timeout: float | None = 30.0,
    ) -> None:
        """Initialize the transport.

        Args:
            process: Subprocess-like object with `stdin` and `stdout` file handles.
            on_notification: Optional callback invoked for JSON-RPC notifications.
            request_timeout: Maximum seconds to wait for a JSON-RPC response.
        """
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._request_timeout = request_timeout
        self._next_id = 1
        self._lock = threading.Lock()
        self._response_events: dict[int, threading.Event] = {}
        self._responses: dict[int, JsonObject] = {}
        self._reader_thread: threading.Thread | None = None
        self._notification_handlers: list[Callable[[JsonObject], None]] = []
        self._reader_error: CodexTransportError | None = None
        if on_notification is not None:
            self._notification_handlers.append(on_notification)

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

    def request(self, method: str, params: JsonObject) -> JsonObject:
        """Send a JSON-RPC request and return the matching result payload."""
        request_id = self._next_request_id()
        response_event = threading.Event()

        with self._lock:
            if self._reader_error is not None:
                raise self._reader_error
            self._response_events[request_id] = response_event

        self.start()
        try:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
        except OSError as exc:
            with self._lock:
                self._response_events.pop(request_id, None)
            msg = "Failed to write request to Codex app-server."
            raise CodexTransportError(msg) from exc

        response_event.wait(timeout=self._request_timeout)

        with self._lock:
            response = self._responses.pop(request_id, None)
            self._response_events.pop(request_id, None)
            reader_error = self._reader_error

        if response is not None:
            return self._extract_result(method, response)
        if reader_error is not None:
            raise reader_error

        msg = f"Timed out waiting for Codex app-server response to {method!r}."
        raise CodexTransportError(msg)

    def notify(self, method: str, params: JsonObject) -> None:
        """Send a JSON-RPC notification."""
        self.start()
        self._write(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def add_notification_handler(
        self,
        on_notification: Callable[[JsonObject], None],
    ) -> Callable[[], None]:
        """Register an additional notification handler and return a remover."""
        with self._lock:
            self._notification_handlers.append(on_notification)

        def remove_handler() -> None:
            with self._lock:
                self._notification_handlers.remove(on_notification)

        return remove_handler

    def _next_request_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
        return request_id

    def _write(self, message: JsonObject) -> None:
        payload = json.dumps(message)
        self._stdin.write(f"{payload}\n")
        self._stdin.flush()

    def _reader_loop(self) -> None:
        try:
            for line in iter(self._stdout.readline, ""):
                if not line:
                    break
                try:
                    raw_message = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = "Codex app-server emitted invalid JSON-RPC output."
                    self._set_reader_error(CodexTransportError(msg))
                    raise CodexTransportError(msg) from exc
                message = as_json_object(raw_message)
                if message is None:
                    msg = "Codex app-server emitted invalid JSON-RPC output."
                    self._set_reader_error(CodexTransportError(msg))
                    raise CodexTransportError(msg)
                if "id" in message:
                    self._deliver_response(message)
                else:
                    self._on_notification_message(message)
        except CodexTransportError:
            return
        except OSError as exc:
            msg = "Failed while reading from Codex app-server."
            self._set_reader_error(CodexTransportError(msg))
            raise CodexTransportError(msg) from exc

        self._set_reader_error(CodexTransportError(self._process_exit_message()))

    def _deliver_response(self, message: JsonObject) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, int):
            msg = "Codex app-server response did not include a numeric id."
            self._set_reader_error(CodexTransportError(msg))
            return
        with self._lock:
            response_event = self._response_events.get(request_id)
            self._responses[request_id] = message

        if response_event is not None:
            response_event.set()

    def _on_notification_message(self, message: JsonObject) -> None:
        with self._lock:
            handlers = list(self._notification_handlers)

        for handler in handlers:
            handler(message)

    def _set_reader_error(self, error: CodexTransportError) -> None:
        with self._lock:
            if self._reader_error is not None:
                return
            self._reader_error = error
            response_events = list(self._response_events.values())

        for response_event in response_events:
            response_event.set()

    def _extract_result(self, method: str, response: JsonObject) -> JsonObject:
        error = get_json_object(response, "error")
        if error is not None:
            error_message = get_str(error, "message") or "unknown error"
            msg = f"Codex app-server request failed for {method!r}: {error_message}"
            raise CodexTransportError(msg)

        result = get_json_object(response, "result")
        if result is None:
            msg = f"Codex app-server response for {method!r} did not include a result object."
            raise CodexTransportError(msg)
        return result

    def _process_exit_message(self) -> str:
        returncode = self._process.poll()
        if isinstance(returncode, int):
            return (
                "Codex app-server exited before responding "
                f"(exit code {returncode})."
            )
        return "Codex app-server closed stdout before responding."

"""Stdio JSON-RPC transport for Codex app-server."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from typing import Protocol, cast

from langchain_codex.errors import CodexServerRequestError, CodexTransportError


class _TextReader(Protocol):
    def readline(self) -> str:
        """Read one line."""
        ...


class _TextWriter(Protocol):
    def write(self, data: str) -> int:
        """Write one line."""
        ...

    def flush(self) -> None:
        """Flush buffered writes."""
        ...


class AppServerProcess(Protocol):
    """Subprocess protocol required by the stdio transport."""

    stdin: _TextWriter
    stdout: _TextReader
    stderr: _TextReader | None

    def poll(self) -> int | None:
        """Return the current process exit code when available."""
        ...


class StdioCodexTransport:
    """Manage newline-delimited JSON-RPC process I/O over stdio."""

    def __init__(
        self,
        *,
        process: object,
        request_timeout: float | None = None,
    ) -> None:
        """Initialize the transport."""
        typed_process = cast("AppServerProcess", process)
        self._process = typed_process
        self._stdin = typed_process.stdin
        self._stdout = typed_process.stdout
        self._stderr = typed_process.stderr
        self._request_timeout = request_timeout
        self._next_id = 1
        self._lock = threading.Lock()
        self._response_events: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, object]] = {}
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._reader_error: CodexTransportError | None = None
        self._notification_handlers: list[Callable[[dict[str, object]], None]] = []
        self._server_request_handlers: list[Callable[[object], object | None]] = []
        self._recent_stderr_lines: deque[str] = deque(maxlen=20)

    def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Send a request and return the response result."""
        request_id = self._next_request_id()
        response_event = threading.Event()
        with self._lock:
            if self._reader_error is not None:
                raise self._reader_error
            self._response_events[request_id] = response_event
        self._start()
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
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
        raise CodexTransportError(self._with_diagnostics(msg))

    def notify(self, method: str, params: dict[str, object]) -> None:
        """Send a notification."""
        self._start()
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def add_notification_handler(
        self,
        handler: Callable[[dict[str, object]], None],
    ) -> Callable[[], None]:
        """Register a notification handler."""
        with self._lock:
            self._notification_handlers.append(handler)

        def remove_handler() -> None:
            with self._lock:
                self._notification_handlers.remove(handler)

        return remove_handler

    def add_server_request_handler(
        self,
        handler: Callable[[object], object | None],
    ) -> Callable[[], None]:
        """Register a server-request handler."""
        with self._lock:
            self._server_request_handlers.append(handler)

        def remove_handler() -> None:
            with self._lock:
                self._server_request_handlers.remove(handler)

        return remove_handler

    def diagnostics(self) -> str | None:
        """Return recent stderr diagnostics."""
        if self._stderr_thread is not None and self._process.poll() is not None:
            self._stderr_thread.join(timeout=0.05)
        with self._lock:
            if not self._recent_stderr_lines:
                return None
            recent = list(self._recent_stderr_lines)
        return "Recent Codex app-server diagnostics:\n" + "\n".join(recent)

    def close(self) -> None:
        """Close the transport."""

    def _start(self) -> None:
        with self._lock:
            if self._reader_thread is not None and self._reader_thread.is_alive():
                return
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            if self._stderr is not None:
                self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
                self._stderr_thread.start()

    def _reader_loop(self) -> None:
        try:
            for line in iter(self._stdout.readline, ""):
                if not line:
                    break
                self._dispatch_message(line)
        except Exception as exc:  # pragma: no cover - surfaced to request callers
            self._set_reader_error(CodexTransportError(str(exc)))
            return
        self._set_reader_error(CodexTransportError(self._process_exit_message()))

    def _stderr_loop(self) -> None:
        if self._stderr is None:
            return
        try:
            for line in iter(self._stderr.readline, ""):
                if not line:
                    break
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                with self._lock:
                    self._recent_stderr_lines.append(cleaned)
        except OSError:
            return

    def _dispatch_message(self, line: str) -> None:
        message = self._parse_message(line)
        if self._is_server_request(message):
            self._handle_server_request(message)
            return
        if self._is_response(message):
            self._deliver_response(message)
            return
        self._handle_notification(message)

    @staticmethod
    def _is_server_request(message: dict[str, object]) -> bool:
        return "id" in message and isinstance(message.get("method"), str)

    @staticmethod
    def _is_response(message: dict[str, object]) -> bool:
        return "id" in message and "method" not in message

    def _handle_server_request(self, message: dict[str, object]) -> None:
        with self._lock:
            handlers = list(self._server_request_handlers)
        for handler in handlers:
            result = handler(dict(message))
            if result is None:
                continue
            request_id = message.get("id")
            if not isinstance(request_id, int):
                msg = "Server request missing numeric id"
                raise CodexServerRequestError(msg)
            self._write({"jsonrpc": "2.0", "id": request_id, "result": result})
            return
        method = message.get("method")
        msg = f"Unhandled server request: {method}"
        raise CodexServerRequestError(msg)

    def _deliver_response(self, message: dict[str, object]) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, int):
            self._set_reader_error(CodexTransportError("Response missing numeric id"))
            return
        with self._lock:
            self._responses[request_id] = message
            event = self._response_events.get(request_id)
        if event is not None:
            event.set()

    def _handle_notification(self, message: dict[str, object]) -> None:
        with self._lock:
            handlers = list(self._notification_handlers)
        for handler in handlers:
            handler(dict(message))

    @staticmethod
    def _parse_message(line: str) -> dict[str, object]:
        raw_message = json.loads(line)
        if not isinstance(raw_message, dict):
            msg = "Codex app-server emitted invalid JSON-RPC output."
            raise CodexTransportError(msg)
        raw_mapping = cast("dict[object, object]", raw_message)
        if not all(isinstance(key, str) for key in raw_mapping):
            msg = "Codex app-server emitted invalid JSON-RPC output."
            raise CodexTransportError(msg)
        return cast("dict[str, object]", raw_mapping)

    def _write(self, message: dict[str, object]) -> None:
        payload = json.dumps(message)
        self._stdin.write(f"{payload}\n")
        self._stdin.flush()

    def _extract_result(
        self,
        method: str,
        response: dict[str, object],
    ) -> dict[str, object]:
        error = response.get("error")
        if isinstance(error, dict):
            error_payload = cast("dict[str, object]", error)
            message = error_payload.get("message")
            if isinstance(message, str):
                raise CodexTransportError(message)
            msg = f"{method} returned a JSON-RPC error"
            raise CodexTransportError(msg)
        result = response.get("result")
        if not isinstance(result, dict):
            return {}
        return cast("dict[str, object]", result)

    def _next_request_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            return request_id

    def _set_reader_error(self, error: CodexTransportError) -> None:
        with self._lock:
            if self._reader_error is None:
                self._reader_error = error
            events = list(self._response_events.values())
        for event in events:
            event.set()

    def _process_exit_message(self) -> str:
        returncode = self._process.poll()
        if returncode is None:
            return self._with_diagnostics("Codex app-server disconnected before responding.")
        return self._with_diagnostics(
            f"Codex app-server exited before responding (exit code {returncode})."
        )

    def _with_diagnostics(self, message: str) -> str:
        diagnostics = self.diagnostics()
        if diagnostics is None:
            return message
        return f"{message}\n\n{diagnostics}"

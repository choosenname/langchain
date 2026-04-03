"""Session lifecycle management for the Codex app server."""

from __future__ import annotations

import threading
from typing import Any, Callable, Protocol


class _CodexTransport(Protocol):
    """Protocol for the transport methods used by `CodexSession`."""

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return the JSON-RPC result payload."""

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification."""

    def set_notification_handler(
        self,
        on_notification: Callable[[dict[str, Any]], None] | None,
    ) -> Callable[[dict[str, Any]], None] | None:
        """Install a notification handler and return the previous one."""


class CodexSession:
    """Own a single app-server connection and a persistent thread."""

    _CLIENT_INFO: dict[str, str] = {
        "name": "langchain_codex",
        "title": "LangChain Codex",
        "version": "0.1.0",
    }

    def __init__(self, *, transport: _CodexTransport, model: str) -> None:
        """Initialize the session.

        Args:
            transport: App-server transport used for JSON-RPC communication.
            model: Model name used when creating the persistent thread.
        """
        self._transport = transport
        self.model = model
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
        self.ensure_started()
        thread_id = self._ensure_thread()

        events: list[dict[str, Any]] = []
        turn_completed = threading.Event()

        def collect_notification(message: dict[str, Any]) -> None:
            events.append(message)
            if message.get("method") == "turn/completed":
                turn_completed.set()

        previous_handler = self._transport.set_notification_handler(collect_notification)
        try:
            result = self._transport.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": input_items,
                },
            )
            turn_completed.wait()
        finally:
            self._transport.set_notification_handler(previous_handler)

        return {
            "thread": {"id": thread_id},
            "turn": result["turn"],
            "events": events,
        }

    def _ensure_thread(self) -> str:
        """Create the app-server thread lazily and reuse it thereafter."""
        if self._thread_id is None:
            result = self._transport.request("thread/start", {"model": self.model})
            self._thread_id = result["thread"]["id"]
        return self._thread_id

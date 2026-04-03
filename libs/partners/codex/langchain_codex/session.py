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
        buffered_notifications: list[dict[str, Any]] = []
        notification_condition = threading.Condition()
        turn_completed = False

        def collect_notification(message: dict[str, Any]) -> None:
            with notification_condition:
                buffered_notifications.append(message)
                notification_condition.notify_all()

        previous_handler = self._transport.set_notification_handler(collect_notification)
        try:
            result = self._transport.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": input_items,
                },
            )
            turn_id = result["turn"]["id"]

            while not turn_completed:
                with notification_condition:
                    while not buffered_notifications and not turn_completed:
                        notification_condition.wait()
                    pending_notifications = buffered_notifications[:]
                    buffered_notifications.clear()

                for message in pending_notifications:
                    if not self._is_turn_notification_for(
                        message,
                        turn_id,
                    ):
                        continue

                    events.append(message)
                    if message.get("method") == "turn/completed":
                        turn_completed = True
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

"""Transport interfaces for Codex app-server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

NotificationHandler = Callable[[dict[str, object]], None]
ServerRequestHandler = Callable[[object], object | None]


class CodexTransport(Protocol):
    """Protocol implemented by transport backends."""

    def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Send a request and wait for the result."""
        ...

    def notify(self, method: str, params: dict[str, object]) -> None:
        """Send a notification."""
        ...

    def add_notification_handler(
        self,
        handler: NotificationHandler,
    ) -> Callable[[], None]:
        """Register a notification handler."""
        ...

    def add_server_request_handler(
        self,
        handler: ServerRequestHandler,
    ) -> Callable[[], None]:
        """Register a server-request handler."""
        ...

    def diagnostics(self) -> str | None:
        """Return recent transport diagnostics."""
        ...

    def close(self) -> None:
        """Close transport resources."""
        ...

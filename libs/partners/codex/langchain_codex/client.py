"""Provider-native Codex client."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

from langchain_codex.errors import CodexError
from langchain_codex.protocol.requests import normalize_launch_command
from langchain_codex.session import CodexSession
from langchain_codex.transport.stdio import StdioCodexTransport

if TYPE_CHECKING:
    from langchain_codex.types import (
        CodexApprovalDecision,
        CodexApprovalRequest,
        CodexClientConfig,
    )


class CodexClient:
    """Own Codex app-server configuration and create sessions."""

    def __init__(
        self,
        config: CodexClientConfig,
        *,
        transport_factory: Callable[[CodexClientConfig], Any] | None = None,
    ) -> None:
        """Initialize the provider client.

        Args:
            config: Shared provider configuration.
            transport_factory: Optional callable that creates a transport object.
        """
        self.config = config
        self._transport_factory = transport_factory
        self._transport: Any | None = None

    def create_session(
        self,
        *,
        approval_handler: Callable[[CodexApprovalRequest], CodexApprovalDecision] | None = None,
        thread_id: str | None = None,
    ) -> CodexSession:
        """Create a session bound to this client configuration.

        Args:
            approval_handler: Optional blocking handler for app-server approval requests.
            thread_id: Optional existing thread id to continue.
        """
        return CodexSession(
            transport=self._ensure_transport(),
            config=self.config,
            approval_handler=approval_handler,
            thread_id=thread_id,
        )

    def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Send a raw request over the transport."""
        return self._ensure_transport().request(method, params)

    def notify(self, method: str, params: dict[str, object]) -> None:
        """Send a raw notification over the transport."""
        self._ensure_transport().notify(method, params)

    def close(self) -> None:
        """Close the client.

        The stdio transport lifecycle is added in later implementation steps.
        """
        transport = self._transport
        if transport is None:
            return
        close_method = getattr(transport, "close", None)
        if callable(close_method):
            close_method()
        self._transport = None

    def __enter__(self) -> Self:
        """Return the client as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the client on context-manager exit."""
        self.close()

    @property
    def launch_command(self) -> tuple[str, ...]:
        """Return the normalized launch argv."""
        return normalize_launch_command(self.config.launch_command)

    def _ensure_transport(self) -> Any:
        if self._transport is not None:
            return self._transport
        if self._transport_factory is None:
            self._transport = self._launch_stdio_transport(self.config)
            return self._transport
        self._transport = self._transport_factory(self.config)
        return self._transport

    @staticmethod
    def _launch_stdio_transport(config: CodexClientConfig) -> StdioCodexTransport:
        process = subprocess.Popen(
            list(normalize_launch_command(config.launch_command)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            msg = "Codex app-server process did not expose stdio pipes."
            raise CodexError(msg)
        return StdioCodexTransport(
            process=process,
            request_timeout=config.request_timeout,
        )

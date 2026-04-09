"""Error types for the LangChain Codex integration."""

from __future__ import annotations


class CodexError(RuntimeError):
    """Base error raised by the LangChain Codex integration."""


class CodexInputError(CodexError):
    """Raised when a LangChain message cannot be represented for Codex."""


class CodexTransportError(CodexError):
    """Raised when the Codex app-server transport fails."""


class CodexProtocolError(CodexError):
    """Raised when the Codex app-server payload shape is invalid."""


class CodexServerRequestError(CodexError):
    """Raised when a server-initiated request cannot be handled."""

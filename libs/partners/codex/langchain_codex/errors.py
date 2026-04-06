"""Error types for the LangChain Codex integration."""

from __future__ import annotations


class CodexError(RuntimeError):
    """Base error raised by the LangChain Codex integration."""


class CodexInputError(CodexError):
    """Raised when a LangChain message cannot be represented for Codex."""


class CodexTransportError(CodexError):
    """Raised when the Codex app-server transport fails."""

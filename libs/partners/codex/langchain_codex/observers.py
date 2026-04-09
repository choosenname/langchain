"""Observer hooks for Codex provider lifecycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from langchain_codex.types import (
        CodexApprovalRequest,
        CodexThreadHandle,
        CodexTurnHandle,
    )


class CodexObserver(Protocol):
    """Observer interface for session lifecycle notifications."""

    def on_process_started(self) -> None:
        """Called when the app-server process starts."""

    def on_thread_selected(self, thread: CodexThreadHandle) -> None:
        """Called when a thread becomes active."""

    def on_turn_started(self, turn: CodexTurnHandle) -> None:
        """Called when a turn starts."""

    def on_approval_requested(self, request: CodexApprovalRequest) -> None:
        """Called when a blocking server request is received."""

    def on_approval_resolved(self, request_id: int) -> None:
        """Called when a server request resolves."""

    def on_turn_completed(self, turn: CodexTurnHandle) -> None:
        """Called when a turn completes."""

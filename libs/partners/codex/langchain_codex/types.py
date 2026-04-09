"""Public types for the Codex provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

JsonValue: TypeAlias = object
JsonObject: TypeAlias = dict[str, JsonValue]


def _tupled_strings(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(values)


def _empty_json_object() -> JsonObject:
    return {}


@dataclass(frozen=True)
class CodexClientConfig:
    """Configuration shared by a `CodexClient` and its sessions."""

    launch_command: tuple[str, ...]
    model: str | None = None
    cwd: str | None = None
    approval_policy: str | None = None
    sandbox_policy: JsonObject | None = None
    sandbox: JsonValue | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    personality: str | None = None
    service_name: str | None = None
    mcp_servers: tuple[JsonObject, ...] = ()
    include_default_mcp_config: bool | None = None
    request_timeout: float | None = None
    turn_timeout: float | None = None
    experimental_api: bool = False
    opt_out_notification_methods: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize tuple-backed configuration fields."""
        object.__setattr__(self, "launch_command", tuple(self.launch_command))
        object.__setattr__(
            self,
            "mcp_servers",
            tuple(dict(server) for server in self.mcp_servers),
        )
        object.__setattr__(
            self,
            "opt_out_notification_methods",
            _tupled_strings(self.opt_out_notification_methods),
        )


@dataclass(frozen=True)
class CodexThreadHandle:
    """Tracked thread metadata."""

    thread_id: str
    name: str | None = None
    status: JsonObject | None = None
    ephemeral: bool | None = None
    preview: str | None = None
    raw: JsonObject = field(default_factory=_empty_json_object)


@dataclass(frozen=True)
class CodexTurnHandle:
    """Tracked turn metadata."""

    turn_id: str
    status: str | None = None
    raw: JsonObject = field(default_factory=_empty_json_object)


@dataclass(frozen=True)
class CodexInputItem:
    """A parsed documented input item."""

    kind: str
    raw: JsonObject


@dataclass(frozen=True)
class CodexThreadItem:
    """A parsed thread item."""

    kind: str
    raw: JsonObject
    item_id: str | None = None


@dataclass(frozen=True)
class CodexEvent:
    """A parsed app-server notification."""

    method: str
    params: JsonObject
    raw: JsonObject
    item: CodexThreadItem | None = None


@dataclass(frozen=True)
class CodexServerRequest:
    """A server-initiated JSON-RPC request."""

    request_id: int
    method: str
    params: JsonObject
    raw: JsonObject


@dataclass(frozen=True)
class CodexServerResponse:
    """A server-request response payload."""

    result: JsonObject


@dataclass(frozen=True)
class CodexApprovalRequest:
    """A typed approval-bearing server request."""

    request_id: int
    method: str
    params: JsonObject
    thread_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True)
class CodexApprovalDecision:
    """A blocking approval decision returned by a user callback."""

    decision: JsonValue


@dataclass(frozen=True)
class CodexTurnResult:
    """Result of one completed turn."""

    thread: CodexThreadHandle
    turn: CodexTurnHandle
    events: tuple[CodexEvent, ...]
    output_text: str = ""

"""Request builders and method registries for Codex app-server."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from langchain_codex.protocol.base import JsonObject, compact_none_values

if TYPE_CHECKING:
    from langchain_codex.types import CodexClientConfig


DOCUMENTED_METHODS = (
    "initialize",
    "initialized",
    "model/list",
    "experimentalFeature/list",
    "experimentalFeature/enablement/set",
    "collaborationMode/list",
    "thread/start",
    "thread/resume",
    "thread/fork",
    "thread/read",
    "thread/list",
    "thread/loaded/list",
    "thread/metadata/update",
    "thread/name/set",
    "thread/archive",
    "thread/unarchive",
    "thread/unsubscribe",
    "thread/compact/start",
    "thread/shellCommand",
    "thread/backgroundTerminals/clean",
    "thread/rollback",
    "turn/start",
    "turn/steer",
    "turn/interrupt",
    "thread/realtime/start",
    "thread/realtime/appendAudio",
    "thread/realtime/appendText",
    "thread/realtime/stop",
    "review/start",
    "command/exec",
    "command/exec/write",
    "command/exec/resize",
    "command/exec/terminate",
    "fs/readFile",
    "fs/writeFile",
    "fs/createDirectory",
    "fs/getMetadata",
    "fs/readDirectory",
    "fs/remove",
    "fs/copy",
    "fs/watch",
    "fs/unwatch",
    "skills/list",
    "plugin/list",
    "plugin/read",
    "plugin/install",
    "plugin/uninstall",
    "skills/config/write",
    "app/list",
    "mcpServer/oauth/login",
    "tool/requestUserInput",
    "config/mcpServer/reload",
    "mcpServerStatus/list",
    "mcpServer/resource/read",
    "feedback/upload",
    "config/read",
    "externalAgentConfig/detect",
    "externalAgentConfig/import",
    "config/value/write",
    "config/batchWrite",
    "configRequirements/read",
    "account/read",
    "account/login/start",
    "account/login/cancel",
    "account/logout",
    "account/rateLimits/read",
    "windowsSandbox/setupStart",
)


def normalize_launch_command(command: Iterable[str]) -> tuple[str, ...]:
    """Normalize a launch command while preserving argv boundaries."""
    normalized = tuple(command)
    if not normalized:
        msg = "launch_command must not be empty"
        raise ValueError(msg)
    return normalized


def build_json_rpc_request(
    method: str,
    *,
    params: JsonObject | None = None,
    request_id: int,
) -> JsonObject:
    """Build a JSON-RPC request payload."""
    payload: JsonObject = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return payload


def build_initialize_request(
    config: CodexClientConfig,
    *,
    client_info: JsonObject,
    request_id: int,
) -> JsonObject:
    """Build the `initialize` request."""
    capabilities = compact_none_values(
        {
            "experimentalApi": config.experimental_api or None,
            "optOutNotificationMethods": list(config.opt_out_notification_methods)
            if config.opt_out_notification_methods
            else None,
        }
    )
    params: JsonObject = {"clientInfo": client_info}
    if capabilities:
        params["capabilities"] = capabilities
    return build_json_rpc_request("initialize", params=params, request_id=request_id)


def build_initialized_notification() -> JsonObject:
    """Build the `initialized` notification payload."""
    return {"jsonrpc": "2.0", "method": "initialized", "params": {}}


def build_thread_start_params(  # noqa: PLR0913
    *,
    model: str | None = None,
    cwd: str | None = None,
    approval_policy: str | None = None,
    sandbox_policy: JsonObject | None = None,
    personality: str | None = None,
    service_name: str | None = None,
    mcp_servers: Sequence[JsonObject] | None = None,
    include_default_mcp_config: bool | None = None,
    experimental_api: bool | None = None,
) -> JsonObject:
    """Build `thread/start` params using documented wire names."""
    return compact_none_values(
        {
            "model": model,
            "cwd": cwd,
            "approvalPolicy": approval_policy,
            "sandboxPolicy": sandbox_policy,
            "personality": personality,
            "serviceName": service_name,
            "mcpServers": list(mcp_servers) if mcp_servers else None,
            "includeDefaultMcpConfig": include_default_mcp_config,
            "experimentalApi": experimental_api,
        }
    )


def build_turn_start_params(  # noqa: PLR0913
    *,
    thread_id: str,
    input_items: Sequence[JsonObject],
    cwd: str | None = None,
    approval_policy: str | None = None,
    sandbox_policy: JsonObject | None = None,
    model: str | None = None,
    effort: str | None = None,
    summary: str | None = None,
    output_schema: JsonObject | None = None,
) -> JsonObject:
    """Build `turn/start` params."""
    return compact_none_values(
        {
            "threadId": thread_id,
            "input": list(input_items),
            "cwd": cwd,
            "approvalPolicy": approval_policy,
            "sandboxPolicy": sandbox_policy,
            "model": model,
            "effort": effort,
            "summary": summary,
            "outputSchema": output_schema,
        }
    )


def build_command_exec_params(  # noqa: PLR0913
    *,
    command: Sequence[str],
    process_id: str | None = None,
    cwd: str | None = None,
    env: JsonObject | None = None,
    size: JsonObject | None = None,
    sandbox_policy: JsonObject | None = None,
    output_bytes_cap: int | None = None,
    timeout_ms: int | None = None,
    tty: bool | None = None,
) -> JsonObject:
    """Build `command/exec` params."""
    return compact_none_values(
        {
            "command": list(command),
            "processId": process_id,
            "cwd": cwd,
            "env": env,
            "size": size,
            "sandboxPolicy": sandbox_policy,
            "outputBytesCap": output_bytes_cap,
            "timeoutMs": timeout_ms,
            "tty": tty,
        }
    )


def build_fs_watch_params(*, watch_id: str, path: str) -> JsonObject:
    """Build `fs/watch` params."""
    return {"watchId": watch_id, "path": path}

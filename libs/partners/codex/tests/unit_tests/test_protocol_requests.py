from __future__ import annotations

from langchain_codex.protocol import requests as protocol_requests
from langchain_codex.types import CodexClientConfig

EXPECTED_METHODS = {
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
    "windowsSandbox/setupStart",
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
}


def test_documented_method_registry_tracks_app_server_surface() -> None:
    assert set(protocol_requests.DOCUMENTED_METHODS) >= EXPECTED_METHODS


def test_initialize_request_builder_includes_client_capabilities() -> None:
    config = CodexClientConfig(
        launch_command=("ai-creds", "run", "codex", "app-server"),
        model="gpt-5.4",
        experimental_api=True,
        opt_out_notification_methods=(
            "thread/started",
            "item/agentMessage/delta",
        ),
    )

    request = protocol_requests.build_initialize_request(
        config,
        client_info={
            "name": "langchain_codex",
            "title": "LangChain Codex",
            "version": "0.1.0",
        },
        request_id=1,
    )

    assert request == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "clientInfo": {
                "name": "langchain_codex",
                "title": "LangChain Codex",
                "version": "0.1.0",
            },
            "capabilities": {
                "experimentalApi": True,
                "optOutNotificationMethods": [
                    "thread/started",
                    "item/agentMessage/delta",
                ],
            },
        },
    }


def test_thread_start_params_follow_documented_wire_names() -> None:
    params = protocol_requests.build_thread_start_params(
        model="gpt-5.4",
        cwd="/repo",
        approval_policy="never",
        sandbox_policy={"type": "dangerFullAccess"},
        personality="pragmatic",
        service_name="langchain-codex-tests",
        mcp_servers=[{"name": "docs", "command": ["uvx", "docs-mcp"]}],
        include_default_mcp_config=False,
        experimental_api=True,
    )

    assert params == {
        "model": "gpt-5.4",
        "cwd": "/repo",
        "approvalPolicy": "never",
        "sandboxPolicy": {"type": "dangerFullAccess"},
        "personality": "pragmatic",
        "serviceName": "langchain-codex-tests",
        "mcpServers": [{"name": "docs", "command": ["uvx", "docs-mcp"]}],
        "includeDefaultMcpConfig": False,
        "experimentalApi": True,
    }


def test_turn_start_params_support_documented_input_unions() -> None:
    params = protocol_requests.build_turn_start_params(
        thread_id="thr_123",
        input_items=[
            {"type": "text", "text": "$demo-app summarize this"},
            {"type": "mention", "name": "Demo App", "path": "app://demo-app"},
            {"type": "skill", "name": "skill-creator", "path": "/workspace/SKILL.md"},
        ],
        cwd="/repo",
        approval_policy="unlessTrusted",
        sandbox_policy={"type": "workspaceWrite"},
        model="gpt-5.4",
        effort="medium",
        summary="concise",
        output_schema={"type": "object"},
    )

    assert params == {
        "threadId": "thr_123",
        "input": [
            {"type": "text", "text": "$demo-app summarize this"},
            {"type": "mention", "name": "Demo App", "path": "app://demo-app"},
            {"type": "skill", "name": "skill-creator", "path": "/workspace/SKILL.md"},
        ],
        "cwd": "/repo",
        "approvalPolicy": "unlessTrusted",
        "sandboxPolicy": {"type": "workspaceWrite"},
        "model": "gpt-5.4",
        "effort": "medium",
        "summary": "concise",
        "outputSchema": {"type": "object"},
    }


def test_command_exec_params_preserve_streaming_and_sandbox_fields() -> None:
    params = protocol_requests.build_command_exec_params(
        command=["bash", "-i"],
        process_id="proc_123",
        cwd="/repo",
        env={"FOO": "bar", "DROP_ME": None},
        size={"rows": 48, "cols": 160},
        sandbox_policy={"type": "workspaceWrite"},
        output_bytes_cap=32768,
        timeout_ms=30000,
        tty=True,
    )

    assert params == {
        "command": ["bash", "-i"],
        "processId": "proc_123",
        "cwd": "/repo",
        "env": {"FOO": "bar", "DROP_ME": None},
        "size": {"rows": 48, "cols": 160},
        "sandboxPolicy": {"type": "workspaceWrite"},
        "outputBytesCap": 32768,
        "timeoutMs": 30000,
        "tty": True,
    }


def test_fs_watch_params_preserve_watch_id_and_absolute_path() -> None:
    assert protocol_requests.build_fs_watch_params(
        watch_id="watch_123",
        path="/repo/.git/HEAD",
    ) == {
        "watchId": "watch_123",
        "path": "/repo/.git/HEAD",
    }


def test_launch_command_normalization_keeps_wrapped_argv() -> None:
    assert protocol_requests.normalize_launch_command(["codex", "app-server"]) == (
        "codex",
        "app-server",
    )
    assert protocol_requests.normalize_launch_command(
        ["ai-creds", "run", "codex", "app-server"]
    ) == ("ai-creds", "run", "codex", "app-server")

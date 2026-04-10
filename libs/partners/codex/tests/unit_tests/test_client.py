from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import cast

from langchain_codex import CodexClient, CodexSession
from langchain_codex.transport.stdio import StdioCodexTransport
from langchain_codex.types import (
    CodexApprovalDecision,
    CodexApprovalRequest,
    CodexClientConfig,
    CodexThreadHandle,
    CodexTurnHandle,
    CodexTurnResult,
)


def test_client_config_preserves_wrapped_launch_argv_without_shell_parsing() -> None:
    config = CodexClientConfig(
        launch_command=("ai-creds", "run", "codex", "app-server"),
        model="gpt-5.4",
        approval_policy="never",
        opt_out_notification_methods=("thread/started",),
    )

    assert config.launch_command == ("ai-creds", "run", "codex", "app-server")
    assert config.model == "gpt-5.4"
    assert config.approval_policy == "never"
    assert config.opt_out_notification_methods == ("thread/started",)


def test_provider_exports_define_approval_and_turn_models() -> None:
    approval_request = CodexApprovalRequest(
        request_id=7,
        method="item/commandExecution/requestApproval",
        params={"threadId": "thr_123", "turnId": "turn_123"},
        thread_id="thr_123",
        turn_id="turn_123",
    )
    approval_decision = CodexApprovalDecision(decision="accept")
    turn_result = CodexTurnResult(
        thread=CodexThreadHandle(thread_id="thr_123", name="main-thread"),
        turn=CodexTurnHandle(turn_id="turn_123", status="completed"),
        events=(),
        output_text="done",
    )

    assert approval_request.method == "item/commandExecution/requestApproval"
    assert approval_decision.decision == "accept"
    assert turn_result.thread.thread_id == "thr_123"
    assert turn_result.turn.turn_id == "turn_123"
    assert turn_result.output_text == "done"


def test_package_exports_provider_native_client_surface() -> None:
    assert CodexClient is not None
    assert CodexSession is not None


def test_client_create_session_accepts_approval_handler_and_thread_id() -> None:
    config = CodexClientConfig(
        launch_command=("ai-creds", "run", "codex", "app-server"),
        model="gpt-5.4",
        approval_policy="on-request",
    )

    def approval_handler(request: CodexApprovalRequest) -> CodexApprovalDecision:
        assert request.method
        return CodexApprovalDecision(decision="accept")

    class FakeTransport:
        def __init__(self) -> None:
            self.requests: list[tuple[str, dict[str, object]]] = []
            self.notification_handler: Callable[[dict[str, object]], None] | None = None
            self.server_request_handler: Callable[[object], object | None] | None = None

        def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
            self.requests.append((method, params))
            if method == "turn/start":
                if self.notification_handler is None:
                    msg = "notification handler was not registered"
                    raise AssertionError(msg)
                self.notification_handler(
                    {
                        "method": "turn/completed",
                        "params": {"turn": {"id": "turn_1", "status": "completed"}},
                    }
                )
                return {"turn": {"id": "turn_1", "status": "running"}}
            return {}

        def notify(self, method: str, params: dict[str, object]) -> None:
            return None

        def add_notification_handler(
            self,
            handler: Callable[[dict[str, object]], None],
        ) -> Callable[[], None]:
            self.notification_handler = handler
            return lambda: None

        def add_server_request_handler(
            self,
            handler: Callable[[object], object | None],
        ) -> Callable[[], None]:
            self.server_request_handler = handler
            return lambda: None

    fake_transport = FakeTransport()
    client = CodexClient(config, transport_factory=lambda _config: fake_transport)

    session = client.create_session(
        approval_handler=approval_handler,
        thread_id="thr_existing",
    )

    assert isinstance(session, CodexSession)
    assert fake_transport.server_request_handler is not None
    response = fake_transport.server_request_handler(
        {
            "id": 7,
            "method": "item/commandExecution/requestApproval",
            "params": {"threadId": "thr_existing", "turnId": "turn_1"},
        }
    )
    assert response == {"decision": "accept"}

    session.run_turn([{"type": "text", "text": "continue the docs"}])

    turn_start_params = [
        params for method, params in fake_transport.requests if method == "turn/start"
    ]
    assert turn_start_params[0]["threadId"] == "thr_existing"


class FakeStdin:
    def __init__(self) -> None:
        self.written: list[str] = []
        self._condition = threading.Condition()

    def write(self, data: str) -> int:
        with self._condition:
            self.written.append(data)
            self._condition.notify_all()
        return len(data)

    def flush(self) -> None:
        return None

    def wait_for_writes(self, count: int) -> None:
        with self._condition:
            while len(self.written) < count:
                self._condition.wait()


class ControlledStdout:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._lines: list[str] = []

    def readline(self) -> str:
        with self._condition:
            while not self._lines:
                self._condition.wait()
            return self._lines.pop(0)

    def release(self, line: str) -> None:
        with self._condition:
            self._lines.append(line)
            self._condition.notify_all()


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = FakeStdin()
        self.stdout = ControlledStdout()
        self.stderr = ControlledStdout()

    def poll(self) -> int | None:
        return None


def test_stdio_transport_matches_out_of_order_responses_by_id() -> None:
    process = FakeProcess()
    transport = StdioCodexTransport(process=cast("object", process))

    results: dict[str, dict[str, object]] = {}
    errors: list[BaseException] = []

    def run_request(name: str, model: str) -> None:
        try:
            results[name] = transport.request("thread/start", {"model": model})
        except Exception as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    first = threading.Thread(target=run_request, args=("first", "gpt-5.4"))
    second = threading.Thread(target=run_request, args=("second", "gpt-5.4-mini"))

    first.start()
    process.stdin.wait_for_writes(1)
    second.start()
    process.stdin.wait_for_writes(2)

    first_request = json.loads(process.stdin.written[0])
    second_request = json.loads(process.stdin.written[1])

    assert first_request["id"] == 1
    assert second_request["id"] == 2

    process.stdout.release('{"jsonrpc":"2.0","id":2,"result":{"thread":{"id":"thr_2"}}}\n')
    process.stdout.release('{"jsonrpc":"2.0","id":1,"result":{"thread":{"id":"thr_1"}}}\n')

    first.join(timeout=1)
    second.join(timeout=1)

    assert errors == []
    assert results == {
        "first": {"thread": {"id": "thr_1"}},
        "second": {"thread": {"id": "thr_2"}},
    }


def test_stdio_transport_dispatches_server_requests_to_registered_handler() -> None:
    process = FakeProcess()
    transport = StdioCodexTransport(process=cast("object", process))
    seen_requests: list[dict[str, object]] = []
    transport.add_server_request_handler(
        lambda request: (
            seen_requests.append(cast("dict[str, object]", request)) or {"decision": "accept"}
        )
    )

    thread = threading.Thread(
        target=lambda: transport.request("thread/start", {"model": "gpt-5.4"})
    )
    thread.start()
    process.stdin.wait_for_writes(1)
    process.stdout.release(
        '{"jsonrpc":"2.0","id":77,"method":"item/commandExecution/requestApproval",'
        '"params":{"threadId":"thr_1","turnId":"turn_1"}}\n'
    )
    process.stdout.release('{"jsonrpc":"2.0","id":1,"result":{"thread":{"id":"thr_1"}}}\n')
    process.stdin.wait_for_writes(2)
    thread.join(timeout=1)

    response_message = json.loads(process.stdin.written[1])

    assert seen_requests[0]["method"] == "item/commandExecution/requestApproval"
    assert response_message == {
        "jsonrpc": "2.0",
        "id": 77,
        "result": {"decision": "accept"},
    }

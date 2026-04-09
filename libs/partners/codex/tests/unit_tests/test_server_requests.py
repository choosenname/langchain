from __future__ import annotations

from collections.abc import Callable

from langchain_codex.session import CodexSession
from langchain_codex.types import (
    CodexApprovalDecision,
    CodexApprovalRequest,
    CodexClientConfig,
)


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.notifications: list[dict[str, object]] = []
        self.server_request_handlers: list[Callable[[object], object | None]] = []
        self.notification_handlers: list[Callable[[dict[str, object]], None]] = []
        self.server_request_results: list[object] = []
        self.pending_server_request: tuple[int, str, dict[str, object]] | None = None

    def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.requests.append((method, params))
        if method == "initialize":
            return {"serverInfo": {"name": "codex"}}
        if method == "thread/start":
            return {"thread": {"id": "thr_123", "status": {"type": "idle"}}}
        if method == "turn/start":
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/started",
                    "params": {"turn": {"id": "turn_123", "status": "inProgress"}},
                }
            )
            if self.pending_server_request is not None:
                request_id, request_method, request_params = self.pending_server_request
                self.pending_server_request = None
                self._dispatch_server_request(request_id, request_method, request_params)
                self._emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "serverRequest/resolved",
                        "params": {"threadId": "thr_123", "requestId": request_id},
                    }
                )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "item/completed",
                    "params": {
                        "turnId": "turn_123",
                        "item": {
                            "id": "msg_123",
                            "type": "agentMessage",
                            "text": "done",
                        },
                    },
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {
                        "turn": {"id": "turn_123", "status": "completed"},
                    },
                }
            )
            return {"turn": {"id": "turn_123", "status": "inProgress"}}
        return {}

    def notify(self, method: str, params: dict[str, object]) -> None:
        self.notifications.append({"jsonrpc": "2.0", "method": method, "params": params})

    def add_notification_handler(
        self,
        handler: Callable[[dict[str, object]], None],
    ) -> Callable[[], None]:
        self.notification_handlers.append(handler)

        def remove_handler() -> None:
            self.notification_handlers.remove(handler)

        return remove_handler

    def add_server_request_handler(
        self,
        handler: Callable[[object], object | None],
    ) -> Callable[[], None]:
        self.server_request_handlers.append(handler)

        def remove_handler() -> None:
            self.server_request_handlers.remove(handler)

        return remove_handler

    def diagnostics(self) -> str | None:
        return None

    def close(self) -> None:
        return None

    def _emit(self, notification: dict[str, object]) -> None:
        for handler in list(self.notification_handlers):
            handler(notification)

    def _dispatch_server_request(
        self,
        request_id: int,
        method: str,
        params: dict[str, object],
    ) -> None:
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        for handler in list(self.server_request_handlers):
            response = handler(request)
            if response is not None:
                self.server_request_results.append(response)
                return
        msg = f"Unhandled server request: {method}"
        raise AssertionError(msg)


def _build_session(
    transport: FakeTransport,
    *,
    approval_handler: Callable[[CodexApprovalRequest], CodexApprovalDecision],
) -> CodexSession:
    return CodexSession(
        transport=transport,
        config=CodexClientConfig(
            launch_command=("codex", "app-server"),
            model="gpt-5.4",
            approval_policy="on-request",
        ),
        approval_handler=approval_handler,
    )


def test_session_blocks_for_command_approval_and_resumes_turn() -> None:
    transport = FakeTransport()
    seen_requests: list[CodexApprovalRequest] = []
    transport.pending_server_request = (
        77,
        "item/commandExecution/requestApproval",
        {
            "threadId": "thr_123",
            "turnId": "turn_123",
            "itemId": "cmd_123",
            "reason": "run tests",
        },
    )
    session = _build_session(
        transport,
        approval_handler=lambda request: (
            seen_requests.append(request) or CodexApprovalDecision(decision="accept")
        ),
    )

    result = session.run_turn([{"type": "text", "text": "run tests"}])

    assert seen_requests[0].method == "item/commandExecution/requestApproval"
    assert transport.server_request_results == [{"decision": "accept"}]
    assert result.turn.turn_id == "turn_123"
    assert result.output_text == "done"


def test_session_routes_request_user_input_through_same_blocking_handler() -> None:
    transport = FakeTransport()
    transport.pending_server_request = (
        88,
        "tool/requestUserInput",
        {
            "threadId": "thr_123",
            "turnId": "turn_123",
            "questions": [
                {
                    "id": "scope",
                    "header": "Scope",
                    "question": "Pick one",
                    "options": [],
                }
            ],
        },
    )
    session = _build_session(
        transport,
        approval_handler=lambda _request: CodexApprovalDecision(
            decision={"answers": {"scope": "recommended"}}
        ),
    )

    session.run_turn([{"type": "text", "text": "ask me"}])

    assert transport.server_request_results == [{"decision": {"answers": {"scope": "recommended"}}}]


def test_session_routes_auth_refresh_request_through_same_handler() -> None:
    transport = FakeTransport()
    seen_methods: list[str] = []
    transport.pending_server_request = (
        91,
        "account/chatgptAuthTokens/refresh",
        {"threadId": "thr_123", "turnId": "turn_123"},
    )
    session = _build_session(
        transport,
        approval_handler=lambda request: (
            seen_methods.append(request.method)
            or CodexApprovalDecision(decision={"status": "refreshed"})
        ),
    )

    session.run_turn([{"type": "text", "text": "refresh"}])

    assert seen_methods == ["account/chatgptAuthTokens/refresh"]
    assert transport.server_request_results == [{"decision": {"status": "refreshed"}}]

from __future__ import annotations

from collections.abc import Callable

from langchain_codex.session import CodexSession, TurnDelta
from langchain_codex.types import CodexClientConfig


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.notification_handlers: list[Callable[[dict[str, object]], None]] = []
        self.server_request_handlers: list[Callable[[object], object | None]] = []

    def request(  # noqa: PLR0911
        self,
        method: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        self.requests.append((method, params))
        if method == "initialize":
            return {"serverInfo": {"name": "codex"}}
        if method == "thread/start":
            return {
                "thread": {
                    "id": "thr_123",
                    "name": "Main",
                    "status": {"type": "idle"},
                    "ephemeral": False,
                }
            }
        if method == "thread/resume":
            return {"thread": {"id": params["threadId"], "status": {"type": "idle"}}}
        if method == "thread/fork":
            return {"thread": {"id": "thr_fork", "status": {"type": "idle"}}}
        if method == "thread/list":
            return {"data": [{"id": "thr_123"}, {"id": "thr_fork"}], "nextCursor": None}
        if method == "thread/loaded/list":
            return {"data": ["thr_123"]}
        if method == "thread/read":
            return {"thread": {"id": params["threadId"], "status": {"type": "notLoaded"}}}
        if method == "thread/rollback":
            return {"thread": {"id": params["threadId"], "turns": []}}
        if method == "turn/start":
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "item/agentMessage/delta",
                    "params": {"turnId": "turn_123", "delta": "Hel"},
                }
            )
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "item/agentMessage/delta",
                    "params": {"turnId": "turn_123", "delta": "lo"},
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
                            "text": "Hello",
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
        if method == "turn/steer":
            return {"turnId": "turn_123"}
        if method == "review/start":
            return {
                "turn": {"id": "turn_review", "status": "inProgress"},
                "reviewThreadId": params["threadId"],
            }
        return {}

    def notify(self, method: str, params: dict[str, object]) -> None:
        self.requests.append((method, params))

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


def _build_session(transport: FakeTransport) -> CodexSession:
    return CodexSession(
        transport=transport,
        config=CodexClientConfig(
            launch_command=("codex", "app-server"),
            model="gpt-5.4",
            approval_policy="on-request",
        ),
    )


def test_session_defers_thread_creation_until_first_turn() -> None:
    transport = FakeTransport()
    session = _build_session(transport)

    session.ensure_started()
    assert ("thread/start", {"model": "gpt-5.4", "approvalPolicy": "on-request"}) not in (
        transport.requests
    )

    session.run_turn([{"type": "text", "text": "hello"}])

    assert ("thread/start", {"model": "gpt-5.4", "approvalPolicy": "on-request"}) in (
        transport.requests
    )


def test_session_supports_resume_fork_list_read_and_rollback() -> None:
    transport = FakeTransport()
    session = _build_session(transport)

    resumed = session.resume_thread("thr_existing")
    forked = session.fork_thread("thr_existing")
    listed = session.list_threads()
    loaded = session.list_loaded_threads()
    read_thread = session.read_thread("thr_existing")
    rolled_back = session.rollback_thread("thr_existing", turns=2)

    assert resumed.thread_id == "thr_existing"
    assert forked.thread_id == "thr_fork"
    assert [thread.thread_id for thread in listed] == ["thr_123", "thr_fork"]
    assert loaded == ["thr_123"]
    assert read_thread.thread_id == "thr_existing"
    assert rolled_back.thread_id == "thr_existing"


def test_session_run_turn_returns_active_thread_turn_and_output_text() -> None:
    transport = FakeTransport()
    session = _build_session(transport)

    result = session.run_turn([{"type": "text", "text": "hello"}])

    assert result.thread.thread_id == "thr_123"
    assert result.turn.turn_id == "turn_123"
    assert result.output_text == "Hello"
    assert [event.method for event in result.events] == [
        "item/agentMessage/delta",
        "item/agentMessage/delta",
        "item/completed",
        "turn/completed",
    ]


def test_session_stream_turn_yields_text_deltas_then_terminal_metadata() -> None:
    transport = FakeTransport()
    session = _build_session(transport)

    deltas = list(session.stream_turn([{"type": "text", "text": "hello"}]))

    assert deltas == [
        TurnDelta(text="Hel"),
        TurnDelta(text="lo"),
        TurnDelta(
            text="",
            thread_id="thr_123",
            turn={"id": "turn_123", "status": "completed"},
            chunk_position="last",
        ),
    ]


def test_session_supports_turn_steering_interrupt_and_review() -> None:
    transport = FakeTransport()
    session = _build_session(transport)
    session.start_thread()

    turn_id = session.steer_turn(
        "thr_123",
        [{"type": "text", "text": "focus on tests"}],
        expected_turn_id="turn_123",
    )
    session.interrupt_turn("thr_123", "turn_123")
    review = session.start_review(
        "thr_123",
        target={"type": "uncommittedChanges"},
        delivery="inline",
    )

    assert turn_id == "turn_123"
    assert ("turn/interrupt", {"threadId": "thr_123", "turnId": "turn_123"}) in (transport.requests)
    assert review.turn.turn_id == "turn_review"


def test_session_known_threads_include_started_thread() -> None:
    transport = FakeTransport()
    session = _build_session(transport)

    started = session.start_thread()
    known = session.list_known_threads()

    assert started.thread_id == "thr_123"
    assert [thread.thread_id for thread in known] == ["thr_123"]

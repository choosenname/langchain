from __future__ import annotations

from collections.abc import Callable

from langchain_codex.session import CodexSession
from langchain_codex.types import CodexClientConfig


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []

    def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.requests.append((method, params))
        if method == "initialize":
            return {"serverInfo": {"name": "codex"}}
        return {}

    def notify(self, method: str, params: dict[str, object]) -> None:
        self.requests.append((method, params))

    def add_notification_handler(
        self,
        handler: Callable[[dict[str, object]], None],
    ) -> Callable[[], None]:
        def remove_handler() -> None:
            return None

        return remove_handler

    def add_server_request_handler(
        self,
        handler: Callable[[object], object | None],
    ) -> Callable[[], None]:
        def remove_handler() -> None:
            return None

        return remove_handler

    def diagnostics(self) -> str | None:
        return None

    def close(self) -> None:
        return None


def test_session_exposes_realtime_wrappers() -> None:
    transport = FakeTransport()
    session = CodexSession(
        transport=transport,
        config=CodexClientConfig(launch_command=("codex", "app-server")),
    )

    session.start_realtime("thr_123", prompt="You are on a call.")
    session.append_realtime_audio("thr_123", audio={"data": "AAA"})
    session.append_realtime_text("thr_123", text="hello")
    session.stop_realtime("thr_123")

    assert transport.requests == [
        (
            "initialize",
            {
                "clientInfo": {
                    "name": "langchain_codex",
                    "title": "LangChain Codex",
                    "version": "0.1.0",
                }
            },
        ),
        ("initialized", {}),
        (
            "thread/realtime/start",
            {"threadId": "thr_123", "prompt": "You are on a call."},
        ),
        ("thread/realtime/appendAudio", {"threadId": "thr_123", "audio": {"data": "AAA"}}),
        ("thread/realtime/appendText", {"threadId": "thr_123", "text": "hello"}),
        ("thread/realtime/stop", {"threadId": "thr_123"}),
    ]

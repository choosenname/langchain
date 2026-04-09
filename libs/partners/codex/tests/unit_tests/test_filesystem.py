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
        if method == "fs/readFile":
            return {"dataBase64": "aGVsbG8="}
        if method == "fs/getMetadata":
            return {
                "isDirectory": False,
                "isFile": True,
                "createdAtMs": 1,
                "modifiedAtMs": 2,
            }
        if method == "fs/readDirectory":
            return {"entries": [{"fileName": "README.md", "isDirectory": False, "isFile": True}]}
        if method == "fs/watch":
            return {"path": params["path"]}
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


def _build_session(transport: FakeTransport) -> CodexSession:
    return CodexSession(
        transport=transport,
        config=CodexClientConfig(launch_command=("codex", "app-server")),
    )


def test_session_exposes_documented_filesystem_wrappers() -> None:
    transport = FakeTransport()
    session = _build_session(transport)

    contents = session.read_file("/repo/README.md")
    session.write_file("/repo/README.md", data_base64="aGVsbG8=")
    session.create_directory("/repo/docs", recursive=True)
    metadata = session.get_metadata("/repo/README.md")
    entries = session.read_directory("/repo")
    session.remove_path("/repo/tmp", recursive=True, force=True)
    session.copy_path("/repo/README.md", "/repo/README-copy.md")
    watched_path = session.watch_path("watch_123", "/repo/README.md")
    session.unwatch_path("watch_123")

    assert contents == "aGVsbG8="
    assert metadata["isFile"] is True
    assert entries == [{"fileName": "README.md", "isDirectory": False, "isFile": True}]
    assert watched_path == "/repo/README.md"
    assert ("fs/unwatch", {"watchId": "watch_123"}) in transport.requests

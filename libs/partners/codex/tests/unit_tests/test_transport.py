from __future__ import annotations

from typing import Any

from langchain_codex.transport import CodexAppServerTransport


class FakeStdin:
    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, data: str) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        return None


class FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class FakeProcess:
    def __init__(self, *, stdout_lines: list[str]) -> None:
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(stdout_lines)


def test_transport_matches_response_by_id() -> None:
    process = FakeProcess(
        stdout_lines=[
            '{"id": 1, "result": {"thread": {"id": "thr_123"}}}\n',
        ]
    )
    transport = CodexAppServerTransport(process=process)
    result = transport.request("thread/start", {"model": "gpt-5.4"})
    assert result == {"thread": {"id": "thr_123"}}


def test_transport_routes_notifications() -> None:
    seen: list[dict[str, Any]] = []
    process = FakeProcess(
        stdout_lines=[
            '{"method": "turn/started", "params": {"turn": {"id": "turn_1"}}}\n',
        ]
    )
    transport = CodexAppServerTransport(process=process, on_notification=seen.append)
    transport.start()
    assert seen == [{"method": "turn/started", "params": {"turn": {"id": "turn_1"}}}]

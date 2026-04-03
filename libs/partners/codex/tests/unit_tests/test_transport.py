from __future__ import annotations

import json
import threading
from typing import Any

from langchain_codex.transport import CodexAppServerTransport


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


class ControlledProcess:
    def __init__(self) -> None:
        self.stdin = FakeStdin()
        self.stdout = ControlledStdout()


def test_transport_matches_out_of_order_responses_by_id() -> None:
    process = ControlledProcess()
    transport = CodexAppServerTransport(process=process)

    results: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []

    def run_request(name: str, model: str) -> None:
        try:
            results[name] = transport.request("thread/start", {"model": model})
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    first = threading.Thread(target=run_request, args=("first", "gpt-5.4"))
    second = threading.Thread(target=run_request, args=("second", "gpt-5.4-mini"))

    first.start()
    process.stdin.wait_for_writes(1)
    second.start()
    process.stdin.wait_for_writes(2)

    first_request = json.loads(process.stdin.written[0])
    second_request = json.loads(process.stdin.written[1])

    assert first_request == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "thread/start",
        "params": {"model": "gpt-5.4"},
    }
    assert second_request == {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "thread/start",
        "params": {"model": "gpt-5.4-mini"},
    }

    process.stdout.release(
        '{"jsonrpc": "2.0", "id": 2, "result": {"thread": {"id": "thr_2"}}}\n'
    )
    process.stdout.release(
        '{"jsonrpc": "2.0", "id": 1, "result": {"thread": {"id": "thr_1"}}}\n'
    )

    first.join(timeout=1)
    second.join(timeout=1)

    assert errors == []
    assert results == {
        "first": {"thread": {"id": "thr_1"}},
        "second": {"thread": {"id": "thr_2"}},
    }


def test_transport_routes_notifications() -> None:
    seen: list[dict[str, Any]] = []
    process = FakeProcess(
        stdout_lines=[
            '{"jsonrpc": "2.0", "method": "turn/started", "params": {"turn": {"id": "turn_1"}}}\n',
        ]
    )
    transport = CodexAppServerTransport(process=process, on_notification=seen.append)
    transport.start()
    assert seen == [
        {
            "jsonrpc": "2.0",
            "method": "turn/started",
            "params": {"turn": {"id": "turn_1"}},
        }
    ]

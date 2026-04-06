from __future__ import annotations

import json
import threading
from typing import cast

import pytest

from langchain_codex._types import AppServerProcess, JsonObject
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
    def __init__(
        self,
        *,
        stdout_lines: list[str],
        returncode: int | None = None,
    ) -> None:
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(stdout_lines)
        self._returncode = returncode

    def poll(self) -> int | None:
        return self._returncode


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

    def poll(self) -> int | None:
        return None


class InspectableTransport(CodexAppServerTransport):
    def wait_for_reader_thread(self) -> None:
        assert self._reader_thread is not None
        self._reader_thread.join(timeout=1)


def test_transport_matches_out_of_order_responses_by_id() -> None:
    process = ControlledProcess()
    transport = InspectableTransport(process=cast(AppServerProcess, process))

    results: dict[str, JsonObject] = {}
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
    seen: list[JsonObject] = []
    process = FakeProcess(
        stdout_lines=[
            '{"jsonrpc": "2.0", "method": "turn/started", "params": {"turn": {"id": "turn_1"}}}\n',
        ]
    )
    transport = InspectableTransport(
        process=cast(AppServerProcess, process),
        on_notification=seen.append,
    )
    transport.start()
    transport.wait_for_reader_thread()
    assert seen == [
        {
            "jsonrpc": "2.0",
            "method": "turn/started",
            "params": {"turn": {"id": "turn_1"}},
        }
    ]


def test_transport_routes_notifications_to_multiple_handlers() -> None:
    seen_primary: list[JsonObject] = []
    seen_secondary: list[JsonObject] = []
    process = FakeProcess(
        stdout_lines=[
            '{"jsonrpc": "2.0", "method": "turn/started", "params": {"turn": {"id": "turn_1"}}}\n',
        ]
    )
    transport = InspectableTransport(
        process=cast(AppServerProcess, process),
        on_notification=seen_primary.append,
    )
    transport.add_notification_handler(seen_secondary.append)
    transport.start()
    transport.wait_for_reader_thread()

    assert seen_primary == [
        {
            "jsonrpc": "2.0",
            "method": "turn/started",
            "params": {"turn": {"id": "turn_1"}},
        }
    ]
    assert seen_secondary == [
        {
            "jsonrpc": "2.0",
            "method": "turn/started",
            "params": {"turn": {"id": "turn_1"}},
        }
    ]


def test_transport_raises_json_rpc_error_message() -> None:
    process = FakeProcess(
        stdout_lines=[
            '{"jsonrpc": "2.0", "id": 1, "error": {"code": 123, "message": "Bad turn"}}\n',
        ]
    )
    transport = InspectableTransport(process=cast(AppServerProcess, process))

    with pytest.raises(RuntimeError, match="Bad turn"):
        transport.request("turn/start", {"threadId": "thr_1", "input": []})


def test_transport_raises_when_process_exits_before_response() -> None:
    process = FakeProcess(stdout_lines=[], returncode=17)
    transport = InspectableTransport(process=cast(AppServerProcess, process))

    with pytest.raises(RuntimeError, match="exit code 17"):
        transport.request("thread/start", {"model": "gpt-5.4"})

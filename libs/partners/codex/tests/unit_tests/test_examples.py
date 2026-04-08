from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

from langchain_core.messages import AIMessage


def _load_example_module(file_name: str) -> ModuleType:
    example_path = Path(__file__).parents[5] / "examples" / file_name
    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simple_invoke_example_exists_and_uses_chat_codex() -> None:
    example_path = Path(__file__).parents[5] / "examples" / "codex_simple_invoke.py"
    pyproject_path = Path(__file__).parents[5] / "examples" / "pyproject.toml"
    package_pyproject_path = Path(__file__).parents[2] / "pyproject.toml"

    assert example_path.exists()
    assert pyproject_path.exists()
    assert package_pyproject_path.exists()

    source = example_path.read_text()
    pyproject_source = pyproject_path.read_text()
    package_pyproject_source = package_pyproject_path.read_text()

    assert "from langchain_codex.chat_models import ChatCodex" in source
    assert 'codex_command="ai-creds run codex"' in source
    assert "for chunk in model.stream(" in source
    assert "def _truncate_preview(" in source
    assert "def _print_live_preview(" in source
    assert 'raise SystemExit(f"Codex example failed: {err}") from err' in source
    assert "print(response.content)" not in source
    assert 'name = "codex-simple-example"' in pyproject_source
    assert 'requires-python = ">=3.10,<3.14"' in pyproject_source
    assert '"langchain-codex>=0.1.0"' in pyproject_source
    assert 'langchain-codex = { path = "../libs/partners/codex", editable = true }' in (
        pyproject_source
    )
    assert 'requires-python = ">=3.10.0,<3.15"' in package_pyproject_source


def test_simple_invoke_example_truncates_live_preview_text() -> None:
    module = _load_example_module("codex_simple_invoke.py")

    assert module._truncate_preview("one\ntwo\nthree", max_lines=2, max_chars=80) == (
        "one\ntwo\n..."
    )
    assert module._truncate_preview("abcdefghij", max_lines=4, max_chars=8) == "abcde..."


def test_project_structuring_example_exists_and_is_multi_turn() -> None:
    example_path = (
        Path(__file__).parents[5] / "examples" / "codex_project_structuring.py"
    )

    assert example_path.exists()

    source = example_path.read_text()

    assert "MAX_REVIEW_ATTEMPTS = 3" in source
    assert "def build_repository_analysis_prompt() -> str:" in source
    assert "def build_execution_plan_prompt() -> str:" in source
    assert "def build_step_list_prompt() -> str:" in source
    assert "def build_step_implementation_prompt(" in source
    assert "def build_step_review_prompt(" in source
    assert "def build_judge_prompt(" in source
    assert "def _parse_json_response(" in source
    assert "def _message_text(message: AIMessage) -> str:" in source
    assert "def _truncate_preview(" in source
    assert "def _print_preview(" in source
    assert "def _summarize_step_results(" in source
    assert "def run_step_review_loop(" in source
    assert "analysis = model.invoke(build_repository_analysis_prompt())" in source
    assert "execution_plan = model.invoke(build_execution_plan_prompt())" in source
    assert 'return str(message.text)' in source
    assert (
        "step_plan = _parse_json_response(_message_text(model.invoke(build_step_list_prompt())))"
        in source
    )
    assert 'while review_attempts < MAX_REVIEW_ATTEMPTS:' in source
    assert 'decision == "continue_fix"' in source
    assert 'decision == "continue_next_step"' in source
    assert 'decision == "cancel_all"' in source
    assert 'approval_policy="never"' in source
    assert 'sandbox="danger-full-access"' in source
    assert "include a short rationale and the likely files or packages impacted" in source
    assert "Call out any phase that should wait for additional review or prerequisites." in source
    assert "Return valid JSON only." in source
    assert '"status": "ok"' in source
    assert '"status": "not_ok"' in source
    assert '"decision": "continue_fix"' in source
    assert 'print("# Step Execution Results")' in source
    assert 'print("# Repository Structuring Report")' in source
    assert 'print("# Phased Execution Plan")' in source
    assert 'print(json.dumps(step_results, indent=2))' not in source


def test_project_structuring_example_compacts_preview_and_step_results() -> None:
    module = _load_example_module("codex_project_structuring.py")

    assert module._truncate_preview("one\ntwo\nthree", max_lines=2, max_chars=80) == (
        "one\ntwo\n..."
    )
    assert module._truncate_preview("abcdefghij", max_lines=4, max_chars=8) == "abcde..."
    assert module._summarize_step_results(
        [
            {"step_id": "step-1", "outcome": "ok"},
            {
                "step_id": "step-2",
                "outcome": "continue_next_step",
                "judge_result": {"decision": "continue_next_step"},
            },
        ]
    ) == "step-1: ok\nstep-2: continue_next_step (judge: continue_next_step)"


def test_project_structuring_example_normalizes_message_content_to_text() -> None:
    module = _load_example_module("codex_project_structuring.py")

    assert module._message_text(AIMessage(content="implementation summary")) == (
        "implementation summary"
    )
    assert module._message_text(
        AIMessage(
            content=[
                {"type": "text", "text": "implementation"},
                " summary",
                {"type": "tool_use", "name": "ignored"},
            ]
        )
    ) == "implementation summary"


def test_cargo_fix_loop_example_exists_and_allows_direct_repo_inspection() -> None:
    example_path = Path(__file__).parents[5] / "examples" / "codex_cargo_fix_loop.py"

    assert example_path.exists()

    source = example_path.read_text()

    assert "from langchain_codex.chat_models import ChatCodex" in source
    assert "MAX_REVIEW_ATTEMPTS = 3" in source
    assert "def build_cargo_diagnosis_prompt(" in source
    assert "def build_fix_plan_prompt(" in source
    assert "def build_step_list_prompt(" in source
    assert "def build_step_implementation_prompt(" in source
    assert "def build_step_review_prompt(" in source
    assert "def build_judge_prompt(" in source
    assert "def _log_action(" in source
    assert "def _message_text(message: AIMessage) -> str:" in source
    assert "def _truncate_preview(" in source
    assert "def _print_preview(" in source
    assert "def _summarize_step_results(" in source
    assert "def run_step_review_loop(" in source
    assert "cargo check output" in source
    assert "Do not run commands." not in source
    assert "Use nix develop for every Rust command." in source
    assert "nix develop -c cargo check" in source
    assert "If the project does not provide a working flake dev shell" in source
    assert "Return valid JSON only." in source
    assert "Inspect the repository directly." in source
    assert 'return str(message.text)' in source
    assert '"status": "ok"' in source
    assert '"status": "not_ok"' in source
    assert '"decision": "continue_fix"' in source
    assert 'while review_attempts < MAX_REVIEW_ATTEMPTS:' in source
    assert 'decision == "continue_fix"' in source
    assert 'decision == "continue_next_step"' in source
    assert 'decision == "cancel_all"' in source
    assert 'approval_policy="never"' in source
    assert 'sandbox="danger-full-access"' in source
    assert 'fix_plan = model.invoke(build_fix_plan_prompt(_message_text(diagnosis)))' in source
    assert "_message_text(model.invoke(build_step_list_prompt(_message_text(fix_plan))))" in source
    assert 'Starting cargo diagnosis' in source
    assert 'Running implementer for' in source
    assert 'Running reviewer for' in source
    assert 'Judge decision for' in source
    assert 'print("# Cargo Diagnosis")' in source
    assert 'print("# Cargo Fix Plan")' in source
    assert 'print("# Cargo Step Execution Results")' in source
    assert 'print(json.dumps(step_results, indent=2))' not in source


def test_cargo_fix_loop_example_normalizes_message_content_to_text() -> None:
    module = _load_example_module("codex_cargo_fix_loop.py")

    assert module._message_text(AIMessage(content="cargo diagnosis")) == (
        "cargo diagnosis"
    )
    assert module._message_text(
        AIMessage(
            content=[
                {"type": "text", "text": "cargo"},
                " diagnosis",
                {"type": "tool_use", "name": "ignored"},
            ]
        )
    ) == "cargo diagnosis"


def test_cargo_fix_loop_example_compacts_preview_and_step_results() -> None:
    module = _load_example_module("codex_cargo_fix_loop.py")

    assert module._truncate_preview("one\ntwo\nthree", max_lines=2, max_chars=80) == (
        "one\ntwo\n..."
    )
    assert module._truncate_preview("abcdefghij", max_lines=4, max_chars=8) == "abcde..."
    assert module._summarize_step_results(
        [
            {"step_id": "step-1", "outcome": "ok"},
            {
                "step_id": "step-2",
                "outcome": "cancel_all",
                "judge_result": {"decision": "cancel_all"},
            },
        ]
    ) == "step-1: ok\nstep-2: cancel_all (judge: cancel_all)"


def test_cargo_fix_loop_logs_agent_actions() -> None:
    module = _load_example_module("codex_cargo_fix_loop.py")
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        module._log_action("reviewer", "step-1 failed")

    assert buffer.getvalue() == "[reviewer] step-1 failed\n"


def test_cargo_fix_loop_creates_missing_output_file() -> None:
    module = _load_example_module("codex_cargo_fix_loop.py")
    tmp_dir = Path("/tmp/codex-cargo-fix-loop-test")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / "cargo-check.txt"
    if output_path.exists():
        output_path.unlink()

    calls: list[tuple[tuple[str, ...], str]] = []

    class _Result:
        stdout = "error[E0001]: example failure\n"

    def _fake_run(
        command: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> _Result:
        calls.append((tuple(command), cwd))
        assert capture_output is True
        assert text is True
        assert check is False
        return _Result()

    module.subprocess.run = _fake_run
    content = module._read_cargo_check_output_file(
        "/tmp/rust-project",
        str(output_path),
    )

    assert calls == [(("nix", "develop", "-c", "cargo", "check"), "/tmp/rust-project")]
    assert output_path.read_text() == content
    assert "example failure" in content

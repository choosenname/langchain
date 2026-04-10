from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_example_module(relative_path: str) -> ModuleType:
    example_path = Path(__file__).parents[5] / "examples" / relative_path
    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_mentions_provider_native_client_and_wrapped_launch() -> None:
    readme_path = Path(__file__).parents[2] / "README.md"

    source = readme_path.read_text()

    assert "CodexClient" in source
    assert "CodexSession" in source
    assert "ai-creds run codex app-server" in source


def test_provider_runtime_doc_example_writes_a_base_project_doc(
    tmp_path: Path,
) -> None:
    module = _load_example_module("codex_provider_runtime_doc.py")

    response = """
        # Codex Provider Runtime Guide

        This base runtime explains how to write project documentation with Codex.
        It covers `ai-creds run codex app-server`, `CodexClientConfig`,
        `CodexClient`, `CodexSession`, thread continuity, approvals, and
        agent-to-agent decision flow.

        ## Warning

        High-level helpers should stay thin and should not hide approvals,
        thread continuity, or verification decisions.

        ## Project Setup

        ## Runtime Shape

        This runtime uses provider-selected subagents and keeps thread
        continuity across turns.

        ## Agent-to-Agent Communication

        ## Step-by-Step Verification

        This section describes step-by-step verification for the runtime.

        ## Approvals

        ## Provider Features

        The provider surface includes `tool/requestUserInput`,
        `DOCUMENTED_METHODS`, and `DOCUMENTED_NOTIFICATION_METHODS`.

        ## High-Level Helpers

        Use high-level helpers only after the provider-native runtime contract
        is understood.

        ## Output

        The runtime writes the generated Markdown to disk.
        """

    class FakeSession:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.thread_ids: list[str | None] = []

        def run_turn(
            self,
            input_items: list[dict[str, object]],
            *,
            thread_id: str | None = None,
        ) -> SimpleNamespace:
            self.prompts.append(str(input_items[0]["text"]))
            self.thread_ids.append(thread_id)
            return SimpleNamespace(
                output_text=response,
                thread=SimpleNamespace(thread_id="thr_runtime"),
                turn=SimpleNamespace(turn_id="turn_1", status="completed"),
            )

    fake_session = FakeSession()
    output_path = tmp_path / "docs" / "codex-provider.md"

    written_path = module.run_codex_provider_runtime_doc(
        project_name="demo-project",
        output_path=output_path,
        session=fake_session,
    )

    assert written_path == output_path
    source = output_path.read_text()
    assert "# Codex Provider Runtime Guide" in source
    assert "base runtime" in source
    assert "ai-creds run codex app-server" in source
    assert "CodexClientConfig" in source
    assert "CodexClient" in source
    assert "CodexSession" in source
    assert "subagent" in source
    assert "step-by-step verification" in source
    assert "thread continuity" in source
    assert "tool/requestUserInput" in source
    assert "DOCUMENTED_METHODS" in source
    assert "DOCUMENTED_NOTIFICATION_METHODS" in source
    assert "## Warning" in source
    assert "High-Level Helpers" in source
    assert len(fake_session.prompts) == 1
    assert fake_session.thread_ids == [None]
    prompt = fake_session.prompts[0].lower()
    assert "write the complete project documentation markdown" in prompt
    assert "decide which subagents should run" in prompt
    assert "write to this output path" in prompt
    assert "approval_handler" in fake_session.prompts[0]
    assert "Warning" in fake_session.prompts[0]
    assert "High-Level Helpers" in fake_session.prompts[0]
    assert "no extra modes" in prompt

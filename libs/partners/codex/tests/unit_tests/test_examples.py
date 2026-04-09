from __future__ import annotations

from pathlib import Path


def test_simple_invoke_example_uses_launch_command_tuple() -> None:
    example_path = Path(__file__).parents[5] / "examples" / "codex_simple_invoke.py"

    source = example_path.read_text()

    assert "from langchain_codex import ChatCodex" in source
    assert 'launch_command=("ai-creds", "run", "codex", "app-server")' in source
    assert "for chunk in model.stream(" in source


def test_readme_mentions_provider_native_client_and_wrapped_launch() -> None:
    readme_path = Path(__file__).parents[2] / "README.md"

    source = readme_path.read_text()

    assert "CodexClient" in source
    assert "CodexSession" in source
    assert "ai-creds run codex app-server" in source

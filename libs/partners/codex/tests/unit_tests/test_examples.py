from __future__ import annotations

from pathlib import Path


def test_simple_invoke_example_exists_and_uses_chat_codex() -> None:
    example_path = Path(__file__).parents[5] / "examples" / "codex_simple_invoke.py"

    assert example_path.exists()

    source = example_path.read_text()

    assert "from langchain_codex import ChatCodex" in source
    assert 'model = ChatCodex(model="gpt-5.4")' in source
    assert "response = model.invoke(" in source
    assert "print(response.content)" in source

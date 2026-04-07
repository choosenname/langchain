from __future__ import annotations

from pathlib import Path


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

    assert "from langchain_codex import ChatCodex" in source
    assert 'codex_command="ai-creds run codex"' in source
    assert "response = model.invoke(" in source
    assert 'raise SystemExit(f"Codex example failed: {err}") from err' in source
    assert "print(response.content)" in source
    assert 'name = "codex-simple-example"' in pyproject_source
    assert 'requires-python = ">=3.10,<3.14"' in pyproject_source
    assert '"langchain-codex>=0.1.0"' in pyproject_source
    assert 'langchain-codex = { path = "../libs/partners/codex", editable = true }' in (
        pyproject_source
    )
    assert 'requires-python = ">=3.10.0,<3.15"' in package_pyproject_source

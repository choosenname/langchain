"""Base Codex runtime example that asks Codex to write project documentation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from langchain_codex import CodexClient
from langchain_codex.protocol.events import DOCUMENTED_NOTIFICATION_METHODS
from langchain_codex.protocol.requests import DOCUMENTED_METHODS
from langchain_codex.types import (
    CodexApprovalDecision,
    CodexApprovalRequest,
    CodexClientConfig,
    JsonObject,
)

LAUNCH_COMMAND = ("ai-creds", "run", "codex", "app-server")
DEFAULT_OUTPUT_PATH = Path("docs/codex-provider-runtime.md")

ApprovalHandler = Callable[[CodexApprovalRequest], CodexApprovalDecision]


class SessionLike(Protocol):
    """Small protocol used by tests and custom runtimes."""

    def run_turn(
        self,
        input_items: Sequence[JsonObject],
        *,
        thread_id: str | None = None,
    ) -> object:
        """Run one Codex turn and return the provider response."""


def ask_human_approval(request: CodexApprovalRequest) -> CodexApprovalDecision:
    """Ask the terminal user before answering a Codex server request.

    Args:
        request: Provider-native server request, including approvals and user-input tools.

    Returns:
        The decision payload sent back to Codex app-server.
    """
    questions = request.params.get("questions")
    if request.method == "tool/requestUserInput" and isinstance(questions, list):
        return CodexApprovalDecision(
            decision={"answers": _ask_question_answers(questions)}
        )

    prompt = (
        f"Codex requested `{request.method}`"
        f" on thread `{request.thread_id or 'unknown'}`. Approve? [y/N] "
    )
    answer = input(prompt).strip().lower()
    decision = "accept" if answer in {"y", "yes"} else "cancel"
    return CodexApprovalDecision(decision=decision)


def build_codex_provider_runtime_prompt(
    *,
    project_name: str,
    output_path: Path,
) -> str:
    """Build the single prompt used by the docs runtime.

    Args:
        project_name: Human-readable project name to document.
        output_path: File path where this wrapper writes the returned Markdown.

    Returns:
        Prompt text for one provider-native Codex turn.
    """
    documented_methods = ", ".join(DOCUMENTED_METHODS)
    documented_notifications = ", ".join(DOCUMENTED_NOTIFICATION_METHODS)
    return f"""You are a Codex provider runtime writing docs for `{project_name}`.

Write the complete project documentation Markdown in one response.
Return only Markdown content. Do not wrap it in JSON or code fences.
The wrapper will write to this output path: `{output_path}`.

Runtime requirements:
- Use one simple run with no extra modes.
- Delegate decisions to the Codex provider wherever possible.
- Decide which subagents should run, what they should inspect, and how they report back.
- Explain agent-to-agent decisions and communication at a high level.
- Explain thread continuity: continue one thread across invokes when a thread id is supplied.
- Explain how to create a new thread when no thread id is supplied.
- Explain step-by-step verification before the documentation is considered complete.
- Explain human approval handling through `approval_handler`.
- Warn that high-level helpers must stay thin and must not hide provider-native behavior.

The document must include these sections:
- `# Codex Provider Runtime Guide`
- `## Warning`
- `## Project Setup`
- `## Runtime Shape`
- `## Agent-to-Agent Communication`
- `## Step-by-Step Verification`
- `## Approvals`
- `## Provider Features`
- `## High-Level Helpers`
- `## Output`

Provider-native API names that must appear:
- `ai-creds run codex app-server`
- `CodexClientConfig`
- `CodexClient`
- `CodexSession`
- `approval_handler`
- `tool/requestUserInput`
- `DOCUMENTED_METHODS`
- `DOCUMENTED_NOTIFICATION_METHODS`

Documented app-server methods available to reference:
{documented_methods}

Documented app-server notifications available to reference:
{documented_notifications}
"""


def run_codex_provider_runtime_doc(
    project_name: str = "current project",
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    session: SessionLike | None = None,
    approval_handler: ApprovalHandler | None = ask_human_approval,
    thread_id: str | None = None,
) -> Path:
    """Run Codex once and write the returned project documentation.

    Args:
        project_name: Human-readable project name to document.
        output_path: Markdown file path to create or replace.
        session: Optional prebuilt session for tests or a larger runtime.
        approval_handler: Optional handler for human approvals and user-input requests.
        thread_id: Existing thread id to continue. Leave unset to let Codex start a thread.

    Returns:
        Path to the written Markdown file.
    """
    client: CodexClient | None = None
    active_session = session
    if active_session is None:
        client = CodexClient(
            CodexClientConfig(
                launch_command=LAUNCH_COMMAND,
                model="gpt-5.4",
                approval_policy="on-request",
                sandbox_policy={"type": "workspace-write"},
                reasoning_effort="high",
                reasoning_summary="auto",
            )
        )
        active_session = client.create_session(
            approval_handler=approval_handler,
            thread_id=thread_id,
        )

    try:
        prompt = build_codex_provider_runtime_prompt(
            project_name=project_name,
            output_path=output_path,
        )
        response = active_session.run_turn(
            [{"type": "text", "text": prompt}],
            thread_id=thread_id,
        )
        markdown = _response_text(response)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        return output_path
    finally:
        if client is not None:
            client.close()


def _ask_question_answers(questions: list[object]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = question.get("id")
        question_text = question.get("question")
        if not isinstance(question_id, str) or not isinstance(question_text, str):
            continue
        answers[question_id] = input(f"{question_text} ")
    return answers


def _response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text.strip()
    return str(response).strip()


if __name__ == "__main__":
    written_path = run_codex_provider_runtime_doc()
    sys.stdout.write(f"{written_path}\n")

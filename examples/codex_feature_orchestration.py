"""Realistic single-file Codex orchestration for feature delivery in another repo."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_codex import ChatCodex

DEFAULT_MAX_SLICES = 4
MAX_REVIEW_ATTEMPTS = 3
PREVIEW_MAX_LINES = 8
PREVIEW_MAX_CHARS = 700
IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "__pycache__",
}


def build_repository_analysis_prompt(
    *,
    repo_path: str,
    feature_request: str,
    repo_snapshot: str,
    constraints: str,
) -> str:
    """Build the planner prompt for repository analysis.

    Args:
        repo_path: Absolute path to the target repository.
        feature_request: Requested feature work.
        repo_snapshot: Local repository snapshot gathered by the script.
        constraints: Additional user-supplied constraints.

    Returns:
        A prompt asking Codex to analyze the repository before execution.
    """
    return f"""You are the planner for a feature delivery workflow.

Target repository:
{repo_path}

Feature request:
{feature_request}

Repository snapshot:
{repo_snapshot}

Constraints:
{constraints}

Inspect the target repository directly when needed.
Do not assume missing facts.
Keep the output concise and practical.

Produce a Markdown report with these sections:

1. `Repository shape`
   - Summarize the most relevant packages, modules, or directories.
   - Call out where this feature most likely belongs.

2. `Implementation constraints`
   - List public API, testing, migration, or integration constraints.
   - Note anything that should remain unchanged unless explicitly approved.

3. `Execution risks`
   - Identify the main technical risks, unknowns, and likely review hotspots.

4. `Delegation guidance`
   - Explain which parts should stay with the main agent.
   - Explain which parts can be delegated safely.
"""


def build_execution_plan_prompt(
    *,
    repo_path: str,
    feature_request: str,
    repo_snapshot: str,
    analysis: str,
    constraints: str,
    max_slices: int,
) -> str:
    """Build the JSON execution-plan prompt.

    Args:
        repo_path: Absolute path to the target repository.
        feature_request: Requested feature work.
        repo_snapshot: Local repository snapshot gathered by the script.
        analysis: Planner analysis text.
        constraints: Additional user-supplied constraints.
        max_slices: Maximum number of work slices to produce.

    Returns:
        A prompt asking Codex for a machine-readable orchestration plan.
    """
    return f"""Turn the repository analysis into a realistic feature orchestration plan.

Target repository:
{repo_path}

Feature request:
{feature_request}

Repository snapshot:
{repo_snapshot}

Planner analysis:
{analysis}

Constraints:
{constraints}

Inspect the target repository directly when needed.
Return valid JSON only.
Use this schema:
{{
  "summary": "one paragraph",
  "open_questions": [
    {{
      "id": "question-1",
      "owner": "explorer",
      "question": "specific unresolved question",
      "paths": ["path/to/module.py"]
    }}
  ],
  "work_slices": [
    {{
      "id": "slice-1",
      "owner": "implementer",
      "goal": "one concrete feature slice",
      "paths": ["path/to/file.py"],
      "dependencies": [],
      "acceptance_criteria": ["reviewable outcome"],
      "verification": ["exact command to run"]
    }}
  ]
}}

Rules:
- Return at most {max_slices} work slices.
- Keep each slice independently reviewable.
- Use "owner": "implementer" for write scopes.
- Main-agent-only work must stay out of `work_slices`.
- Add `open_questions` only when a question blocks safe implementation.
"""


def build_explorer_prompt(
    *,
    repo_path: str,
    feature_request: str,
    repo_snapshot: str,
    question: dict[str, Any],
) -> str:
    """Build a narrow explorer prompt for one unresolved question.

    Args:
        repo_path: Absolute path to the target repository.
        feature_request: Requested feature work.
        repo_snapshot: Local repository snapshot gathered by the script.
        question: Structured explorer question from the plan.

    Returns:
        A prompt requesting a short, focused repository investigation.
    """
    return f"""Investigate one narrow implementation question in the target repository.

Target repository:
{repo_path}

Feature request:
{feature_request}

Repository snapshot:
{repo_snapshot}

Question:
{json.dumps(question, indent=2, sort_keys=True)}

Inspect the target repository directly.
Do not edit files.
Answer in 3-6 bullets with:
- the answer
- exact file references
- patterns to copy
- risks only if they affect implementation
"""


def build_slice_implementation_prompt(
    *,
    repo_path: str,
    feature_request: str,
    repo_snapshot: str,
    work_slice: dict[str, Any],
    explorer_findings: list[str],
    constraints: str,
) -> str:
    """Build the implementer prompt for one work slice.

    Args:
        repo_path: Absolute path to the target repository.
        feature_request: Requested feature work.
        repo_snapshot: Local repository snapshot gathered by the script.
        work_slice: Structured work slice from the plan.
        explorer_findings: Explorer answers relevant to implementation.
        constraints: Additional user-supplied constraints.

    Returns:
        A prompt that asks Codex to implement one bounded slice.
    """
    findings = "\n\n".join(explorer_findings) if explorer_findings else "No explorer findings."
    return f"""Implement one bounded feature slice in the target repository.

Target repository:
{repo_path}

Feature request:
{feature_request}

Repository snapshot:
{repo_snapshot}

Work slice:
{json.dumps(work_slice, indent=2, sort_keys=True)}

Explorer findings:
{findings}

Constraints:
{constraints}

Inspect the target repository directly.
Edit only the files needed for this slice.
Preserve public interfaces unless explicitly approved.
Add or update focused tests for any behavior change.
Do not revert unrelated local changes.

End with a brief Markdown summary that includes:
- changed files
- tests run
- remaining risks
"""


def build_slice_review_prompt(
    *,
    repo_path: str,
    work_slice: dict[str, Any],
    implementation_summary: str,
) -> str:
    """Build the reviewer prompt for one implementation attempt.

    Args:
        repo_path: Absolute path to the target repository.
        work_slice: Structured work slice from the plan.
        implementation_summary: Implementer summary for the slice.

    Returns:
        A prompt asking Codex for a strict JSON review result.
    """
    return f"""Review the latest implementation attempt for one feature slice.

Target repository:
{repo_path}

Work slice:
{json.dumps(work_slice, indent=2, sort_keys=True)}

Implementation summary:
{implementation_summary}

Inspect the target repository directly.
Return valid JSON only.
Use this schema:
{{
  "status": "ok",
  "summary": "why the slice is acceptable",
  "issues": []
}}

If the slice is not acceptable, return:
{{
  "status": "not_ok",
  "summary": "high-level reason",
  "issues": ["specific problem to fix", "another problem if needed"]
}}

Review rules:
- Only return "status": "ok" when the slice goal and acceptance criteria are satisfied.
- Call out missing tests, unsafe scope expansion, or public API risk.
"""


def build_integrator_prompt(
    *,
    repo_path: str,
    feature_request: str,
    analysis: str,
    plan: dict[str, Any],
    explorer_results: list[dict[str, str]],
    slice_results: list[dict[str, Any]],
    mode: str,
) -> str:
    """Build the final integrator prompt.

    Args:
        repo_path: Absolute path to the target repository.
        feature_request: Requested feature work.
        analysis: Planner analysis text.
        plan: Parsed JSON execution plan.
        explorer_results: Completed explorer outputs.
        slice_results: Completed implementer/reviewer outputs.
        mode: Selected orchestration mode.

    Returns:
        A prompt asking Codex for the final orchestration summary.
    """
    return f"""Produce the final orchestration summary for this feature run.

Mode:
{mode}

Target repository:
{repo_path}

Feature request:
{feature_request}

Planner analysis:
{analysis}

Execution plan:
{json.dumps(plan, indent=2, sort_keys=True)}

Explorer results:
{json.dumps(explorer_results, indent=2, sort_keys=True)}

Slice results:
{json.dumps(slice_results, indent=2, sort_keys=True)}

Write a concise Markdown report with these sections:
1. `Outcome`
2. `What changed`
3. `Verification`
4. `Remaining risks`

If mode is `plan`, describe the next implementation steps rather than claiming changes were made.
"""


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse a JSON object returned by Codex.

    Args:
        content: Raw model response expected to contain a JSON object.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the response is not a JSON object.
    """
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        msg = "Expected a JSON object response."
        raise ValueError(msg)
    return parsed


def _message_text(message: AIMessage) -> str:
    """Return the text view of a model response.

    Args:
        message: The model response message.

    Returns:
        The plain-text content extracted from the response.
    """
    return str(message.text)


def _truncate_preview(
    text: str,
    *,
    max_lines: int = PREVIEW_MAX_LINES,
    max_chars: int = PREVIEW_MAX_CHARS,
) -> str:
    """Return a compact preview for terminal display."""
    normalized = text.strip()
    if not normalized:
        return "(no output)"

    lines = normalized.splitlines()
    preview = "\n".join(lines[:max_lines]).strip()

    if len(preview) > max_chars:
        return f"{preview[: max_chars - 3].rstrip()}..."
    if len(lines) > max_lines:
        return f"{preview}\n..."
    return preview


def _print_preview(title: str, text: str) -> None:
    """Print a short labeled preview block."""
    print(f"[preview] {title}")
    print(_truncate_preview(text))
    print()


def _log_action(role: str, message: str) -> None:
    """Print a concise orchestration action line.

    Args:
        role: Orchestration role producing the message.
        message: Short action description.
    """
    print(f"[{role}] {message}")


def _summarize_slice_results(slice_results: list[dict[str, Any]]) -> str:
    """Return a compact summary for completed slice results.

    Args:
        slice_results: Per-slice orchestration results.

    Returns:
        A terminal-friendly multi-line summary.
    """
    lines: list[str] = []
    for result in slice_results:
        summary = f"{result['slice_id']}: {result['outcome']}"
        review_result = result.get("review_result")
        if isinstance(review_result, dict) and review_result.get("status"):
            summary += f" (review: {review_result['status']})"
        lines.append(summary)
    return "\n".join(lines)


def _read_constraints_file(constraints_file: str | None) -> str:
    """Read the optional constraints file.

    Args:
        constraints_file: Optional path to a text file with extra constraints.

    Returns:
        File contents or a fallback string.
    """
    if not constraints_file:
        return "No extra constraints."
    return Path(constraints_file).read_text(encoding="utf-8").strip() or "No extra constraints."


def _git_status_summary(repo_path: Path) -> str:
    """Return a compact git status summary if available.

    Args:
        repo_path: Repository root path.

    Returns:
        A short git status summary.
    """
    command = ["git", "status", "--short"]
    try:
        result = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "git status unavailable"

    output = result.stdout.strip()
    if not output:
        return "clean worktree"
    lines = output.splitlines()
    preview = "\n".join(lines[:10])
    if len(lines) > 10:
        return f"{preview}\n..."
    return preview


def _build_repo_snapshot(repo_path: Path) -> str:
    """Build a compact repository snapshot for prompting.

    Args:
        repo_path: Repository root path.

    Returns:
        A plain-text snapshot of the target repository.
    """
    files: list[str] = []
    for path in sorted(repo_path.rglob("*")):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.is_file():
            files.append(str(path.relative_to(repo_path)))
        if len(files) >= 40:
            break

    top_level = sorted(
        child.name + ("/" if child.is_dir() else "")
        for child in repo_path.iterdir()
        if child.name not in IGNORED_DIR_NAMES
    )
    snapshot_lines = [
        f"repo: {repo_path}",
        "top_level:",
        *[f"- {name}" for name in top_level[:20]],
        "files:",
        *[f"- {name}" for name in files],
        "git_status:",
        _git_status_summary(repo_path),
    ]
    return "\n".join(snapshot_lines)


def run_slice_review_loop(
    *,
    model: ChatCodex,
    repo_path: str,
    feature_request: str,
    repo_snapshot: str,
    work_slice: dict[str, Any],
    explorer_findings: list[str],
    constraints: str,
) -> dict[str, Any]:
    """Run implementer and reviewer turns for one slice.

    Args:
        model: ChatCodex instance used for orchestration.
        repo_path: Absolute path to the target repository.
        feature_request: Requested feature work.
        repo_snapshot: Local repository snapshot gathered by the script.
        work_slice: Structured work slice from the execution plan.
        explorer_findings: Explorer answers relevant to implementation.
        constraints: Additional user-supplied constraints.

    Returns:
        Slice result data including implementation and review summaries.
    """
    review_attempts = 0
    implementation_summary = ""
    review_result: dict[str, Any] = {
        "status": "not_ok",
        "summary": "Implementation not attempted.",
        "issues": [],
    }

    while review_attempts < MAX_REVIEW_ATTEMPTS:
        _log_action("implementer", f"Running implementer for {work_slice['id']}")
        implementation = model.invoke(
            build_slice_implementation_prompt(
                repo_path=repo_path,
                feature_request=feature_request,
                repo_snapshot=repo_snapshot,
                work_slice=work_slice,
                explorer_findings=explorer_findings,
                constraints=constraints,
            )
        )
        implementation_summary = _message_text(implementation)

        _log_action("reviewer", f"Running reviewer for {work_slice['id']}")
        review_message = model.invoke(
            build_slice_review_prompt(
                repo_path=repo_path,
                work_slice=work_slice,
                implementation_summary=implementation_summary,
            )
        )
        review_result = _parse_json_response(_message_text(review_message))
        if review_result.get("status") == "ok":
            return {
                "slice_id": work_slice["id"],
                "outcome": "ok",
                "implementation_summary": implementation_summary,
                "review_result": review_result,
            }
        review_attempts += 1

    return {
        "slice_id": work_slice["id"],
        "outcome": "not_ok",
        "implementation_summary": implementation_summary,
        "review_result": review_result,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the orchestration example.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a realistic Codex feature orchestration loop."
    )
    parser.add_argument("--repo", required=True, help="Path to the target repository.")
    parser.add_argument(
        "--feature",
        required=True,
        help="Feature request for the target repository.",
    )
    parser.add_argument(
        "--mode",
        default="plan",
        choices=("plan", "implement"),
        help="Whether to stop at planning or continue through implementation prompts.",
    )
    parser.add_argument(
        "--constraints-file",
        help="Optional path to a text file with extra implementation constraints.",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=DEFAULT_MAX_SLICES,
        help="Maximum number of implementation slices to request from the planner.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Codex model name to use for all orchestration roles.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feature orchestration example."""
    args = parse_args()
    repo_path = Path(args.repo).expanduser().resolve()
    if not repo_path.exists():
        raise SystemExit(f"Target repository does not exist: {repo_path}")

    constraints = _read_constraints_file(args.constraints_file)
    repo_snapshot = _build_repo_snapshot(repo_path)
    model = ChatCodex(
        model=args.model,
        launch_command=("ai-creds", "run", "codex", "app-server"),
        approval_policy="never",
        sandbox_policy={"type": "dangerFullAccess"},
    )

    _log_action("planner", "Starting repository analysis")
    planner = model.invoke(
        build_repository_analysis_prompt(
            repo_path=str(repo_path),
            feature_request=args.feature,
            repo_snapshot=repo_snapshot,
            constraints=constraints,
        )
    )
    planner_text = _message_text(planner)
    _print_preview("Repository analysis", planner_text)

    _log_action("planner", "Building execution plan")
    execution_plan = model.invoke(
        build_execution_plan_prompt(
            repo_path=str(repo_path),
            feature_request=args.feature,
            repo_snapshot=repo_snapshot,
            analysis=planner_text,
            constraints=constraints,
            max_slices=args.max_slices,
        )
    )
    plan = _parse_json_response(_message_text(execution_plan))
    _print_preview("Execution plan", json.dumps(plan, indent=2, sort_keys=True))

    explorer_results: list[dict[str, str]] = []
    for question in plan.get("open_questions", []):
        _log_action("explorer", f"Investigating {question['id']}")
        explorer = model.invoke(
            build_explorer_prompt(
                repo_path=str(repo_path),
                feature_request=args.feature,
                repo_snapshot=repo_snapshot,
                question=question,
            )
        )
        explorer_results.append(
            {
                "question_id": question["id"],
                "summary": _message_text(explorer),
            }
        )

    slice_results: list[dict[str, Any]] = []
    if args.mode == "implement":
        for work_slice in plan.get("work_slices", []):
            slice_result = run_slice_review_loop(
                model=model,
                repo_path=str(repo_path),
                feature_request=args.feature,
                repo_snapshot=repo_snapshot,
                work_slice=work_slice,
                explorer_findings=[result["summary"] for result in explorer_results],
                constraints=constraints,
            )
            slice_results.append(slice_result)

    _log_action("integrator", "Preparing final orchestration summary")
    integrator = model.invoke(
        build_integrator_prompt(
            repo_path=str(repo_path),
            feature_request=args.feature,
            analysis=planner_text,
            plan=plan,
            explorer_results=explorer_results,
            slice_results=slice_results,
            mode=args.mode,
        )
    )
    integration_summary = _message_text(integrator)

    print("# Repository Analysis")
    print()
    print(_truncate_preview(planner_text))
    print()
    print("# Execution Plan")
    print()
    print(_truncate_preview(json.dumps(plan, indent=2, sort_keys=True)))
    print()
    if explorer_results:
        print("# Explorer Findings")
        print()
        for result in explorer_results:
            print(f"## {result['question_id']}")
            print()
            print(_truncate_preview(result["summary"]))
            print()
    if slice_results:
        print("# Slice Results")
        print()
        print(_summarize_slice_results(slice_results))
        print()
    print("# Feature Orchestration Summary")
    print()
    print(_truncate_preview(integration_summary))


if __name__ == "__main__":
    main()

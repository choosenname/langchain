"""Codex workflow example for planning, implementing, and reviewing refactors."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_codex.chat_models import ChatCodex

MAX_REVIEW_ATTEMPTS = 3
PREVIEW_MAX_LINES = 8
PREVIEW_MAX_CHARS = 700


def build_repository_analysis_prompt() -> str:
    """Build the first prompt for a repository structure review.

    Returns:
        A Markdown-oriented prompt that asks Codex to map the project, explain
        refactor rationale, and identify review constraints before editing code.
    """
    return """You are reviewing a Python monorepo and preparing it for refactoring.

Produce a concise Markdown report with these sections:

1. `Project structure`
   - Summarize the major packages and what each appears to do.
   - Call out the most important files or directories a new contributor should read first.

2. `Refactor opportunities`
   - List the top 5 refactors that would improve clarity, maintainability, or testability.
   - Group them by effort: low, medium, or high.
   - Prefer safe, incremental changes over large rewrites.
   - For each item, include a short rationale and the likely files or packages impacted.

3. `Review constraints`
   - List the conditions that must be true before editing code.
   - Separate must-have constraints from nice-to-have ones.
   - Include anything needed for running tests, updating docs, or reviewing risky refactors.

4. `Documentation plan`
   - Propose the docs pages or README sections that would help someone understand the repo.
   - Include the intended audience for each page.

5. `First implementation steps`
   - Give the next 3 concrete tasks in priority order.
   - Make each task small enough to finish in one focused PR.

Keep the answer practical, specific, and grounded in what you can infer from the codebase.
If information is missing, state the assumption clearly instead of guessing silently.
"""


def build_execution_plan_prompt() -> str:
    """Build the follow-up prompt for turning analysis into an execution plan.

    Returns:
        A prompt that asks Codex to prioritize the earlier findings into phases
        with refactoring constraints and risk in mind.
    """
    return """Turn the previous repository analysis into a phased execution plan.

Focus on refactoring work that can be done safely after the required review constraints are in place.

Output a Markdown table with these columns:

- `phase`
- `goal`
- `scope`
- `risk`

Then add a short checklist for the first phase only.
Keep the plan optimized for documentation work and incremental refactoring.
Call out any phase that should wait for additional review or prerequisites.
"""


def build_step_list_prompt() -> str:
    """Ask Codex for a machine-readable list of implementation steps.

    Returns:
        A prompt that requests a JSON step list suitable for orchestration.
    """
    return """Using the previous analysis and execution plan, produce a step list for implementation.

Return valid JSON only.
Use this schema:
{
  "steps": [
    {
      "id": "step-1",
      "title": "short title",
      "goal": "one concrete implementation goal",
      "files": ["path/to/file.py"],
      "acceptance_criteria": ["specific reviewable outcome"]
    }
  ]
}

Rules:
- Return between 1 and 5 steps.
- Each step must be small enough to implement and review independently.
- Prefer safe refactors and documentation improvements over large rewrites.
"""


def build_step_implementation_prompt(
    step: dict[str, Any],
    review_history: list[dict[str, Any]],
) -> str:
    """Build the implementer prompt for a single step.

    Args:
        step: Structured step data from the step plan.
        review_history: Prior review outcomes for the same step.

    Returns:
        A prompt that asks Codex to implement or fix one step.
    """
    review_feedback = "No reviewer feedback yet."
    if review_history:
        review_feedback = json.dumps(review_history[-1], indent=2, sort_keys=True)

    return f"""Implement this refactor step in the current repository.

Step:
{json.dumps(step, indent=2, sort_keys=True)}

Most recent reviewer feedback:
{review_feedback}

Instructions:
- Make the smallest change that satisfies the step goal.
- If reviewer feedback is present, fix the problems it identifies before doing anything else.
- Keep the work scoped to this step.
- End with a brief implementation summary that mentions changed files and any remaining risks.
"""


def build_step_review_prompt(step: dict[str, Any], implementation_summary: str) -> str:
    """Build the reviewer prompt for a single implementation attempt.

    Args:
        step: Structured step data from the step plan.
        implementation_summary: The implementer model's summary of the attempt.

    Returns:
        A prompt that requests a strict JSON review result.
    """
    return f"""Review the latest implementation attempt for this step.

Step:
{json.dumps(step, indent=2, sort_keys=True)}

Implementation summary:
{implementation_summary}

Return valid JSON only.
Use this schema:
{{
  "status": "ok",
  "summary": "why the step is acceptable",
  "issues": []
}}

If the step is not acceptable, return:
{{
  "status": "not_ok",
  "summary": "high-level reason",
  "issues": ["specific problem to fix", "another problem if needed"]
}}

Review rules:
- Only return "status": "ok" when the step goal and acceptance criteria are satisfied.
- If not acceptable, explain what is wrong in `issues`.
- Keep the review specific and action-oriented.
"""


def build_judge_prompt(
    step: dict[str, Any],
    implementation_history: list[str],
    review_history: list[dict[str, Any]],
) -> str:
    """Build the judge prompt after repeated failed reviews.

    Args:
        step: Structured step data from the step plan.
        implementation_history: Summaries from implementation attempts.
        review_history: Structured reviewer outputs for the step.

    Returns:
        A prompt that asks for one of three terminal decisions.
    """
    return f"""Three review attempts failed for the same step.

Step:
{json.dumps(step, indent=2, sort_keys=True)}

Implementation history:
{json.dumps(implementation_history, indent=2)}

Review history:
{json.dumps(review_history, indent=2, sort_keys=True)}

Return valid JSON only.
Use this schema:
{{
  "decision": "continue_fix",
  "reason": "why another fix attempt is justified"
}}

Allowed decisions:
- "decision": "continue_fix"
- "decision": "continue_next_step"
- "decision": "cancel_all"

Choose exactly one decision.
"""


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse a JSON response from Codex.

    Args:
        content: Raw model response that should contain one JSON object.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: If the content is not a JSON object.
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


def _summarize_step_results(step_results: list[dict[str, Any]]) -> str:
    """Return a compact step-result summary."""
    if not step_results:
        return "No step results."

    lines: list[str] = []
    for result in step_results:
        step_id = str(result.get("step_id", "unknown-step"))
        outcome = str(result.get("outcome", "unknown"))
        line = f"{step_id}: {outcome}"
        judge_result = result.get("judge_result")
        if isinstance(judge_result, dict):
            decision = judge_result.get("decision")
            if isinstance(decision, str):
                line = f"{line} (judge: {decision})"
        lines.append(line)
    return "\n".join(lines)


def run_step_review_loop(model: ChatCodex, step: dict[str, Any]) -> dict[str, Any]:
    """Run the implement-review-judge loop for one step.

    Args:
        model: ChatCodex instance with a persistent thread.
        step: Step metadata from the generated step plan.

    Returns:
        Structured result for the step outcome.
    """
    implementation_history: list[str] = []
    review_history: list[dict[str, Any]] = []

    while True:
        review_attempts = 0
        while review_attempts < MAX_REVIEW_ATTEMPTS:
            implementation_summary = _message_text(
                model.invoke(build_step_implementation_prompt(step, review_history))
            )
            implementation_history.append(implementation_summary)
            _print_preview(f"{step['id']} implementer", implementation_summary)

            review_result = _parse_json_response(
                _message_text(
                    model.invoke(build_step_review_prompt(step, implementation_summary))
                )
            )
            review_history.append(review_result)
            _print_preview(
                f"{step['id']} reviewer",
                json.dumps(review_result, indent=2, sort_keys=True),
            )

            if review_result.get("status") == "ok":
                return {
                    "step_id": step["id"],
                    "outcome": "ok",
                    "implementation_history": implementation_history,
                    "review_history": review_history,
                }

            review_attempts += 1

        judge_result = _parse_json_response(
            _message_text(
                model.invoke(
                    build_judge_prompt(step, implementation_history, review_history)
                )
            )
        )
        _print_preview(
            f"{step['id']} judge",
            json.dumps(judge_result, indent=2, sort_keys=True),
        )
        decision = judge_result.get("decision")

        if decision == "continue_fix":
            review_history.append(
                {
                    "status": "not_ok",
                    "summary": "Judge requested another fix cycle.",
                    "issues": [judge_result.get("reason", "No reason provided.")],
                }
            )
            continue

        if decision == "continue_next_step":
            return {
                "step_id": step["id"],
                "outcome": "continue_next_step",
                "judge_result": judge_result,
                "implementation_history": implementation_history,
                "review_history": review_history,
            }

        if decision == "cancel_all":
            return {
                "step_id": step["id"],
                "outcome": "cancel_all",
                "judge_result": judge_result,
                "implementation_history": implementation_history,
                "review_history": review_history,
            }

        msg = f"Unexpected judge decision: {decision!r}"
        raise ValueError(msg)


def main() -> None:
    """Run a multi-turn Codex workflow with per-step review controls."""
    model = ChatCodex(
        model="gpt-5.4",
        codex_command="ai-creds run codex",
        approval_policy="never",
        sandbox="danger-full-access",
    )
    try:
        analysis = model.invoke(build_repository_analysis_prompt())
        _print_preview("analysis", _message_text(analysis))
        execution_plan = model.invoke(build_execution_plan_prompt())
        _print_preview("execution plan", _message_text(execution_plan))
        step_plan = _parse_json_response(_message_text(model.invoke(build_step_list_prompt())))
        _print_preview("step plan", json.dumps(step_plan, indent=2, sort_keys=True))

        step_results: list[dict[str, Any]] = []
        for step in step_plan["steps"]:
            result = run_step_review_loop(model, step)
            step_results.append(result)
            if result["outcome"] == "cancel_all":
                break
    except (RuntimeError, ValueError, KeyError, TypeError) as err:
        raise SystemExit(f"Codex example failed: {err}") from err

    print("# Repository Structuring Report")
    print()
    print(_truncate_preview(_message_text(analysis)))
    print()
    print("# Phased Execution Plan")
    print()
    print(_truncate_preview(_message_text(execution_plan)))
    print()
    print("# Step Execution Results")
    print()
    print(_summarize_step_results(step_results))


if __name__ == "__main__":
    main()

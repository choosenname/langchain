"""Minimal synchronous example for `ChatCodex`."""

from langchain_codex import ChatCodex

LIVE_PREVIEW_MAX_CHARS = 120
FINAL_PREVIEW_MAX_LINES = 6
FINAL_PREVIEW_MAX_CHARS = 300


def _truncate_preview(
    text: str,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    """Return a compact preview of streamed or final text."""
    normalized = text.strip()
    if not normalized:
        return "(waiting for output)"

    lines = normalized.splitlines()
    preview = "\n".join(lines[:max_lines]).strip()

    if len(preview) > max_chars:
        return f"{preview[: max_chars - 3].rstrip()}..."
    if len(lines) > max_lines:
        return f"{preview}\n..."
    return preview


def _print_live_preview(text: str) -> None:
    """Render a bounded one-line live preview in place."""
    preview = _truncate_preview(text, max_lines=1, max_chars=LIVE_PREVIEW_MAX_CHARS)
    print(f"\r[assistant] {preview:<{LIVE_PREVIEW_MAX_CHARS}}", end="", flush=True)


def main() -> None:
    """Run one synchronous Codex prompt and print the reply."""
    model = ChatCodex(
        model="gpt-5.4",
        launch_command=("ai-creds", "run", "codex", "app-server"),
        approval_policy="never",
        sandbox_policy={"type": "dangerFullAccess"},
    )
    try:
        chunks: list[str] = []
        for chunk in model.stream("Summarize this repository in one sentence."):
            if chunk.text:
                chunks.append(chunk.text)
                _print_live_preview("".join(chunks))
    except RuntimeError as err:
        raise SystemExit(f"Codex example failed: {err}") from err

    response_text = "".join(chunks)
    print()
    print()
    print("# Final Reply")
    print()
    print(
        _truncate_preview(
            response_text,
            max_lines=FINAL_PREVIEW_MAX_LINES,
            max_chars=FINAL_PREVIEW_MAX_CHARS,
        )
    )


if __name__ == "__main__":
    main()

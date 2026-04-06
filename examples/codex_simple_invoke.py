"""Minimal synchronous example for `ChatCodex`."""

from langchain_codex import ChatCodex


def main() -> None:
    """Run one synchronous Codex prompt and print the reply."""
    model = ChatCodex(
        model="gpt-5.4",
        codex_command="ai-creds run codex",
    )
    response = model.invoke("Summarize this repository in one sentence.")
    print(response.content)


if __name__ == "__main__":
    main()

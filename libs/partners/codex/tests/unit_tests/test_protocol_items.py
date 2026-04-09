from __future__ import annotations

from langchain_codex.protocol import items as protocol_items


def test_input_item_parser_supports_documented_input_unions() -> None:
    text_item = protocol_items.parse_input_item({"type": "text", "text": "hello"})
    image_item = protocol_items.parse_input_item(
        {"type": "image", "url": "https://example.com/test.png"}
    )
    local_image_item = protocol_items.parse_input_item(
        {"type": "localImage", "path": "/workspace/test.png"}
    )
    mention_item = protocol_items.parse_input_item(
        {"type": "mention", "name": "Demo App", "path": "app://demo-app"}
    )
    skill_item = protocol_items.parse_input_item(
        {"type": "skill", "name": "skill-creator", "path": "/workspace/SKILL.md"}
    )

    assert text_item.kind == "text"
    assert image_item.kind == "image"
    assert local_image_item.kind == "localImage"
    assert mention_item.kind == "mention"
    assert skill_item.kind == "skill"


def test_item_parser_supports_documented_thread_item_unions() -> None:
    parsed_items = [
        protocol_items.parse_thread_item({"id": "msg_1", "type": "agentMessage", "text": "done"}),
        protocol_items.parse_thread_item(
            {
                "id": "reason_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "content": [],
            }
        ),
        protocol_items.parse_thread_item(
            {
                "id": "cmd_1",
                "type": "commandExecution",
                "command": ["pytest"],
                "cwd": "/repo",
                "status": "completed",
            }
        ),
        protocol_items.parse_thread_item(
            {
                "id": "file_1",
                "type": "fileChange",
                "changes": [{"path": "README.md", "kind": "update", "diff": "@@"}],
                "status": "completed",
            }
        ),
        protocol_items.parse_thread_item(
            {
                "id": "mcp_1",
                "type": "mcpToolCall",
                "server": "docs",
                "tool": "search",
                "status": "completed",
                "arguments": {"q": "codex"},
            }
        ),
        protocol_items.parse_thread_item(
            {"id": "web_1", "type": "webSearch", "query": "codex docs"}
        ),
        protocol_items.parse_thread_item(
            {"id": "img_1", "type": "imageView", "path": "/workspace/screenshot.png"}
        ),
        protocol_items.parse_thread_item(
            {"id": "review_1", "type": "enteredReviewMode", "review": "current changes"}
        ),
        protocol_items.parse_thread_item({"id": "compact_1", "type": "contextCompaction"}),
    ]

    assert [item.kind for item in parsed_items] == [
        "agentMessage",
        "reasoning",
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "webSearch",
        "imageView",
        "enteredReviewMode",
        "contextCompaction",
    ]


def test_unknown_item_parser_preserves_raw_payload_for_forward_compatibility() -> None:
    parsed = protocol_items.parse_thread_item({"id": "custom_1", "type": "futureItem", "value": 1})

    assert parsed.kind == "futureItem"
    assert parsed.raw == {"id": "custom_1", "type": "futureItem", "value": 1}

# langchain-codex

`langchain-codex` provides a LangChain chat model wrapper for a local `codex app-server`
process.

## Requirements

- The `codex` CLI must be installed and available on `PATH`.
- This integration launches `codex app-server` locally over `stdio`.

## Usage

```python
from langchain_codex import ChatCodex

model = ChatCodex(model="gpt-5.4")

response = model.invoke("Summarize this repository.")
print(response.content)
```

You can also use the built-in provider registry:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("codex:gpt-5.4")
```

## Behavior

- One `ChatCodex` instance keeps one long-lived local `codex app-server` subprocess.
- The same instance reuses one Codex thread across calls.
- Supported v1 methods: `invoke`, `ainvoke`, `stream`, and `astream`.
- Streaming is built from `item/agentMessage/delta` text updates and ends with the
  authoritative completed-turn metadata.

## Current Limits

- Local `stdio` transport only
- No WebSocket transport
- No approval callback plumbing
- No app/tool passthrough beyond plain text conversation input

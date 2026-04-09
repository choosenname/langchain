# langchain-codex

`langchain-codex` provides both a provider-native Codex client/session API and a
LangChain `ChatCodex` adapter for a local `codex app-server` process.

## Requirements

- The `codex` CLI must be installed and available on `PATH`.
- This integration launches `codex app-server` locally over `stdio`.
- Wrapped launch commands such as `ai-creds run codex app-server` are supported
  through argv-style `launch_command` configuration.

## Usage

```python
from langchain_codex import ChatCodex, CodexClient
from langchain_codex.types import CodexClientConfig
client = CodexClient(
    config=CodexClientConfig(
        launch_command=("codex", "app-server"),
        model="gpt-5.4",
    )
)
session = client.create_session()

result = session.run_turn([{"type": "text", "text": "Summarize this repository."}])
print(result.output_text)

model = ChatCodex(model="gpt-5.4", client=client)

response = model.invoke("Summarize this repository.")
print(response.content)
```

If your environment requires a command prefix, pass `launch_command`:

```python
from langchain_codex import ChatCodex

model = ChatCodex(
    model="gpt-5.4",
    launch_command=("ai-creds", "run", "codex", "app-server"),
)

response = model.invoke("Summarize this repository.")
print(response.content)
```

You can also use the built-in provider registry:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("codex:gpt-5.4")
```

## Provider-Native API

Use `CodexClient` and `CodexSession` when you need explicit thread reuse,
blocking approval handling, or direct app-server method access:

```python
from langchain_codex import CodexClient
from langchain_codex.types import CodexClientConfig

client = CodexClient(
    CodexClientConfig(
        launch_command=("ai-creds", "run", "codex", "app-server"),
        model="gpt-5.4",
        approval_policy="on-request",
    )
)
session = client.create_session()
thread = session.start_thread()
result = session.run_turn(
    [{"type": "text", "text": "Continue on this thread."}],
    thread_id=thread.thread_id,
)
print(result.thread.thread_id, result.turn.turn_id, result.output_text)
```

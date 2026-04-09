# Codex Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `langchain-codex` as a documented-protocol Codex provider with full app-server API and event coverage, wrapped-launch support such as `ai-creds run codex app-server`, synchronous human approvals, explicit in-process thread continuity, and a thin LangChain `ChatCodex` adapter.

**Architecture:** Replace the current narrow chat-only stack with a layered provider: `CodexClient` owns process and connection state, `CodexSession` owns thread and turn workflows, protocol modules own request/event/item parsing, and `ChatCodex` adapts provider-native execution to LangChain. The implementation must mirror the documented app-server surface exactly, including documented experimental methods and notifications behind capability gates, while keeping one transport only: local `stdio` to `codex app-server`.

**Tech Stack:** Python, `langchain-core`, `pytest`, `ruff`, `pyright`, `uv`

---

## Planned File Structure

Run all commands from `libs/partners/codex` unless a step says otherwise.

### Create

- `langchain_codex/client.py`
- `langchain_codex/types.py`
- `langchain_codex/observers.py`
- `langchain_codex/protocol/__init__.py`
- `langchain_codex/protocol/base.py`
- `langchain_codex/protocol/requests.py`
- `langchain_codex/protocol/events.py`
- `langchain_codex/protocol/items.py`
- `langchain_codex/transport/__init__.py`
- `langchain_codex/transport/base.py`
- `langchain_codex/transport/stdio.py`
- `tests/unit_tests/test_client.py`
- `tests/unit_tests/test_protocol_requests.py`
- `tests/unit_tests/test_protocol_events.py`
- `tests/unit_tests/test_protocol_items.py`
- `tests/unit_tests/test_server_requests.py`
- `tests/unit_tests/test_filesystem.py`
- `tests/unit_tests/test_realtime.py`

### Replace or heavily rewrite

- `langchain_codex/__init__.py`
- `langchain_codex/chat_models.py`
- `langchain_codex/session.py`
- `langchain_codex/errors.py`
- `tests/unit_tests/test_chat_models.py`
- `tests/unit_tests/test_examples.py`
- `tests/unit_tests/test_imports.py`

### Delete after replacement is complete

- `langchain_codex/_types.py`
- `langchain_codex/transport.py`
- `tests/unit_tests/test_transport.py`
- `tests/unit_tests/test_types.py`

## Guardrails

- Match the documented app-server methods and notifications exactly. If a name in code differs from the docs, the docs win.
- Keep wrapped launch support argv-based. Do not introduce shell execution to support `ai-creds run codex app-server`.
- Do not reintroduce compatibility shims for the old package shape.
- Keep documented experimental surfaces behind capability checks rather than omitting them.
- Do not commit `.venv`, `dist`, `.pytest_cache`, `.ruff_cache`, `__pycache__`, `.ropeproject`, or similar artifacts from `libs/partners/codex`.

### Task 1: Lock down the documented protocol surface with failing tests

**Files:**
- Create: `tests/unit_tests/test_protocol_requests.py`
- Create: `tests/unit_tests/test_protocol_events.py`
- Create: `tests/unit_tests/test_protocol_items.py`
- Modify: `tests/unit_tests/test_examples.py`

- [ ] **Step 1: Write the failing request coverage tests**

Add tests that assert the provider can build or raw-forward the documented request families:

- initialization and capability requests
- thread lifecycle methods
- turn lifecycle methods
- realtime methods
- review, command, filesystem, plugin, skills, app, MCP, config, auth, and feedback methods
- wrapped launch command parsing for both `["codex", "app-server"]` and `["ai-creds", "run", "codex", "app-server"]`

- [ ] **Step 2: Write the failing event and item coverage tests**

Add table-driven tests that assert typed parsing for:

- thread, turn, item, auth, MCP, filesystem, search, app, skills, command, and realtime notifications
- documented item unions such as `agentMessage`, `reasoning`, `commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`, `imageView`, `enteredReviewMode`, and `contextCompaction`
- documented input unions such as `text`, `image`, `localImage`, `mention`, and `skill`

- [ ] **Step 3: Run the new protocol tests to confirm failure**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_protocol_requests.py tests/unit_tests/test_protocol_events.py tests/unit_tests/test_protocol_items.py -q
```
Expected: FAIL because the new protocol modules and wrappers do not exist yet.

- [ ] **Step 4: Commit the baseline**

```bash
git add tests/unit_tests/test_protocol_requests.py tests/unit_tests/test_protocol_events.py tests/unit_tests/test_protocol_items.py tests/unit_tests/test_examples.py
git commit -m "test(codex): lock protocol coverage"
```

### Task 2: Build the shared type system and error model

**Files:**
- Create: `langchain_codex/types.py`
- Modify: `langchain_codex/errors.py`
- Create: `tests/unit_tests/test_client.py`
- Modify: `tests/unit_tests/test_protocol_items.py`

- [ ] **Step 1: Write the failing type and import tests**

Add tests that expect public exports for the provider-native surface:

```python
from langchain_codex import ChatCodex, CodexClient, CodexSession
from langchain_codex.types import CodexClientConfig, CodexThreadHandle
```

Add failing assertions for:

- launch config storing argv tokens without shell semantics
- typed approval request and response models
- typed thread and turn result metadata

- [ ] **Step 2: Run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_client.py tests/unit_tests/test_protocol_items.py tests/unit_tests/test_imports.py -q
```
Expected: FAIL because these public symbols do not exist yet.

- [ ] **Step 3: Implement the type layer**

Add concrete dataclasses, enums, and typed aliases for at least:

```python
@dataclass(frozen=True)
class CodexClientConfig:
    launch_command: tuple[str, ...]
    model: str | None = None
    cwd: str | None = None
    approval_policy: str | None = None
    sandbox: object | None = None
    experimental_api: bool = False
    opt_out_notification_methods: tuple[str, ...] = ()
```

```python
@dataclass(frozen=True)
class CodexThreadHandle:
    thread_id: str
    name: str | None
    status: object | None
    ephemeral: bool | None
```

Also define provider-facing request, response, item, and event models that the
later tasks can reuse.

- [ ] **Step 4: Re-run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_client.py tests/unit_tests/test_protocol_items.py tests/unit_tests/test_imports.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/types.py langchain_codex/errors.py tests/unit_tests/test_client.py tests/unit_tests/test_protocol_items.py tests/unit_tests/test_imports.py
git commit -m "feat(codex): add provider types"
```

### Task 3: Implement protocol request builders and item-event parsers

**Files:**
- Create: `langchain_codex/protocol/__init__.py`
- Create: `langchain_codex/protocol/base.py`
- Create: `langchain_codex/protocol/requests.py`
- Create: `langchain_codex/protocol/events.py`
- Create: `langchain_codex/protocol/items.py`
- Modify: `tests/unit_tests/test_protocol_requests.py`
- Modify: `tests/unit_tests/test_protocol_events.py`
- Modify: `tests/unit_tests/test_protocol_items.py`

- [ ] **Step 1: Implement the request builder API**

Add request builders for each documented method family. Use exact wire names and
field names:

```python
def build_initialize_request(config: CodexClientConfig) -> JsonObject: ...
def build_thread_start_params(...) -> JsonObject: ...
def build_turn_start_params(...) -> JsonObject: ...
def build_command_exec_params(...) -> JsonObject: ...
def build_fs_watch_params(...) -> JsonObject: ...
```

- [ ] **Step 2: Implement item parsing**

Add discriminated parsing for documented input and item unions. Preserve raw
payloads for additive forward compatibility.

- [ ] **Step 3: Implement event parsing**

Add notification parsers for all documented event families, including realtime,
filesystem, app updates, rate limit updates, and `serverRequest/resolved`.

- [ ] **Step 4: Run the protocol tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_protocol_requests.py tests/unit_tests/test_protocol_events.py tests/unit_tests/test_protocol_items.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/protocol/__init__.py langchain_codex/protocol/base.py langchain_codex/protocol/requests.py langchain_codex/protocol/events.py langchain_codex/protocol/items.py tests/unit_tests/test_protocol_requests.py tests/unit_tests/test_protocol_events.py tests/unit_tests/test_protocol_items.py
git commit -m "feat(codex): add protocol models"
```

### Task 4: Implement the transport and connection layer

**Files:**
- Create: `langchain_codex/transport/__init__.py`
- Create: `langchain_codex/transport/base.py`
- Create: `langchain_codex/transport/stdio.py`
- Create: `langchain_codex/client.py`
- Modify: `tests/unit_tests/test_client.py`
- Create: `tests/unit_tests/test_server_requests.py`
- Delete: `langchain_codex/transport.py`
- Delete: `tests/unit_tests/test_transport.py`

- [ ] **Step 1: Write the failing transport and launch tests**

Add tests for:

- request-response matching and notification dispatch
- server-initiated JSON-RPC request handling
- stderr diagnostics capture
- launch argv validation for direct and wrapped commands
- process startup using `codex app-server` and `ai-creds run codex app-server`

- [ ] **Step 2: Run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_client.py tests/unit_tests/test_server_requests.py -q
```
Expected: FAIL because the new transport package and server-request plumbing are not complete.

- [ ] **Step 3: Implement the transport base and stdio transport**

Create a transport abstraction with:

```python
class CodexTransport(Protocol):
    def request(self, method: str, params: JsonObject) -> JsonObject: ...
    def notify(self, method: str, params: JsonObject) -> None: ...
    def add_notification_handler(self, handler: NotificationHandler) -> Callable[[], None]: ...
    def add_server_request_handler(self, handler: ServerRequestHandler) -> Callable[[], None]: ...
```

Implement the stdio transport in `transport/stdio.py` and move process I/O out
of the old flat `transport.py`.

- [ ] **Step 4: Implement `CodexClient` connection lifecycle**

Implement:

- launch-command normalization
- `initialize` and `initialized`
- raw `request()` and `notify()`
- `create_session()`
- `close()`
- context-manager support

- [ ] **Step 5: Re-run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_client.py tests/unit_tests/test_server_requests.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add langchain_codex/transport/__init__.py langchain_codex/transport/base.py langchain_codex/transport/stdio.py langchain_codex/client.py tests/unit_tests/test_client.py tests/unit_tests/test_server_requests.py
git rm langchain_codex/transport.py tests/unit_tests/test_transport.py
git commit -m "feat(codex): add connection layer"
```

### Task 5: Implement server-request handling and blocking approvals

**Files:**
- Create: `langchain_codex/observers.py`
- Modify: `langchain_codex/client.py`
- Modify: `langchain_codex/session.py`
- Modify: `tests/unit_tests/test_server_requests.py`
- Create: `tests/unit_tests/test_session.py`

- [ ] **Step 1: Write the failing approval-flow tests**

Add tests for:

- `item/commandExecution/requestApproval`
- `item/fileChange/requestApproval`
- `tool/requestUserInput`
- `account/chatgptAuthTokens/refresh`
- `serverRequest/resolved`

Each test should verify that the provider blocks, calls the configured handler,
encodes the response on the original request id, and resumes turn processing.

- [ ] **Step 2: Run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_server_requests.py tests/unit_tests/test_session.py -q
```
Expected: FAIL until approval orchestration exists.

- [ ] **Step 3: Implement approval orchestration**

Add a session-level dispatcher that converts server requests into typed approval
objects and routes them through:

```python
def approval_handler(request: CodexServerRequest) -> CodexServerResponse: ...
```

Also add a minimal callback/logging hook layer in `observers.py` for:

- process started
- thread selected
- turn started
- approval requested
- approval resolved
- turn completed

- [ ] **Step 4: Re-run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_server_requests.py tests/unit_tests/test_session.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/observers.py langchain_codex/client.py langchain_codex/session.py tests/unit_tests/test_server_requests.py tests/unit_tests/test_session.py
git commit -m "feat(codex): add approval handling"
```

### Task 6: Implement session, thread, turn, review, command, realtime, and filesystem workflows

**Files:**
- Modify: `langchain_codex/session.py`
- Modify: `langchain_codex/client.py`
- Create: `tests/unit_tests/test_filesystem.py`
- Create: `tests/unit_tests/test_realtime.py`
- Modify: `tests/unit_tests/test_session.py`

- [ ] **Step 1: Write the failing session workflow tests**

Add tests for:

- lazy `thread/start`
- explicit `thread/resume`, `thread/fork`, `thread/list`, `thread/loaded/list`, and `thread/rollback`
- active-thread visibility in results and callbacks
- `turn/start`, `turn/steer`, and `turn/interrupt`
- `review/start`
- `command/exec*`
- `thread/realtime/*`
- `fs/*` wrapper behavior

- [ ] **Step 2: Run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_session.py tests/unit_tests/test_filesystem.py tests/unit_tests/test_realtime.py -q
```
Expected: FAIL until the session API covers the documented workflows.

- [ ] **Step 3: Implement the provider-native session API**

Add explicit methods with exact behavior mapping, for example:

```python
def start_thread(self, ...) -> CodexThreadHandle: ...
def resume_thread(self, thread_id: str, ...) -> CodexThreadHandle: ...
def fork_thread(self, thread_id: str, *, ephemeral: bool = False) -> CodexThreadHandle: ...
def run_turn(self, input_items: list[CodexInputItem], *, thread_id: str | None = None) -> CodexTurnResult: ...
def stream_turn(self, ...) -> Iterator[CodexEvent]: ...
def interrupt_turn(self, thread_id: str, turn_id: str) -> None: ...
```

Include wrappers for documented method families rather than hiding them behind a
single generic session call.

- [ ] **Step 4: Re-run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_session.py tests/unit_tests/test_filesystem.py tests/unit_tests/test_realtime.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/session.py langchain_codex/client.py tests/unit_tests/test_session.py tests/unit_tests/test_filesystem.py tests/unit_tests/test_realtime.py
git commit -m "feat(codex): add session workflows"
```

### Task 7: Rebuild `ChatCodex` as a thin LangChain adapter

**Files:**
- Modify: `langchain_codex/chat_models.py`
- Modify: `langchain_codex/__init__.py`
- Modify: `tests/unit_tests/test_chat_models.py`
- Modify: `tests/unit_tests/test_examples.py`

- [ ] **Step 1: Write the failing LangChain adapter tests**

Add tests for:

- `invoke`, `ainvoke`, `stream`, and `astream`
- response metadata carrying `thread_id` and `turn_id`
- adapter reuse of a configured client or session
- conversion of LangChain messages into documented Codex input items
- surfacing streamed `agentMessage` text without reimplementing protocol parsing

- [ ] **Step 2: Run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_chat_models.py tests/unit_tests/test_examples.py -q
```
Expected: FAIL until the new adapter is in place.

- [ ] **Step 3: Implement the adapter**

Keep `ChatCodex` thin:

- accept a `CodexClient` or config
- delegate thread and turn execution to `CodexSession`
- map provider-native outputs to LangChain `AIMessage`, `AIMessageChunk`, and metadata

- [ ] **Step 4: Re-run the targeted tests**

Run:
```bash
uv run --offline --group test pytest tests/unit_tests/test_chat_models.py tests/unit_tests/test_examples.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/chat_models.py langchain_codex/__init__.py tests/unit_tests/test_chat_models.py tests/unit_tests/test_examples.py
git commit -m "feat(codex): add langchain adapter"
```

### Task 8: Replace legacy modules, update docs, and verify package hygiene

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` only if package metadata or test groups need adjustment
- Delete: `langchain_codex/_types.py`
- Delete: `tests/unit_tests/test_types.py`

- [ ] **Step 1: Remove obsolete files and references**

Delete the old helper-only modules once all imports are moved to the new
provider layout. Ensure `py.typed` exists in the final package if packaging
expects it.

- [ ] **Step 2: Update README examples**

Document:

- direct launch and wrapped launch via `ai-creds run codex app-server`
- provider-native `CodexClient` and `CodexSession` usage
- `ChatCodex` usage
- approval handler usage
- explicit thread continuation usage

- [ ] **Step 3: Run package checks**

Run:
```bash
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typing pyright .
uv run --offline --group test pytest tests/unit_tests -q
```
Expected: PASS.

- [ ] **Step 4: Record equivalent local `.venv` verification commands**

From `libs/partners/codex`, the same checks must also work in the activated
virtualenv:

```bash
source .venv/bin/activate
ruff check .
ruff format --check .
pyright .
pytest tests/unit_tests -q
```

Expected: PASS.

- [ ] **Step 5: Sanity-check package cleanliness**

Confirm that no local artifact directories are staged:

```bash
git status --short
```
Expected: no staged `.venv`, `dist`, cache, rope, or `__pycache__` files from `libs/partners/codex`.

- [ ] **Step 6: Commit the final implementation**

```bash
git add -A
git commit -m "feat(codex): rebuild provider around app-server api"
```

## Final Verification Gate

Do not claim the work is complete until all of these succeed:

- `uv run --group lint ruff check .`
- `uv run --group lint ruff format --check .`
- `uv run --group typing pyright .`
- `uv run --offline --group test pytest tests/unit_tests -q`
- `source .venv/bin/activate && ruff check .`
- `source .venv/bin/activate && ruff format --check .`
- `source .venv/bin/activate && pyright .`

## Notes For The Implementer

- Use the generated app-server schema as a version-locked reference when a
  payload detail in the docs is ambiguous:

```bash
codex app-server generate-json-schema --out /tmp/codex-schema
```

- Keep request builders and event parsers table-driven where possible so new
  documented methods can be added without rewriting control flow.
- Preserve raw payload access on typed models for forward-compatible additive
  fields.
- If a documented experimental method is present in the README, implement it
  behind capability checks rather than omitting it.

# LangChain Codex Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `langchain_codex` to make the package easier to read and maintain while preserving public behavior, constructor signatures, and app-server protocol handling.

**Architecture:** Keep the current three-layer split: `ChatCodex` as the public facade, `CodexSession` as the orchestration layer, and `CodexAppServerTransport` as the process I/O adapter. Move repeated JSON-shape checks into `_types.py`, keep protocol-specific parsing close to the layer that uses it, and avoid introducing new public modules or API changes. Prefer extracting small private helpers over creating new files unless a responsibility is clearly shared across multiple modules.

**Tech Stack:** Python, `langchain-core`, `pytest`, `ruff`, `pyright`, `uv`

---

### Task 1: Lock in the current package boundaries with focused tests

**Files:**
- Modify: `tests/unit_tests/test_chat_models.py`
- Modify: `tests/unit_tests/test_session.py`
- Modify: `tests/unit_tests/test_transport.py`

- [ ] **Step 1: Write the failing tests**

Add or tighten tests that describe the current contract:
- `ChatCodex._build_session()` still passes the expected `request_timeout`, `turn_timeout`, `approval_policy`, and `sandbox` values into `CodexAppServerTransport` and `CodexSession`.
- `CodexSession.run_turn()` still returns `thread`, `turn`, and `events` with the same shapes.
- `CodexAppServerTransport.request()` still matches responses by request id and still forwards notifications to handlers.

- [ ] **Step 2: Run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_chat_models.py tests/unit_tests/test_session.py tests/unit_tests/test_transport.py -q`
Expected: PASS. These tests are just a contract baseline before the refactor starts.

- [ ] **Step 3: Commit the test baseline**

```bash
git add tests/unit_tests/test_chat_models.py tests/unit_tests/test_session.py tests/unit_tests/test_transport.py
git commit -m "test(codex): lock in package boundaries"
```

### Task 2: Clean up the shared JSON helpers

**Files:**
- Modify: `langchain_codex/_types.py`
- Modify: `tests/unit_tests/test_session.py`
- Modify: `tests/unit_tests/test_transport.py`

- [ ] **Step 1: Write the failing test**

Add one focused test that proves the helper functions continue to accept only string-keyed objects and nested JSON values for the payload shapes used by `turn` and `item` notifications.

- [ ] **Step 2: Run the targeted test**

Run: `uv run --offline --group test pytest tests/unit_tests/test_session.py tests/unit_tests/test_transport.py -q`
Expected: PASS before the refactor, then fail only if shape handling regresses.

- [ ] **Step 3: Refactor `_types.py`**

Consolidate repeated nested lookup helpers and keep the file focused on typed JSON access. Do not change public behavior, only reduce repeated boilerplate in callers.

- [ ] **Step 4: Re-run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_session.py tests/unit_tests/test_transport.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/_types.py tests/unit_tests/test_session.py tests/unit_tests/test_transport.py
git commit -m "refactor(codex): simplify json helpers"
```

### Task 3: Simplify `ChatCodex` into clearer helpers

**Files:**
- Modify: `langchain_codex/chat_models.py`
- Modify: `tests/unit_tests/test_chat_models.py`

- [ ] **Step 1: Write the failing tests**

Add tests that cover:
- prompt rendering for supported message types in `_render_message()`
- turn-text extraction from completed, streamed, and legacy event shapes in `_extract_turn_text()`
- session construction from the public model fields in `_build_session()`

- [ ] **Step 2: Run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_chat_models.py -q`
Expected: PASS before the refactor baseline, then fail only if the extracted helpers alter behavior.

- [ ] **Step 3: Refactor `ChatCodex`**

Split `_build_session`, `_app_server_command`, message rendering, and turn-text extraction into smaller helpers with descriptive names. Keep the public class, constructor fields, and returned LangChain objects unchanged. Do not add a new module unless the same helper would otherwise be duplicated across files.

- [ ] **Step 4: Re-run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_chat_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/chat_models.py tests/unit_tests/test_chat_models.py
git commit -m "refactor(codex): clarify chat model flow"
```

### Task 4: Simplify `CodexSession` turn orchestration

**Files:**
- Modify: `langchain_codex/session.py`
- Modify: `tests/unit_tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

Add or refine tests for:
- lazy thread creation in `_ensure_thread()`
- thread-start payload assembly in `_start_turn()`
- turn timeout handling in `_wait_for_next_notification()` and `_turn_timeout_message()`
- notification filtering by turn id in `_is_turn_notification_for()`
- extraction of streamed text deltas from supported app-server shapes in `_extract_text_delta()`

- [ ] **Step 2: Run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_session.py -q`
Expected: PASS before the refactor baseline, then fail only if the extracted helpers change behavior.

- [ ] **Step 3: Refactor the session lifecycle**

Extract smaller helpers for:
- creating the thread request payload
- waiting for turn notifications
- computing timeout deadlines
- formatting timeout diagnostics
- extracting turn ids and delta text

Keep the external session API and event shapes unchanged.

- [ ] **Step 4: Re-run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_session.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/session.py tests/unit_tests/test_session.py
git commit -m "refactor(codex): simplify session lifecycle"
```

### Task 5: Simplify the transport reader loop

**Files:**
- Modify: `langchain_codex/transport.py`
- Modify: `tests/unit_tests/test_transport.py`

- [ ] **Step 1: Write the failing tests**

Add coverage for:
- request/response matching in `request()` and `_deliver_response()`
- server-request rejection with diagnostics in `_handle_server_request()`
- notification dispatch in `_on_notification_message()`
- stderr diagnostics collection in `diagnostics()` and `_stderr_loop()`

- [ ] **Step 2: Run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_transport.py -q`
Expected: PASS before the refactor baseline, then fail only if the extracted helpers alter behavior.

- [ ] **Step 3: Refactor the transport**

Break `_reader_loop` and `_handle_reader_message` into small helpers for parsing, response delivery, notification dispatch, and failure handling. Preserve thread startup, response waiting, and diagnostics behavior.

- [ ] **Step 4: Re-run the targeted tests**

Run: `uv run --offline --group test pytest tests/unit_tests/test_transport.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_codex/transport.py tests/unit_tests/test_transport.py
git commit -m "refactor(codex): clarify transport flow"
```

### Task 6: Verify the full package and update docs if needed

**Files:**
- Modify: `README.md` only if the refactor changes any documented behavior

- [ ] **Step 1: Run the full checks**

Run:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --group lint ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run --group typing pyright .
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --group test pytest tests/unit_tests/test_chat_models.py tests/unit_tests/test_session.py tests/unit_tests/test_transport.py tests/unit_tests/test_examples.py
```
Expected: all checks pass.

- [ ] **Step 2: Update docs only if necessary**

If the refactor changes any user-facing behavior, update the README example or package notes. If behavior is unchanged, skip docs edits.

- [ ] **Step 3: Commit the final state**

```bash
git add -A
git commit -m "refactor(codex): simplify package internals"
```

## Guardrails

- Keep the public API unchanged unless a test proves a correctness fix is required.
- Prefer private helper extraction over new modules.
- Do not change the app-server protocol shapes the package accepts unless a regression test is added first.
- Re-run `ruff`, `pyright`, and the focused unit tests after each major task, not only at the end.

# Codex Provider Design

## Summary

Replace the current chat-only `langchain-codex` shape with a Codex provider
that mirrors the official Codex app-server JSON-RPC API as documented today.
The provider must expose both:

- a first-class provider API for sessions, threads, approvals, command
  execution, review, config/auth queries, and event consumption
- a LangChain `ChatCodex` adapter built on top of that provider API

The design is intentionally not constrained by the current implementation.
Feature scope must match the official app-server protocol exactly. The official
app-server protocol is the source of truth for method names, payload shape,
event shape, lifecycle semantics, and experimental/stable boundaries.

## Goals

- Support the documented Codex app-server API surface over local
  `codex app-server` `stdio`.
- Support wrapped launch commands such as `ai-creds run codex app-server`, not
  only direct `codex` execution.
- Support the documented event surface, not only text deltas.
- Expose synchronous human approvals through a user-supplied callback.
- Support in-process thread continuity, explicit thread reuse, thread fork, and
  explicit continuation of older threads.
- Surface which thread is active for each task/invoke/turn.
- Apply repo root and MCP configuration when creating threads.
- Keep the package structure simple, typed, and easy to extend.
- Pass local quality gates, including Ruff and Pyright.
- Avoid cache, build, legacy, or compatibility-shim clutter in the package.

## Non-Goals

- Backward compatibility with the current `ChatCodex` constructor or semantics.
- Multiple transport implementations in v1.
- Asynchronous external approval controllers in v1.
- Implementing undocumented protocol behavior.

## Protocol Source Of Truth

Implementation must track the official Codex app-server documentation and the
upstream app-server README instead of inferring behavior from the current local
package.

Source references:

- https://developers.openai.com/codex/app-server
- https://developers.openai.com/codex/app-server#events
- https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md#message-schema

## Design Principles

- Use the provided app-server API directly. Do not invent a parallel provider
  protocol.
- Preserve official method and event names internally. Python wrappers may use
  snake_case, but they must map one-to-one to wire methods.
- Keep stable and experimental surfaces distinct. Experimental features must be
  capability-gated, never silently assumed.
- Parse all documented events into typed models. Do not special-case only text
  outputs.
- Match the documented app-server surface as-is. Do not omit documented
  features, and do not invent product-facing features that are not part of the
  documented protocol.
- Separate transport, protocol parsing, provider state, and LangChain adapter
  concerns into different files.
- Expose typed wrappers for documented protocol features, plus a raw
  request/notification escape hatch only as compatibility plumbing.

## Public API

### Core provider API

- `CodexClient`
  - owns the app-server subprocess and transport lifecycle
  - performs `initialize` and `initialized`
  - exposes stable API methods that are not thread-turn scoped
  - creates and resumes `CodexSession` instances
- `CodexSession`
  - owns loaded-thread interaction and turn execution
  - can start, resume, fork, switch, compact, archive, rollback, and unsubscribe
    threads
  - handles approval requests during active turns
  - exposes streaming and non-streaming turn APIs
- `CodexThreadHandle`
  - typed record for `thread.id`, name, timestamps, loaded status, and current
    active turn metadata
- `CodexEventStream`
  - typed event iterator/dispatcher for all documented app-server notifications
- `CodexApprovalHandler`
  - user-supplied synchronous callback invoked for approval-bearing
    server-requests
- `ChatCodex`
  - LangChain `BaseChatModel` wrapper on top of `CodexSession`

### Provider config

`CodexClientConfig` should include at least:

- `launch_command`
- `model`
- `cwd`
- `approval_policy`
- `sandbox_policy`
- `reasoning_effort`
- `reasoning_summary`
- `personality`
- `service_name`
- `mcp_servers`
- `include_default_mcp_config`
- `request_timeout`
- `turn_timeout`
- `experimental_api`
- `opt_out_notification_methods`
- `approval_handler`

`launch_command` must support both:

- direct execution such as `codex app-server`
- wrapped execution such as `ai-creds run codex app-server`

The provider should treat this as an argv-style command definition, not shell
evaluation. The implementation must not require shell invocation just to support
command prefixes.

### Raw protocol access

To guarantee full API coverage without bloating the high-level object model, the
provider must expose:

- `client.request(method: str, params: JsonObject) -> JsonObject`
- `client.notify(method: str, params: JsonObject) -> None`
- `session.request(method: str, params: JsonObject) -> JsonObject`

This raw escape hatch is not the preferred ergonomic API, but it prevents the
package from blocking documented features while a typed wrapper is still being
added. It is compatibility plumbing, not a broader feature proposal.

## Required API Coverage

The implementation must provide typed wrappers or raw passthrough support for
every documented method family below. Method names and event names must match
the docs and upstream README exactly.

### Initialization and capability discovery

- `initialize`
- `initialized`
- `model/list`
- `experimentalFeature/list`
- `experimentalFeature/enablement/set`
- `collaborationMode/list`

### Thread lifecycle

- `thread/start`
- `thread/resume`
- `thread/fork`
- `thread/read`
- `thread/list`
- `thread/loaded/list`
- `thread/metadata/update`
- `thread/name/set`
- `thread/archive`
- `thread/unarchive`
- `thread/unsubscribe`
- `thread/compact/start`
- `thread/shellCommand`
- `thread/backgroundTerminals/clean`
- `thread/rollback`

### Turn lifecycle

- `turn/start`
- `turn/steer`
- `turn/interrupt`

### Thread realtime lifecycle

- `thread/realtime/start`
- `thread/realtime/appendAudio`
- `thread/realtime/appendText`
- `thread/realtime/stop`

### Review and execution

- `review/start`
- `command/exec`
- `command/exec/write`
- `command/exec/resize`
- `command/exec/terminate`

### Filesystem API

- `fs/readFile`
- `fs/writeFile`
- `fs/createDirectory`
- `fs/getMetadata`
- `fs/readDirectory`
- `fs/remove`
- `fs/copy`
- `fs/watch`
- `fs/unwatch`

### Skills, apps, MCP, and related config

- `skills/list`
- `plugin/list`
- `plugin/read`
- `plugin/install`
- `plugin/uninstall`
- `skills/config/write`
- `app/list`
- `mcpServer/oauth/login`
- `config/mcpServer/reload`
- `mcpServerStatus/list`
- `mcpServer/resource/read`

### Human input and approvals

- `tool/requestUserInput`
- handling of server-initiated approval requests such as
  `item/commandExecution/requestApproval`
- handling of server-initiated approval requests such as
  `item/fileChange/requestApproval`
- handling of server-initiated dynamic tool requests such as `item/tool/call`

### Feedback, config, and admin surfaces

- `feedback/upload`
- `config/read`
- `config/value/write`
- `config/batchWrite`
- `configRequirements/read`

### External-agent migration and auth surfaces

- `externalAgentConfig/detect`
- `externalAgentConfig/import`
- `account/read`
- `account/login/start`
- `account/login/cancel`
- `account/logout`
- handling of server-initiated `account/chatgptAuthTokens/refresh`
- `account/rateLimits/read`

### Platform-specific support

- `windowsSandbox/setupStart`

## Required Event Coverage

All documented notification families must be parsed and surfaced as typed event
objects. Unsupported event parsing is a bug.

### Thread events

- `thread/started`
- `thread/status/changed`
- `thread/name/updated`
- `thread/archived`
- `thread/unarchived`
- `thread/closed`
- `thread/tokenUsage/updated`

### Thread realtime events

- `thread/realtime/started`
- `thread/realtime/itemAdded`
- `thread/realtime/transcriptUpdated`
- `thread/realtime/outputAudio/delta`
- `thread/realtime/error`
- `thread/realtime/closed`

### Turn events

- `turn/started`
- `turn/completed`
- `turn/diff/updated`
- `turn/plan/updated`
- `model/rerouted`
- `error`

### Item lifecycle events

- `item/started`
- `item/updated`
- `item/completed`
- `item/autoApprovalReview/started`
- `item/autoApprovalReview/completed`
- `item/agentMessage/delta`
- `item/plan/delta` when emitted
- `item/reasoning/summaryTextDelta`
- `item/reasoning/summaryPartAdded`
- `item/reasoning/textDelta`
- `item/commandExecution/outputDelta`
- `item/fileChange/outputDelta`

### Server-request and approval lifecycle events

- `serverRequest/resolved`
- `mcpServer/oauthLogin/completed`
- `account/login/completed`
- `account/updated`
- `account/rateLimits/updated`

### Skills, apps, filesystem, and MCP events

- `command/exec/outputDelta`
- `skills/changed`
- `app/list/updated`
- `fs/changed`
- `mcpServer/startupStatus/updated`

### Search and platform events

- `fuzzyFileSearch/sessionUpdated`
- `fuzzyFileSearch/sessionCompleted`
- `windowsSandbox/setupCompleted`

### Notification-surface compatibility

- per-connection support for `optOutNotificationMethods`
- compatibility with documented `rawResponseItem/*` opt-out names, even when the
  provider surfaces those notifications as raw passthrough rather than typed
  models

## Required Item Coverage

The typed event and result models must support at least the documented item
union members:

- `userMessage`
- `agentMessage`
- `plan`
- `reasoning`
- `commandExecution`
- `fileChange`
- `mcpToolCall`
- `dynamicToolCall`
- `collabToolCall`
- `webSearch`
- `imageView`
- `enteredReviewMode`
- `exitedReviewMode`
- `contextCompaction`
- `compacted` as deprecated-but-readable compatibility for persisted history

## Required Input Coverage

Request builders and LangChain adapters must support the documented input item
shapes:

- `text`
- `image`
- `localImage`
- `mention`
- `skill`

`mention` support must work for both:

- `app://<connector-id>`
- `plugin://<plugin-name>@<marketplace-name>`

`ChatCodex` may expose only a subset as LangChain message content, but the
provider layer must retain the full typed item model.

## Wire Compatibility Rules

- Use app-server field names and enum values exactly on the wire.
- Keep a dedicated protocol layer that translates between Python naming and wire
  naming.
- Keep process-launch parsing separate from protocol parsing so command wrappers
  such as `ai-creds run codex app-server` are supported cleanly.
- Preserve unknown fields in debug/raw payloads where practical so the provider
  remains resilient to additive protocol changes.
- Fail clearly on missing required fields or invalid payload variants.
- Capability-gate experimental request fields such as `dynamicTools` and
  `persistExtendedHistory`, experimental approval payload extensions, and
  documented realtime or plugin surfaces that require experimental opt-in.
- Request models must carry documented field names and enum values as-is,
  including fields such as `approvalPolicy`, `sandbox`, `personality`,
  `serviceName`, `dynamicTools`, `settings.developer_instructions`,
  `persistExtendedHistory`, and documented approval-decision payload variants.

## Human Approval Model

The provider must support blocking human approval via a callback:

- `approval_handler(request: CodexServerRequest) -> CodexServerResponse`

The request model must support at least:

- command execution approval requests
- file change approval requests
- tool user-input requests
- dynamic tool call requests
- external token refresh requests

Decision models must support documented outcomes, including:

- `accept`
- `accept_for_session`
- `decline`
- `cancel`
- command-specific execution policy amendments when offered by the API
- network policy amendments when offered by the API

Behavior rules:

- the active turn blocks until the handler returns
- the provider responds on the same JSON-RPC request id
- the implementation's logging or callback layer receives both request and
  resolution events
- `serverRequest/resolved` is treated as authoritative completion of the pending
  request lifecycle
- command approval payloads must preserve documented fields such as
  `proposedExecpolicyAmendment`, `networkApprovalContext`,
  `availableDecisions`, and experimental `additionalPermissions` when present

## Thread Continuity Model

Thread continuity is in-process only for v1, but it must be explicit and
visible.

Required session behaviors:

- lazy thread creation on first turn when no thread is selected
- `start_thread()` for explicit new-thread creation
- `resume_thread(thread_id)` for rebinding to an existing thread
- `fork_thread(thread_id)` for branch creation
- `use_thread(thread_id)` for selecting a loaded or resumed thread
- `current_thread()` to inspect the active thread handle
- `list_known_threads()` to inspect locally known threads and their last-known
  status

Every non-streaming and streaming turn result must include:

- `thread_id`
- `turn_id`
- active thread status snapshot when available

Every thread-activity log or callback emitted by the provider must include the
active `thread_id` when one exists.

## Repo Root And MCP Rules

- `cwd` and MCP configuration belong to client and thread-start config.
- Every new thread started by a configured client inherits those values unless
  explicitly overridden.
- MCP support in v1 is configuration-driven only. The provider does not manage
  external MCP server processes itself.
- The provider must still expose read/list/status methods for the official
  app-server MCP surfaces.
- Thread start and resume must preserve documented config behavior such as
  thread-level model and reasoning settings fallback when those overrides are
  omitted.

## LangChain Adapter Rules

`ChatCodex` is a thin adapter over `CodexSession`.

It must support:

- `invoke`
- `ainvoke`
- `stream`
- `astream`

It must surface:

- final assistant text
- turn metadata including `thread_id` and `turn_id`
- streamed deltas from `agentMessage` and other relevant text-bearing item
  updates
- documented input item support where representable from LangChain calls, with
  provider-native access used for app mentions, plugin mentions, skills, and
  image inputs that do not map cleanly to plain chat text

It must not:

- own subprocess lifecycle logic directly
- duplicate protocol parsing logic already implemented by the provider layer

## Package Structure

The package layout should remain small and responsibility-oriented:

- `langchain_codex/__init__.py`
- `langchain_codex/client.py`
- `langchain_codex/session.py`
- `langchain_codex/chat_models.py`
- `langchain_codex/types.py`
- `langchain_codex/errors.py`
- `langchain_codex/observers.py`
- `langchain_codex/protocol/base.py`
- `langchain_codex/protocol/requests.py`
- `langchain_codex/protocol/events.py`
- `langchain_codex/protocol/items.py`
- `langchain_codex/transport/base.py`
- `langchain_codex/transport/stdio.py`

Keep test layout mirrored under `tests/unit_tests/`.

## Testing Requirements

The provider is not complete unless tests cover:

- request building for all documented stable methods
- request building for all documented experimental methods and fields behind
  capability gates
- capability-gated request building for experimental fields
- parsing for all documented event families
- item union parsing for all documented item types
- input union parsing and serialization for all documented input types
- thread lifecycle operations
- thread continuity and active-thread logging
- approval callback blocking flow and response encoding
- interruption and steering flows
- realtime request and event handling
- command execution request and streamed output handling
- filesystem API wrappers and `fs/changed`
- skills, apps, plugins, and MCP wrapper methods plus change/update events
- review mode lifecycle
- auth/config/admin method wrappers
- external token refresh server-request handling
- LangChain adapter invoke and stream behavior
- invalid payload and server error handling

The tests should remain unit-level and use fake transports instead of live Codex
processes whenever possible.

## Quality And Cleanup Rules

- All public Python code must have explicit type hints.
- Public functions and classes need Google-style docstrings.
- Ruff and Pyright must run cleanly for the package.
- Do not leave `.venv`, `dist`, caches, rope metadata, or generated junk inside
  `libs/partners/codex`.
- Do not keep legacy compatibility files solely to preserve the previous
  implementation shape.

## Open Implementation Constraint

The official docs include stable and experimental surfaces. The provider must
fully support the documented stable API and event families, and must preserve
the documented experimental API and event families behind capability gates.
Experimental features must be:

- discoverable through capability methods
- parsed when emitted
- exposed through raw passthrough and typed wrappers where documented
- clearly marked experimental in Python types and docstrings

This keeps the provider aligned to the documented app-server surface as-is,
without broadening it or shrinking it.

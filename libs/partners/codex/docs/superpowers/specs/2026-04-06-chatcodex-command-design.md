# ChatCodex Launch Command Design

## Summary

Add a new optional `ChatCodex` field that supports launching the Codex app server
through a command prefix such as `ai-creds run codex`, while preserving the
existing `codex_binary="codex"` behavior.

## Problem

`ChatCodex` currently launches the local transport with a fixed argv shape:
`[codex_binary, "app-server"]`.

That works only when the executable itself is directly invokable as `codex`. It
does not support environments where Codex must be wrapped by another command,
such as `ai-creds run codex app-server`.

## Goals

- Preserve the current default behavior and public constructor compatibility.
- Make it possible to launch `ChatCodex` through a prefixed command.
- Keep the change local to process startup rather than session or transport
  behavior.
- Cover the new launch path with unit tests.

## Non-Goals

- Changing how requests, threads, or streaming operate after startup.
- Adding shell execution or platform-specific quoting behavior.
- Removing or redefining `codex_binary`.

## Recommended Approach

Add a new optional field on `ChatCodex`:

- `codex_command: str | None = None`

Behavior:

- If `codex_command` is unset, keep the existing launch path:
  `["codex_binary", "app-server"]`.
- If `codex_command` is set, parse it into argv with `shlex.split()` and append
  `"app-server"`.
- Validate that the parsed command is non-empty and raise a clear `CodexError`
  if it is empty or invalid.
- Validate the executable by checking the first argv token with `shutil.which()`
  before launch, matching the current user-facing error style as closely as
  possible.

Examples:

- `ChatCodex(model="gpt-5.4")` launches `codex app-server`
- `ChatCodex(model="gpt-5.4", codex_command="ai-creds run codex")` launches
  `ai-creds run codex app-server`

## Alternatives Considered

### 1. Reinterpret `codex_binary` as a shell-style command string

This would avoid a new field, but it weakens the meaning of an existing public
field and introduces ambiguous behavior for callers that expect a single binary
name.

### 2. Add a list-valued argv field

This is explicit, but it is less ergonomic for common usage and makes README
examples noisier. A string command plus `shlex.split()` is a better fit for the
target use case.

## API Impact

This is an additive public API change only. Existing code that passes
`codex_binary` continues to work unchanged.

## Testing

Add unit tests that verify:

- Default launch still calls `subprocess.Popen(["codex", "app-server"], ...)`
- A configured `codex_command` launches
  `["ai-creds", "run", "codex", "app-server"]`
- Invalid or empty `codex_command` raises a clear error
- Existing missing-binary validation still behaves correctly

## Documentation

Update the README requirements and usage sections to mention the new field and
show an `ai-creds run codex` example.

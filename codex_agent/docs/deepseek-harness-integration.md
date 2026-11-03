# DeepSeek Harness Integration

## Architecture Decision

DeepSeek Harness is the general Agent control plane. Existing `codex_agent`
modules remain the Triton-RISCV domain engine.

```text
User
  -> React workbench
  -> FastAPI session, confirmation and SSE layer
  -> DeepSeek Harness Python SDK
  -> Harness agent loop, session log and tool pipeline
  -> Triton-RISCV domain tools
  -> existing Python discovery, validation, diagnosis and repair workflows
  -> local checkout or SSH-connected RISC-V validation host
```

This avoids maintaining two implementations of sessions, model streaming,
tool-call lifecycle, approval and replay. The existing bounded operator state
machine remains authoritative for acceptance-test locking, repair limits and
failure-stage classification.

## Difference From LangGraph

LangGraph is an embedded graph orchestration library. The application defines
state keys, nodes and every allowed transition. It is useful when a fixed graph
is the product's primary control mechanism.

DeepSeek Harness is a complete Agent runtime. Its agent loop repeatedly builds
model context, calls a model, executes registered tools and appends the result
to an event-sourced session. Models, tools, sessions, storage, approval,
sandboxing, workflows and UI are replaceable plugins.

The initial architecture therefore uses one top-level orchestrator: DeepSeek
Harness. A LangGraph subworkflow may be added later only if one domain operation
requires graph-specific checkpoint or transition semantics.

## Incremental Delivery

1. **Bootstrap and web bridge (complete):** prepare bounded Triton-RISCV
   context, launch the official SDK behind an adapter, and stream Harness
   notifications through FastAPI to React. No model call begins before user
   confirmation. The SDK subprocess is reused across turns and closed with the
   FastAPI application lifecycle.
2. **Typed tools (next):** register `discover_operator`, `validate_operator`,
   `diagnose_failure` and `develop_operator` as Harness tools instead of asking
   the model to construct shell commands.
3. **Remote executor:** add a guarded SSH provider so compilation and testing
   execute on the RISC-V host while Harness stays on a supported control host.
4. **Approval and repair loop:** require approval for writes and expensive runs,
   then expose the existing locked-test repair workflow as one domain tool.
5. **Memory and evaluation:** connect evidence memory, replay representative
   sessions, and compare success rate, tool errors, repair attempts and cost.

## Local Bootstrap

DeepSeek Harness requires Python 3.10 or newer. Keep it separate from the
existing Python 3.9 environment:

```sh
/opt/homebrew/Caskroom/miniconda/base/bin/python3 -m venv .harness-venv
.harness-venv/bin/python -m pip install -r codex_agent/harness/requirements.txt
```

Prepare and inspect a task without calling a model:

```sh
.harness-venv/bin/python -m codex_agent.harness \
  "验证 relu_and_mul 算子"
```

For a real run, configure a compatible endpoint and credential, then opt in:

```sh
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=...
export DSH_MODEL=...
.harness-venv/bin/python -m codex_agent.harness \
  "验证 relu_and_mul 算子" --live
```

Do not commit API keys. The Harness session home defaults to
`agent-results/deepseek-harness/`, which is already ignored by Git.

Start the connected React, FastAPI and Harness workbench with:

```sh
.harness-venv/bin/python -m codex_agent.platform --port 8765
```

## Verification Levels

The integration is verified at three levels:

1. Unit tests use a fake backend to check prompt boundaries, SDK argument
   mapping, runtime reuse, event forwarding and shutdown without model cost.
2. An HTTP integration test creates a session, posts an arbitrary task,
   confirms it, crosses the FastAPI-to-Harness adapter and verifies that the
   result and Harness event are persisted.
3. Frontend tests verify message and confirmation requests, followed by a
   strict TypeScript and Vite production build.

A real provider run still requires `DEEPSEEK_API_KEY`; a company endpoint also
uses `DEEPSEEK_BASE_URL` and `DSH_MODEL`. The no-key failure is intentional and
must not be reported as a successful live Agent run.

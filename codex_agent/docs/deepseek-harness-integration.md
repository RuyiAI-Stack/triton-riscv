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
  -> Harness MCP client
  -> Triton-RISCV Python MCP server
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
2. **Typed lifecycle tools (complete):** MCP exposes discovery, validation,
   diagnosis, repair proposal and approved repair application. Validation plans
   by default; live execution and source writes use separate host-side switches.
   Repair proposals lock both implementation and acceptance-test hashes.
3. **Approval boundary (complete):** FastAPI exposes proposal inspection and a
   separate human decision endpoint. Approval is deliberately not available to
   the model-facing MCP server, so the Agent cannot approve its own patch.
4. **Remote executor:** add a guarded SSH provider so compilation and testing
   execute on the RISC-V host while Harness stays on a supported control host.
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

Run the real stdio MCP server, list its lifecycle tools and call the read-only
operator discovery tool without a model credential:

```sh
.harness-venv/bin/python -m codex_agent.harness.mcp_demo relu_and_mul
```

The JSON output records the `stdio` transport, all five advertised lifecycle
tools, the discovery call and its structured result. For a known
operator, the result includes implementation and test paths, pytest nodes,
validation command, Triton kernels, Torch references and risk hints. This is
the deterministic second-delivery demo; it exercises the same subprocess
boundary that the Harness MCP client uses.

For a real run, export the ISRC credential, then opt in. The checked-in Harness
configuration defaults to `https://llmapi.isrc.ac.cn/v1`, the
`openai-responses` wire protocol and model `gpt-5.6-sol`:

```sh
export ISRC_API_KEY=...
# Optional overrides; the defaults above normally require no changes.
# export ISRC_BASE_URL=https://llmapi.isrc.ac.cn/v1
# export DSH_MODEL=gpt-5.6-sol
# Enable only the operations that this host is allowed to execute.
export TRITON_RISCV_ALLOW_VALIDATION=1
# Keep source writes disabled until repair application is intentionally tested.
# export TRITON_RISCV_ALLOW_REPAIR_APPLY=1
.harness-venv/bin/python -m codex_agent.harness \
  "验证 relu_and_mul 算子" --live
```

Do not commit API keys. The Harness session home defaults to
`agent-results/deepseek-harness/`, which is already ignored by Git.

Start the connected React, FastAPI and Harness workbench with:

```sh
.harness-venv/bin/python -m codex_agent.platform --port 8765
```

The repair lifecycle is:

```text
discover_operator
  -> validate_operator(execute=false)
  -> human confirms command
  -> validate_operator(execute=true)
  -> diagnose_failure(run_id)
  -> propose_repair(run_id, replacement_source, rationale)
  -> GET /api/repair-proposals/{proposal_id}
  -> POST /api/repair-proposals/{proposal_id}/decision
  -> apply_repair(proposal_id)
  -> validate_operator(execute=true)
```

The decision endpoint accepts JSON such as:

```json
{
  "approve": true,
  "reviewer": "ada-cl25",
  "note": "Reviewed the diff and preserved the acceptance test."
}
```

## Verification Levels

The integration is verified at three levels:

1. Unit tests use a fake backend to check prompt boundaries, SDK argument
   mapping, runtime reuse, event forwarding and shutdown without model cost.
2. An HTTP integration test creates a session, posts an arbitrary task,
   confirms it, crosses the FastAPI-to-Harness adapter and verifies that the
   result and Harness event are persisted.
3. Lifecycle tests prove that live validation is opt-in, unapproved patches are
   not applied, changed acceptance tests block a patch, and compiler limitations
   are not silently rewritten as operator fixes.
4. Frontend tests verify message and confirmation requests, followed by a
   strict TypeScript and Vite production build.

A real provider run still requires `ISRC_API_KEY`. `ISRC_BASE_URL` and
`DSH_MODEL` may override the checked-in defaults. The no-key failure is
intentional and must not be reported as a successful live Agent run. The MCP
demo proves tool registration and execution, but model-driven tool selection
requires a provider credential. Without that key, deterministic MCP and
lifecycle tests exercise the tool boundary but do not claim model autonomy.

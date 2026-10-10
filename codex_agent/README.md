# Triton-RISCV Autonomous Validation Agent

This directory contains tools for building an autonomous validation agent for
Triton-RISCV.

## Web Control Plane

The experimental web workbench connects a React interface to FastAPI and the
official DeepSeek Harness Python SDK. Build the frontend, then start the API:

```sh
cd codex_agent/frontend
npm ci
npm run build
cd ../..
.harness-venv/bin/python -m codex_agent.platform --port 8765
```

Open `http://127.0.0.1:8765`. Task execution requires explicit confirmation;
live model runs also require a configured `ISRC_API_KEY`. The default provider
uses the ISRC Responses-compatible endpoint and model `gpt-5.6-sol`. See
`codex_agent/docs/deepseek-harness-integration.md` for the architecture,
verification levels and next delivery steps.

The Harness MCP server now exposes a guarded five-step operator lifecycle:
discovery, validation, diagnosis, repair proposal and approved repair
application. Expensive validation and source writes are disabled by default;
the model-facing tools cannot approve their own repair proposals.

The agent is intended to discover validation targets, run them, classify
failures, and produce project-level coverage signals. It is separate from the
older operator-task workflow under `codex/`, which assumes that a human already
selected an operator.

There are now two operator flows:

- `operator_agent.py` discovers and validates operators that already have an
  implementation and pytest coverage in the repository.
- `develop_operator.py` accepts a structured semantic specification for a new
  or incomplete operator, asks Codex to implement it and its acceptance test,
  validates it, and performs bounded implementation-only repairs.

## New Operator Development Agent

An operator name alone is not enough to establish correctness. Describe the
operator in a JSON file conforming to `operator-spec.schema.json`. The minimum
contract contains:

- operator name and semantics
- PyTorch reference expression
- named inputs and output behavior
- shape and dtype cases
- numerical tolerances
- whether backward validation is required

See `specs/tanh_and_mul.json` for a complete example.

The development loop has been exercised with two independent specifications:

- `tanh_and_mul`: generated, diagnosed at `mlir-translate`, repaired with the
  locked test unchanged, then passed 20 forward/backward cases.
- `square_and_mul`: generated and passed 20 forward/backward cases on its first
  RISC-V validation run without repair.

### Prepare Context Without Editing Code

```sh
python -m codex_agent.develop_operator \
  --spec codex_agent/specs/tanh_and_mul.json \
  --remote-host "$RISCV_HOST" \
  --remote-root "$RISCV_REPO" \
  --prepare-only
```

This checks the RISC-V environment, discovers similar operators, and writes a
Codex task to `tasks/operators/<operator>.md`.

### Run the Complete Development Loop

Run this command from the local checkout where Codex CLI is authenticated:

```sh
python -m codex_agent.develop_operator \
  --spec codex_agent/specs/tanh_and_mul.json \
  --remote-host "$RISCV_HOST" \
  --remote-root "$RISCV_REPO" \
  --max-repairs 2
```

The default loop is:

1. Validate the structured specification and remote environment.
2. Discover and rank nearby repository implementations as references.
3. Generate an operator-specific task.
4. Invoke `codex exec` locally to create the implementation and pytest file.
5. Audit that the test imports the implementation, uses the specified PyTorch
   reference, covers required shapes/dtypes/backward, and calls
   `torch.testing.assert_close`.
6. Lock the acceptance-test hash and sync only the implementation/test files to
   the configured RISC-V host.
7. Run pytest, save the full log, and classify the failure stage.
8. For repairable failures, ask Codex for a bounded implementation-only fix and
   rerun the locked test.

The agent refuses code changes outside the two-file allowlist. Environment and
target-capability failures are diagnosis-only. Backend compiler failures are
also diagnosis-only by default; `--allow-compiler-workaround` explicitly
permits an implementation-only workaround attempt while keeping the test and
semantic contract locked.

Each run writes an auditable directory under
`agent-results/development/<timestamp>-<operator>/` containing the normalized
specification, selected references, generated patches, Codex logs, contract
audits, validation logs, and `final-result.json`.

After validation, the agent also writes a concise validation record back to the
tracked task file. Full run artifacts remain local and ignored by Git, while
the task retains the acceptance-test hash, attempt statuses, failure stage, and
pytest summary for code review.

The contract audit prevents obvious test weakening, but it is a structural
guard rather than a mathematical proof. Human review of generated code and the
recorded patch remains required before submission.

## Current Tooling

### `discover.py`

Discover validation targets in the repository:

```sh
python codex_agent/discover.py --output agent-results/targets.json
```

The discovery step scans:

- `python/examples/**/*.py` for pytest tests, Triton JIT kernels, and `tl.*`
  operations.
- `test/**/*.mlir` and `test/**/*.ll` for lit `RUN:` commands and compiler
  pass names.

The output JSON is the input for future runner and failure-classification
tools.

### `run_validation.py`

Run discovered targets or preview the commands that would run:

```sh
python codex_agent/run_validation.py --dry-run --limit 5
```

On a configured RISC-V server environment, source the Triton-RISCV environment
before each target:

```sh
python codex_agent/run_validation.py --kind pytest --path-contains vec_add --source-env
```

The runner writes:

- `agent-results/validation-<timestamp>.jsonl`
- `agent-results/logs/<timestamp>-<target>.log` for real runs

## Complete Operator Agent

Run the complete operator workflow with one command:

```sh
python -m codex_agent.operator_agent --limit 20
```

By default, this command:

- checks the configured Triton-RISCV environment
- discovers every existing FlagGems operator
- selects up to 20 public operators without completed results
- runs their mapped basic tests sequentially
- checkpoints every completed attempt immediately
- retries only timeout and unknown failures when `--retries` is set
- keeps the latest 200 logs while protecting logs referenced by latest results
- regenerates the latest-result summary and complete operator status reports

Useful variants:

```sh
# Preview the next batch without running tests.
python -m codex_agent.operator_agent --limit 20 --dry-run

# Allow one bounded retry for retryable failures.
python -m codex_agent.operator_agent --limit 20 --retries 1

# Recheck failed public operators.
python -m codex_agent.operator_agent --selection failed --limit 10

# Validate explicitly selected operators.
python -m codex_agent.operator_agent \
  --operator add --operator relu_and_mul
```

The one-command flow writes:

- `agent-results/operators.json`
- `agent-results/operator-validation-<timestamp>-agent.jsonl`
- `agent-results/logs/<timestamp>-<operator>.log`
- `agent-results/operator-summary.md`
- `agent-results/operator-status.json`
- `agent-results/operator-status.md`
- `agent-results/operator-agent-last-run.json`

`operator-status.md` lists every discovered operator as passed, failed,
skipped, planned, unvalidated, or unverified. A passed status means only that
the mapped basic tests passed in the recorded environment; it does not claim
correctness for untested inputs.

The lower-level commands below remain available for inspecting or running one
step manually.

## Manual Operator Flow

### 1. Discover Operators

```sh
python codex_agent/discover_operators.py --output agent-results/operators.json
```

This scans `python/examples/flaggems/*.py`, maps each implementation to pytest
tests, and records:

- operator name
- operator visibility (`public` or underscore-prefixed `internal`)
- implementation file
- test files and pytest node IDs
- Triton JIT kernels
- `tl.*` operations used by the implementation
- PyTorch reference expressions found in tests
- validation command
- likely risk hints

### 2. Preview One Operator Validation

```sh
python codex_agent/validate_operator.py sigmoid_and_mul --dry-run
```

List discovered operators:

```sh
python codex_agent/validate_operator.py --list
```

List operators whose names contain a substring:

```sh
python codex_agent/validate_operator.py --list --contains relu
```

List only public operators:

```sh
python codex_agent/validate_operator.py --list --public-only --limit 20
```

### 3. Run One Operator on a Configured RISC-V Environment

```sh
python codex_agent/validate_operator.py sigmoid_and_mul --source-env
```

Run the first five discovered operators as a small batch:

```sh
python codex_agent/validate_operator.py --all --limit 5 --source-env
```

Run the first five public operators and skip underscore-prefixed internal
helpers:

```sh
python codex_agent/validate_operator.py \
  --all --public-only --limit 5 --source-env
```

Each completed operator result is flushed to the JSONL file immediately. If a
batch is interrupted, rerun the same selection and append only missing results:

```sh
python codex_agent/validate_operator.py \
  --all --public-only --limit 5 --source-env \
  --resume-from agent-results/operator-validation-<timestamp>-batch.jsonl
```

Preview a filtered batch:

```sh
python codex_agent/validate_operator.py --all --contains relu --limit 3 --dry-run
```

The validator writes:

- `agent-results/operator-validation-<timestamp>.jsonl`
- `agent-results/logs/<timestamp>-<operator>.log`

It also classifies basic failure stages, including environment, import, build,
Triton conversion, Buddy lowering, MLIR translation, LLVM code generation,
target capability, runtime, and numerical correctness. Failed result records
include an `error_excerpt` field with the most useful lines extracted from the
full log.

### 4. Summarize Results

Summarize the latest validation result for each operator:

```sh
python codex_agent/summarize_operator_results.py
```

Write a markdown summary:

```sh
python codex_agent/summarize_operator_results.py \
  --output agent-results/operator-summary.md
```

The markdown summary includes pass/fail totals, failure-stage counts, stable
failure-signature clusters with affected operators, result paths, and
per-operator error excerpts for failed validations.

Summarize every attempt instead of only the latest result per operator:

```sh
python codex_agent/summarize_operator_results.py --all-attempts
```

## Operator Agent Scope

The existing-operator agent establishes whether mapped basic tests pass or fail
and reports operators without executable validation. The development agent
adds specification-driven generation and bounded repair for individual new or
incomplete operators. It does not claim that every possible Triton operator is
supported: complex layouts, dynamic contracts, unsupported dtypes, or missing
compiler lowering may still stop at a diagnosed failure.

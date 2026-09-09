#!/usr/bin/env python3
"""Generate, validate, diagnose, and optionally repair one Triton operator."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .discover_operators import discover_operators
from .operator_agent import run_preflight
from .validate_operator import classify_log, extract_error_excerpt, slugify


NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REMOTE_HOST_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
TORCH_SYMBOL_RE = re.compile(r"\b(torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
OPERATOR_ROOT = Path("python/examples/flaggems")
DEFAULT_RESULTS_DIR = Path("agent-results/development")
REPAIRABLE_STAGES = {
    "import",
    "triton-frontend",
    "runtime",
    "correctness",
    "pytest",
    "unknown",
}
COMPILER_STAGES = {
    "compilation",
    "triton-shared-opt",
    "buddy-opt",
    "mlir-translate",
    "llc",
}
VALIDATION_START = "<!-- autonomous-validation:start -->"
VALIDATION_END = "<!-- autonomous-validation:end -->"


@dataclass(frozen=True)
class OperatorSpec:
    schema_version: int
    name: str
    semantics: str
    pytorch_reference: str
    inputs: list[dict[str, str]]
    output: str
    shape_cases: list[list[int]]
    input_shape_cases: list[dict[str, list[int]]]
    dtypes: list[str]
    tolerances: dict[str, float]
    backward: bool
    implementation_file: str
    test_file: str
    reference_operators: list[str]
    notes: str


@dataclass
class ContractAudit:
    status: str
    errors: list[str]
    warnings: list[str]
    test_sha256: str | None


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_pytest_summary(text: str) -> str | None:
    summaries = re.findall(
        r"(?:\d+ (?:passed|failed|skipped|xfailed|xpassed))(?:, \d+ "
        r"(?:passed|failed|skipped|xfailed|xpassed))* in [0-9.]+s",
        text,
    )
    return summaries[-1] if summaries else None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes the repository: {relative_path}") from exc
    return path


def validate_operator_path(path: str, *, test_file: bool) -> None:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"operator path must be repository-relative: {path}")
    if candidate.suffix != ".py" or not candidate.is_relative_to(OPERATOR_ROOT):
        raise ValueError(f"operator path must be a Python file under {OPERATOR_ROOT}: {path}")
    if test_file and not candidate.name.startswith("test_"):
        raise ValueError(f"test file must start with test_: {path}")
    if not test_file and candidate.name.startswith("test_"):
        raise ValueError(f"implementation file cannot start with test_: {path}")


def load_operator_spec(path: Path, repo_root: Path) -> OperatorSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "name",
        "semantics",
        "pytorch_reference",
        "inputs",
        "output",
        "shape_cases",
        "dtypes",
        "tolerances",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"operator spec is missing fields: {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError("only operator spec schema_version 1 is supported")
    name = data["name"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid operator name: {name!r}")
    if not isinstance(data["semantics"], str) or len(data["semantics"].strip()) < 10:
        raise ValueError("semantics must be a concrete description")
    if not isinstance(data["pytorch_reference"], str) or "torch." not in data["pytorch_reference"]:
        raise ValueError("pytorch_reference must contain a torch expression")
    inputs = data["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must contain at least one input contract")
    input_names: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"name", "description"}:
            raise ValueError("each input requires only name and description")
        if not NAME_RE.fullmatch(str(item["name"])):
            raise ValueError(f"invalid input name: {item['name']!r}")
        if item["name"] in input_names:
            raise ValueError(f"duplicate input name: {item['name']}")
        input_names.add(item["name"])
    shape_cases = data["shape_cases"]
    if (
        not isinstance(shape_cases, list)
        or not shape_cases
        or any(
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            for shape in shape_cases
        )
    ):
        raise ValueError("shape_cases must be non-empty lists of positive integers")
    dtypes = data["dtypes"]
    if not isinstance(dtypes, list) or not dtypes or any(
        not isinstance(dtype, str) or not dtype.startswith("torch.") for dtype in dtypes
    ):
        raise ValueError("dtypes must contain one or more torch dtype names")
    input_shape_cases = data.get("input_shape_cases", [])
    if not isinstance(input_shape_cases, list):
        raise ValueError("input_shape_cases must be a list")
    for case in input_shape_cases:
        if not isinstance(case, dict) or set(case) != input_names:
            raise ValueError(
                "each input_shape_cases entry must provide every named input"
            )
        if any(
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            for shape in case.values()
        ):
            raise ValueError("input_shape_cases must use positive integer shapes")
    tolerances = data["tolerances"]
    if set(tolerances) != {"rtol", "atol"} or any(
        not isinstance(tolerances[key], (int, float)) or tolerances[key] < 0
        for key in ("rtol", "atol")
    ):
        raise ValueError("tolerances must contain non-negative rtol and atol")
    implementation_file = data.get(
        "implementation_file",
        (OPERATOR_ROOT / f"{name}.py").as_posix(),
    )
    test_file = data.get(
        "test_file",
        (OPERATOR_ROOT / f"test_{name}.py").as_posix(),
    )
    validate_operator_path(implementation_file, test_file=False)
    validate_operator_path(test_file, test_file=True)
    resolve_repo_path(repo_root, implementation_file)
    resolve_repo_path(repo_root, test_file)
    references = data.get("reference_operators", [])
    if not isinstance(references, list) or any(not isinstance(item, str) for item in references):
        raise ValueError("reference_operators must be a list of operator names")
    return OperatorSpec(
        schema_version=1,
        name=name,
        semantics=data["semantics"].strip(),
        pytorch_reference=data["pytorch_reference"].strip(),
        inputs=inputs,
        output=str(data["output"]).strip(),
        shape_cases=shape_cases,
        input_shape_cases=input_shape_cases,
        dtypes=dtypes,
        tolerances={key: float(tolerances[key]) for key in ("rtol", "atol")},
        backward=bool(data.get("backward", False)),
        implementation_file=implementation_file,
        test_file=test_file,
        reference_operators=references,
        notes=str(data.get("notes", "")).strip(),
    )


def operator_tokens(value: str) -> set[str]:
    ignored = {"and", "operator", "torch", "tensor", "compute", "elementwise"}
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9]*", value.lower())
        if token not in ignored and len(token) > 1
    }


def choose_references(spec: OperatorSpec, inventory: dict, limit: int = 4) -> list[dict]:
    operators = inventory.get("operators", [])
    explicit = {name: index for index, name in enumerate(spec.reference_operators)}
    query_tokens = operator_tokens(
        f"{spec.name} {spec.semantics} {spec.pytorch_reference}"
    )
    fused_suffix = spec.name.split("_", 1)[1] if "_" in spec.name else ""
    scored: list[tuple[int, str, dict]] = []
    for operator in operators:
        name = operator["name"]
        if name == spec.name:
            continue
        score = 0
        if name in explicit:
            score += 100 - explicit[name]
        candidate_text = " ".join(
            [name, *operator.get("torch_references", []), *operator.get("tl_ops", [])]
        )
        score += 6 * len(query_tokens & operator_tokens(candidate_text))
        if fused_suffix and name.endswith(fused_suffix):
            score += 12
        if spec.backward and any("backward" in function for function in operator.get("public_functions", [])):
            score += 3
        if score:
            scored.append((-score, name, operator))
    return [item[2] for item in sorted(scored)[:limit]]


def render_task(spec: OperatorSpec, references: list[dict]) -> str:
    inputs = "\n".join(
        f"- `{item['name']}`: {item['description']}" for item in spec.inputs
    )
    reference_lines = "\n".join(
        f"- `{item['implementation_file']}` ({item['name']}; tl ops: "
        f"{', '.join(item.get('tl_ops', [])) or 'none detected'})"
        for item in references
    ) or "- No close repository reference was discovered; inspect nearby FlagGems operators."
    shapes = ", ".join(str(tuple(shape)) for shape in spec.shape_cases)
    input_shape_cases = (
        json.dumps(spec.input_shape_cases, sort_keys=True)
        if spec.input_shape_cases
        else "none"
    )
    dtypes = ", ".join(spec.dtypes)
    backward = "required" if spec.backward else "not required"
    return f"""# Operator Development Task: {spec.name}

## Immutable Contract

- Semantics: {spec.semantics}
- PyTorch reference: `{spec.pytorch_reference}`
- Output: {spec.output}
- Shape cases: {shapes}
- Per-input shape cases: `{input_shape_cases}`
- Dtypes: {dtypes}
- Maximum tolerance: rtol={spec.tolerances['rtol']}, atol={spec.tolerances['atol']}
- Backward validation: {backward}

### Inputs

{inputs}

## Automatically Selected References

{reference_lines}

## Allowed Files

- Implementation: `{spec.implementation_file}`
- Test: `{spec.test_file}`

Do not modify the contract, reference expression, shape/dtype coverage, or
tolerances merely to make validation pass. During repair iterations, the test
file is locked and only the implementation may be changed.

## Required Work

1. Inspect the selected references and `docs/05-Operator Migration.md`.
2. Implement a clear Triton kernel and Python wrapper matching the contract.
3. Add pytest coverage that computes the PyTorch reference independently and
   compares it with `torch.testing.assert_close`.
4. Include every required shape and dtype and the backward path when required.
5. Keep pointer arithmetic, masks, casts, and boundary behavior explicit.
6. Do not use autotuning in the first implementation.

## Validation

```sh
source scripts/triton-riscv-env.sh
python -m pytest -q {spec.test_file} -s
```

The orchestration agent runs this command in the configured RISC-V environment,
classifies failures, and permits only bounded semantic-preserving repairs.

## Notes

{spec.notes or "No additional notes."}
"""


def collect_literal_shapes(tree: ast.AST) -> set[tuple[int, ...]]:
    shapes: set[tuple[int, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        values: list[int] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, int):
                break
            values.append(element.value)
        else:
            if values:
                shapes.add(tuple(values))
    return shapes


def audit_test_contract(spec: OperatorSpec, repo_root: Path) -> ContractAudit:
    test_path = resolve_repo_path(repo_root, spec.test_file)
    errors: list[str] = []
    warnings: list[str] = []
    if not test_path.exists():
        return ContractAudit("failed", [f"missing test file: {spec.test_file}"], [], None)
    try:
        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        return ContractAudit("failed", [f"test file cannot be parsed: {exc}"], [], None)

    module_name = Path(spec.implementation_file).stem
    imports_implementation = any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.split(".")[-1] == module_name
        for node in ast.walk(tree)
    )
    if not imports_implementation:
        errors.append(f"test does not import the {module_name} implementation module")
    test_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]
    if not test_functions:
        errors.append("test file defines no pytest test functions")
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if "torch.testing.assert_close" not in calls:
        errors.append("test must compare results with torch.testing.assert_close")
    source_without_space = "".join(source.split())
    required_symbols = sorted(set(TORCH_SYMBOL_RE.findall(spec.pytorch_reference)))
    for symbol in required_symbols:
        if symbol not in source_without_space:
            errors.append(f"test is missing PyTorch reference symbol {symbol}")
    unparsed = ast.unparse(tree)
    for dtype in spec.dtypes:
        if dtype not in unparsed:
            errors.append(f"test is missing required dtype {dtype}")
    literal_shapes = collect_literal_shapes(tree)
    for shape in spec.shape_cases:
        if tuple(shape) not in literal_shapes:
            errors.append(f"test is missing required shape {tuple(shape)}")
    for case in spec.input_shape_cases:
        for input_name, shape in case.items():
            if tuple(shape) not in literal_shapes:
                errors.append(
                    f"test is missing {input_name} shape {tuple(shape)} from an input-shape case"
                )
    if spec.backward and not any(
        "backward" in node.name for node in test_functions
    ):
        errors.append("operator requires a backward pytest case")
    if spec.backward and ".backward(" not in source_without_space:
        warnings.append("backward test does not visibly call the PyTorch backward reference")
    assert_close_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.testing.assert_close"
    ]
    for call in assert_close_calls:
        keywords = {item.arg: item.value for item in call.keywords if item.arg}
        for tolerance in ("rtol", "atol"):
            value = keywords.get(tolerance)
            if value is None:
                warnings.append(
                    f"an assert_close call does not set {tolerance} explicitly"
                )
            elif isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                if float(value.value) > spec.tolerances[tolerance]:
                    errors.append(
                        f"assert_close {tolerance}={value.value} exceeds the contract maximum "
                        f"{spec.tolerances[tolerance]}"
                    )
            elif isinstance(value, ast.Name):
                warnings.append(
                    f"assert_close {tolerance} uses variable {value.id}; review its parameter values"
                )
    return ContractAudit(
        "passed" if not errors else "failed",
        errors,
        warnings,
        file_sha256(test_path),
    )


def collect_workspace_state(repo_root: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    state: dict[str, str] = {}
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", "surrogateescape")
        path = repo_root / relative
        state[relative] = file_sha256(path) if path.is_file() else "<missing>"
    return state


def unauthorized_changes(
    before: dict[str, str],
    after: dict[str, str],
    allowed_paths: set[str],
) -> list[str]:
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path) and path not in allowed_paths
    )


def snapshot_files(repo_root: Path, paths: list[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for relative in paths:
        path = resolve_repo_path(repo_root, relative)
        snapshot[relative] = path.read_text(encoding="utf-8") if path.exists() else ""
    return snapshot


def write_patch(path: Path, before: dict[str, str], after: dict[str, str]) -> None:
    lines: list[str] = []
    for relative in sorted(before.keys() | after.keys()):
        lines.extend(
            difflib.unified_diff(
                before.get(relative, "").splitlines(keepends=True),
                after.get(relative, "").splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    path.write_text("".join(lines), encoding="utf-8")


def run_codex(
    repo_root: Path,
    prompt: str,
    run_dir: Path,
    label: str,
    timeout_seconds: int,
) -> dict:
    executable = shutil.which("codex")
    if not executable:
        return {"status": "failed", "exit_code": 127, "reason": "codex executable not found"}
    final_message = run_dir / f"{label}-final-message.md"
    command = [
        executable,
        "exec",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--cd",
        repo_root.as_posix(),
        "--output-last-message",
        final_message.as_posix(),
        "-",
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_seconds} seconds\n"
        exit_code = 124
    (run_dir / f"{label}.log").write_text(output, encoding="utf-8")
    return {
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "duration_seconds": round(time.monotonic() - start, 3),
        "log_path": (run_dir / f"{label}.log").as_posix(),
        "final_message_path": final_message.as_posix() if final_message.exists() else None,
    }


def generation_prompt(spec: OperatorSpec, task_path: Path, references: list[dict]) -> str:
    references_text = "\n".join(
        f"- {item['implementation_file']}" for item in references
    )
    return f"""Implement the Triton-RISCV operator described in {task_path.as_posix()}.

Read the task, docs/05-Operator Migration.md, and these selected references:
{references_text or '- inspect nearby FlagGems operators'}

You may create or edit only:
- {spec.implementation_file}
- {spec.test_file}

Do not edit the task, workflow, reference expression, expected values, shapes,
dtypes, or tolerances. The pytest test must independently compute
`{spec.pytorch_reference}` and use torch.testing.assert_close. Do not run git
commands or commit. The RISC-V validation environment is remote, so finish with
local syntax/static checks only and let the orchestration agent run pytest.
"""


def contract_repair_prompt(spec: OperatorSpec, audit: ContractAudit) -> str:
    errors = "\n".join(f"- {item}" for item in audit.errors)
    return f"""The generated test for {spec.name} failed the immutable contract audit:
{errors}

Correct the implementation/test package without changing the operator contract.
You may edit only {spec.implementation_file} and {spec.test_file}. The test must
compute `{spec.pytorch_reference}` independently, cover all required shapes and
dtypes, and use torch.testing.assert_close. Do not reduce coverage or loosen the
specified maximum tolerances. Do not run git commands or commit.
"""


def repair_prompt(spec: OperatorSpec, validation: dict) -> str:
    excerpt = "\n".join(validation.get("error_excerpt", []))
    return f"""Repair the {spec.name} Triton implementation using this validation evidence.

Failure stage: {validation.get('failure_stage')}
Likely reason: {validation.get('likely_reason')}
Error excerpt:
{excerpt}

The test file is a locked acceptance test. Edit only
{spec.implementation_file}. Do not edit {spec.test_file}, the PyTorch reference,
shapes, dtypes, expected values, or tolerances. Preserve the semantics
`{spec.pytorch_reference}` and make the smallest evidence-based implementation
change. Do not run git commands or commit.
"""


def run_ssh(host: str, command: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    if not REMOTE_HOST_RE.fullmatch(host):
        raise ValueError(f"invalid SSH host: {host!r}")
    return subprocess.run(
        ["ssh", host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def remote_environment_prefix(remote_root: str) -> str:
    root = shlex.quote(remote_root)
    return (
        f"cd {root} && "
        "source .venv/bin/activate && "
        "source scripts/triton-riscv-env.sh"
    )


def run_remote_preflight(host: str, remote_root: str) -> dict:
    checks = (
        "python -c 'import triton; print(triton.__version__)' && "
        "command -v triton-shared-opt && command -v buddy-opt"
    )
    start = time.monotonic()
    try:
        completed = run_ssh(
            host,
            f"{remote_environment_prefix(remote_root)} && {checks}",
            90,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "exit_code": 124,
            "reason": "remote environment check timed out",
            "output": exc.stdout or "",
        }
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "reason": None if completed.returncode == 0 else "remote environment check failed",
        "duration_seconds": round(time.monotonic() - start, 3),
        "output": completed.stdout.strip(),
    }


def sync_remote_files(
    spec: OperatorSpec,
    repo_root: Path,
    host: str,
    remote_root: str,
) -> None:
    for relative in (spec.implementation_file, spec.test_file):
        local_path = resolve_repo_path(repo_root, relative)
        if not local_path.exists():
            raise FileNotFoundError(f"cannot sync missing file: {relative}")
        remote_path = f"{remote_root.rstrip('/')}/{relative}"
        parent = str(Path(remote_path).parent)
        mkdir_result = run_ssh(host, f"mkdir -p {shlex.quote(parent)}", 30)
        if mkdir_result.returncode != 0:
            raise RuntimeError(f"remote mkdir failed: {mkdir_result.stdout.strip()}")
        completed = subprocess.run(
            ["scp", local_path.as_posix(), f"{host}:{remote_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"scp failed for {relative}: {completed.stdout.strip()}")


def run_validation(
    spec: OperatorSpec,
    repo_root: Path,
    run_dir: Path,
    iteration: int,
    timeout_seconds: int,
    remote_host: str | None,
    remote_root: str | None,
    source_env: bool,
) -> dict:
    command = f"python -m pytest -q {shlex.quote(spec.test_file)} -s"
    start = time.monotonic()
    if remote_host:
        assert remote_root is not None
        sync_remote_files(spec, repo_root, remote_host, remote_root)
        shell_command = f"{remote_environment_prefix(remote_root)} && {command}"
        try:
            completed = run_ssh(remote_host, shell_command, timeout_seconds)
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_seconds} seconds\n"
            exit_code = 124
        display_command = f"ssh {remote_host} {shlex.quote(shell_command)}"
    else:
        shell_command = command
        if source_env:
            shell_command = f"source scripts/triton-riscv-env.sh && {shell_command}"
        try:
            completed = subprocess.run(
                ["bash", "-lc", shell_command],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_seconds} seconds\n"
            exit_code = 124
        display_command = shell_command
    log_path = run_dir / f"validation-{iteration}.log"
    log_path.write_text(output, encoding="utf-8")
    status, failure_stage, likely_reason = classify_log(output, exit_code)
    result = {
        "operator": spec.name,
        "iteration": iteration,
        "command": display_command,
        "status": status,
        "exit_code": exit_code,
        "failure_stage": failure_stage,
        "likely_reason": likely_reason,
        "error_excerpt": extract_error_excerpt(output) if status == "failed" else [],
        "test_summary": extract_pytest_summary(output),
        "duration_seconds": round(time.monotonic() - start, 3),
        "log_path": log_path.as_posix(),
    }
    write_json(run_dir / f"validation-{iteration}.json", result)
    return result


def update_task_validation_record(task_path: Path, final: dict) -> None:
    content = task_path.read_text(encoding="utf-8")
    if VALIDATION_START in content and VALIDATION_END in content:
        prefix = content.split(VALIDATION_START, 1)[0].rstrip()
    else:
        prefix = content.rstrip()
    lines = [
        VALIDATION_START,
        "",
        "## Autonomous Validation Record",
        "",
        f"- Final status: `{final['status']}`",
        f"- Acceptance-test SHA-256: `{final['locked_test_sha256']}`",
        f"- Repair attempts: {final['repair_attempts']}",
        "",
        "| Attempt | Status | Failure stage | Test summary |",
        "| --- | --- | --- | --- |",
    ]
    for validation in final["validations"]:
        lines.append(
            f"| {validation['iteration']} | {validation['status']} | "
            f"{validation.get('failure_stage') or ''} | "
            f"{validation.get('test_summary') or ''} |"
        )
    failure_excerpts = [
        line
        for validation in final["validations"]
        for line in validation.get("error_excerpt", [])[:1]
    ]
    if failure_excerpts:
        lines.extend(["", "### Failure Evidence", ""])
        lines.extend(f"- `{line}`" for line in failure_excerpts)
    lines.extend(
        [
            "",
            "Full logs, Codex transcripts, and per-iteration patches are stored "
            "under the run directory shown by the command output.",
            "",
            VALIDATION_END,
        ]
    )
    task_path.write_text(prefix + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def may_repair(stage: str | None, allow_compiler_workaround: bool) -> bool:
    if stage in REPAIRABLE_STAGES:
        return True
    return bool(allow_compiler_workaround and stage in COMPILER_STAGES)


def ensure_codex_changes_allowed(
    repo_root: Path,
    before_state: dict[str, str],
    allowed: set[str],
) -> None:
    changed = unauthorized_changes(before_state, collect_workspace_state(repo_root), allowed)
    if changed:
        raise RuntimeError(
            "Codex changed files outside the allowlist; review them manually: "
            + ", ".join(changed)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop and validate one operator from a structured specification."
    )
    parser.add_argument("--spec", required=True, help="Operator specification JSON.")
    parser.add_argument("--repo-root", default=".", help="Local repository root.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR.as_posix())
    parser.add_argument("--remote-host", default=None, help="SSH host used for RISC-V validation.")
    parser.add_argument("--remote-root", default=None, help="Repository root on the remote host.")
    parser.add_argument("--source-env", action="store_true", help="Source the environment for local validation.")
    parser.add_argument("--prepare-only", action="store_true", help="Generate context without invoking Codex.")
    parser.add_argument("--allow-existing", action="store_true", help="Allow completion of existing implementation/test files.")
    parser.add_argument("--force-task", action="store_true", help="Overwrite an existing generated task file.")
    parser.add_argument("--max-generation-attempts", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--codex-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--allow-compiler-workaround",
        action="store_true",
        help="Allow implementation-only repair attempts for backend compiler failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_generation_attempts < 1 or args.max_repairs < 0:
        raise SystemExit("generation attempts must be >= 1 and repairs must be >= 0")
    if bool(args.remote_host) != bool(args.remote_root):
        raise SystemExit("--remote-host and --remote-root must be supplied together")
    repo_root = Path(args.repo_root).resolve()
    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = repo_root / spec_path
    try:
        spec = load_operator_spec(spec_path, repo_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid operator spec: {exc}") from exc

    implementation_path = resolve_repo_path(repo_root, spec.implementation_file)
    test_path = resolve_repo_path(repo_root, spec.test_file)
    if not args.allow_existing and (implementation_path.exists() or test_path.exists()):
        raise SystemExit("operator files already exist; pass --allow-existing to complete them")

    inventory = discover_operators(repo_root)
    references = choose_references(spec, inventory)
    task_path = repo_root / "tasks" / "operators" / f"{spec.name}.md"
    if task_path.exists() and not args.force_task:
        raise SystemExit(f"task already exists: {task_path}; pass --force-task to replace it")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(render_task(spec, references), encoding="utf-8")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    run_dir = results_root / f"{timestamp}-{slugify(spec.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "operator-spec.json", asdict(spec))
    write_json(run_dir / "references.json", references)

    if args.remote_host:
        preflight = run_remote_preflight(args.remote_host, args.remote_root)
    else:
        preflight = run_preflight(repo_root, args.source_env, False)
    write_json(run_dir / "preflight.json", preflight)
    prepared = {
        "operator": spec.name,
        "task_path": task_path.relative_to(repo_root).as_posix(),
        "references": [item["name"] for item in references],
        "preflight": preflight,
        "run_dir": run_dir.as_posix(),
    }
    print(json.dumps(prepared, indent=2, sort_keys=True))
    if args.prepare_only:
        return 0
    if preflight["status"] != "passed":
        write_json(run_dir / "final-result.json", {**prepared, "status": "environment-failed"})
        return 2

    allowed_generation = {spec.implementation_file, spec.test_file}
    file_paths = [spec.implementation_file, spec.test_file]
    previous_snapshot = snapshot_files(repo_root, file_paths)
    contract_audit: ContractAudit | None = None
    for attempt in range(1, args.max_generation_attempts + 1):
        before_state = collect_workspace_state(repo_root)
        prompt = (
            generation_prompt(spec, task_path.relative_to(repo_root), references)
            if attempt == 1
            else contract_repair_prompt(spec, contract_audit)
        )
        codex_result = run_codex(
            repo_root,
            prompt,
            run_dir,
            f"codex-generation-{attempt}",
            args.codex_timeout_seconds,
        )
        write_json(run_dir / f"codex-generation-{attempt}.json", codex_result)
        if codex_result["status"] != "passed":
            write_json(run_dir / "final-result.json", {**prepared, "status": "generation-failed", "codex": codex_result})
            return 3
        ensure_codex_changes_allowed(repo_root, before_state, allowed_generation)
        current_snapshot = snapshot_files(repo_root, file_paths)
        write_patch(run_dir / f"generation-{attempt}.patch", previous_snapshot, current_snapshot)
        previous_snapshot = current_snapshot
        contract_audit = audit_test_contract(spec, repo_root)
        write_json(run_dir / f"contract-audit-{attempt}.json", asdict(contract_audit))
        if contract_audit.status == "passed":
            break
    assert contract_audit is not None
    if contract_audit.status != "passed":
        write_json(run_dir / "final-result.json", {**prepared, "status": "test-contract-failed", "contract_audit": asdict(contract_audit)})
        return 4

    locked_test_hash = contract_audit.test_sha256
    validations: list[dict] = []
    validation = run_validation(
        spec,
        repo_root,
        run_dir,
        1,
        args.timeout_seconds,
        args.remote_host,
        args.remote_root,
        args.source_env,
    )
    validations.append(validation)

    for repair_index in range(1, args.max_repairs + 1):
        if validation["status"] == "passed":
            break
        if not may_repair(validation["failure_stage"], args.allow_compiler_workaround):
            break
        before_state = collect_workspace_state(repo_root)
        before_snapshot = snapshot_files(repo_root, file_paths)
        codex_result = run_codex(
            repo_root,
            repair_prompt(spec, validation),
            run_dir,
            f"codex-repair-{repair_index}",
            args.codex_timeout_seconds,
        )
        write_json(run_dir / f"codex-repair-{repair_index}.json", codex_result)
        if codex_result["status"] != "passed":
            break
        ensure_codex_changes_allowed(repo_root, before_state, {spec.implementation_file})
        if file_sha256(test_path) != locked_test_hash:
            raise RuntimeError("locked acceptance test changed during implementation repair")
        after_snapshot = snapshot_files(repo_root, file_paths)
        write_patch(run_dir / f"repair-{repair_index}.patch", before_snapshot, after_snapshot)
        validation = run_validation(
            spec,
            repo_root,
            run_dir,
            repair_index + 1,
            args.timeout_seconds,
            args.remote_host,
            args.remote_root,
            args.source_env,
        )
        validations.append(validation)

    final = {
        **prepared,
        "status": validation["status"],
        "contract_audit": asdict(contract_audit),
        "locked_test_sha256": locked_test_hash,
        "validations": validations,
        "repair_attempts": len(validations) - 1,
    }
    update_task_validation_record(task_path, final)
    write_json(run_dir / "final-result.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if validation["status"] == "passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate discovered operators and classify the results."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_OPERATORS = Path("agent-results/operators.json")
DEFAULT_RESULTS_DIR = Path("agent-results")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
PRIMARY_ERROR_LINE_PATTERNS = (
    re.compile(r"\berror:", re.IGNORECASE),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
    re.compile(r"\bdialect .+ not found\b", re.IGNORECASE),
    re.compile(r"\bmismatched elements\b", re.IGNORECASE),
    re.compile(r"^ninja: build stopped", re.IGNORECASE),
    re.compile(r"\bTIMEOUT after\b", re.IGNORECASE),
)
ROOT_CAUSE_LINE_PATTERNS = (
    re.compile(
        r"\b(?:assertion|attribute|import|module.?not.?found|runtime|value)error\b",
        re.IGNORECASE,
    ),
)
SECONDARY_ERROR_LINE_PATTERNS = (
    re.compile(r"^traceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r"\bcompilationerror\b", re.IGNORECASE),
    re.compile(r"\bsubprocess\.calledprocesserror\b", re.IGNORECASE),
    re.compile(r"^FAILED\s+", re.IGNORECASE),
)


@dataclass
class OperatorValidationResult:
    operator: str
    implementation_file: str
    test_files: list[str]
    command: str
    dry_run: bool
    exit_code: int | None
    status: str
    failure_stage: str | None
    likely_reason: str | None
    error_excerpt: list[str]
    duration_seconds: float
    log_path: str | None


def load_operators(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("operators", [])


def load_completed_operators(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") in {"passed", "failed", "skipped"}:
            completed.add(item["operator"])
    return completed


def find_operator(operators: list[dict], name: str) -> dict:
    for operator in operators:
        if operator["name"] == name:
            return operator
    available = ", ".join(item["name"] for item in operators)
    raise SystemExit(f"operator {name!r} not found; available: {available}")


def slugify(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_").replace(":", "_")


def build_shell_command(command: str, source_env: bool) -> str:
    if not source_env:
        return command
    return f"source scripts/triton-riscv-env.sh && {command}"


def operator_visibility(operator: dict) -> str:
    return operator.get(
        "visibility",
        "internal" if operator["name"].startswith("_") else "public",
    )


def operator_matches(
    operator: dict,
    contains: str | None,
    visibility: str | None,
) -> bool:
    if contains and contains not in operator["name"]:
        return False
    if visibility and operator_visibility(operator) != visibility:
        return False
    return True


def extract_error_excerpt(text: str, max_lines: int = 8) -> list[str]:
    primary: list[str] = []
    root_causes: list[str] = []
    secondary: list[str] = []
    seen: set[str] = set()
    nonempty_lines: list[str] = []

    for raw_line in text.splitlines():
        line = " ".join(ANSI_ESCAPE_RE.sub("", raw_line).strip().split())
        if not line:
            continue
        nonempty_lines.append(line)
        if any(pattern.search(line) for pattern in PRIMARY_ERROR_LINE_PATTERNS):
            destination = primary
        elif any(pattern.search(line) for pattern in ROOT_CAUSE_LINE_PATTERNS):
            destination = root_causes
        elif any(pattern.search(line) for pattern in SECONDARY_ERROR_LINE_PATTERNS):
            destination = secondary
        else:
            continue
        shortened = line[:500]
        lowered = line.lower()
        error_index = lowered.find("error:")
        if error_index >= 0:
            dedupe_key = lowered[error_index:]
        else:
            dedupe_key = re.sub(
                r"/tmp/tmp[^/\s'\"]+",
                "/tmp/<tmp>",
                lowered,
            )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        destination.append(shortened)

    selected = primary or root_causes or secondary
    if selected:
        return selected[:max_lines]

    for line in nonempty_lines[-max_lines:]:
        shortened = line[:500]
        if shortened not in seen:
            seen.add(shortened)
            selected.append(shortened)
    return selected


def classify_log(text: str, exit_code: int | None) -> tuple[str, str | None, str | None]:
    if exit_code is None:
        return "planned", None, None
    if exit_code == 0:
        return "passed", None, None
    if exit_code == 124:
        return "failed", "timeout", "validation command timed out"

    lowered = text.lower()
    if "no such file or directory" in lowered and "triton-riscv-env.sh" in lowered:
        return "failed", "environment", "environment helper was not found"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "failed", "import", "python import failed"
    if "building wheel for triton" in lowered or "failed building wheel" in lowered:
        return "failed", "build", "triton build or wheel installation failed"
    if "not supported in this architecture" in lowered and "fp8" in lowered:
        return (
            "failed",
            "target-capability",
            "RISC-V target does not support the requested FP8 dtype",
        )
    if "triton.compiler.errors.compilationerror" in lowered:
        if "triton.language.math" in lowered and "has no attribute" in lowered:
            return (
                "failed",
                "triton-frontend",
                "operator uses a Triton language math function unavailable in this Triton version",
            )
        return "failed", "triton-frontend", "Triton frontend compilation failed"
    if "dialect `tptr' not found" in lowered:
        return (
            "failed",
            "buddy-opt",
            "Buddy optimizer could not load the tptr dialect produced by pointer lowering",
        )
    if "dialect `ttx' not found" in lowered:
        return (
            "failed",
            "buddy-opt",
            "Buddy optimizer could not load the ttx dialect produced by Triton-Shared",
        )
    if "unsupported linalg.reduce for -lower-linalg-to-vir" in lowered:
        return (
            "failed",
            "buddy-opt",
            "Buddy VIR lowering does not support the generated linalg.reduce form",
        )
    if "unexpected op in ptr sequence" in lowered:
        return (
            "failed",
            "triton-shared-opt",
            "pointer lowering could not represent an operation in the pointer sequence",
        )
    if "triton-shared-opt" in lowered and "error" in lowered:
        return "failed", "triton-shared-opt", "Triton to MLIR conversion failed"
    if "dialect `linalg' not found" in lowered or "linalg.generic" in lowered:
        return "failed", "mlir-translate", "unsupported linalg operation reached LLVM translation"
    if "buddy-opt" in lowered and "error" in lowered:
        return "failed", "buddy-opt", "Buddy MLIR lowering failed"
    if "mlir-translate" in lowered and "error" in lowered:
        return "failed", "mlir-translate", "MLIR to LLVM IR translation failed"
    if "llc" in lowered and "error" in lowered:
        return "failed", "llc", "LLVM code generation failed"
    if "assert_close" in lowered or "mismatched elements" in lowered:
        return "failed", "correctness", "result differs from PyTorch reference"
    if "subprocess.calledprocesserror" in lowered:
        return "failed", "compilation", "subprocess compilation command failed"
    if "failed" in lowered:
        return "failed", "pytest", "pytest reported failing tests"
    return "failed", "unknown", "nonzero exit code without a known signature"


def run_operator(
    operator: dict,
    *,
    repo_root: Path,
    results_dir: Path,
    source_env: bool,
    dry_run: bool,
    timeout_seconds: int,
) -> OperatorValidationResult:
    command = build_shell_command(operator["validation_command"], source_env)
    start = time.monotonic()
    log_path: Path | None = None
    exit_code: int | None = None
    output = ""

    if not command:
        status = "skipped"
        failure_stage = "selection"
        likely_reason = "operator has no validation command"
        duration = time.monotonic() - start
        return OperatorValidationResult(
            operator=operator["name"],
            implementation_file=operator["implementation_file"],
            test_files=operator["test_files"],
            command=command,
            dry_run=dry_run,
            exit_code=None,
            status=status,
            failure_stage=failure_stage,
            likely_reason=likely_reason,
            error_excerpt=[],
            duration_seconds=round(duration, 3),
            log_path=None,
        )

    if not dry_run:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{timestamp}-{slugify(operator['name'])}.log"
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
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
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            output = stdout + stderr + f"\nTIMEOUT after {timeout_seconds} seconds\n"
            exit_code = 124
        log_path.write_text(output, encoding="utf-8")

    status, failure_stage, likely_reason = classify_log(output, exit_code)
    error_excerpt = extract_error_excerpt(output) if status == "failed" else []
    duration = time.monotonic() - start
    return OperatorValidationResult(
        operator=operator["name"],
        implementation_file=operator["implementation_file"],
        test_files=operator["test_files"],
        command=command,
        dry_run=dry_run,
        exit_code=exit_code,
        status=status,
        failure_stage=failure_stage,
        likely_reason=likely_reason,
        error_excerpt=error_excerpt,
        duration_seconds=round(duration, 3),
        log_path=log_path.as_posix() if log_path else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate operator targets discovered by discover_operators.py."
    )
    parser.add_argument(
        "operator",
        nargs="?",
        help="Operator name, for example silu_and_mul.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--operators",
        default=DEFAULT_OPERATORS.as_posix(),
        help="Operator discovery JSON.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR.as_posix(),
        help="Directory for logs and result JSONL files.",
    )
    parser.add_argument(
        "--source-env",
        action="store_true",
        help="Source scripts/triton-riscv-env.sh before validation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered operators and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all discovered operators, optionally filtered by --contains.",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Filter operator names when using --list or --all.",
    )
    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--public-only",
        action="store_true",
        help="Select public operators whose names do not start with an underscore.",
    )
    visibility_group.add_argument(
        "--internal-only",
        action="store_true",
        help="Select internal operators whose names start with an underscore.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of selected operators.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Append to an existing JSONL batch and skip operators that already "
            "have passed, failed, or skipped results."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    operators_path = Path(args.operators)
    if not operators_path.is_absolute():
        operators_path = repo_root / operators_path
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    visibility = None
    if args.public_only:
        visibility = "public"
    elif args.internal_only:
        visibility = "internal"

    operators = [
        operator
        for operator in load_operators(operators_path)
        if operator_matches(operator, args.contains, visibility)
    ]
    if args.limit is not None:
        operators = operators[: args.limit]

    if args.list:
        for operator in operators:
            print(
                f"{operator['name']}\t"
                f"visibility={operator_visibility(operator)}\t"
                f"tests={len(operator['test_nodes'])}\t"
                f"tl_ops={','.join(operator['tl_ops'])}"
            )
        return 0

    if args.all:
        selected = operators
    elif args.operator:
        selected = [find_operator(load_operators(operators_path), args.operator)]
    else:
        raise SystemExit("pass an operator name, or use --list, or use --all")

    if args.resume_from:
        result_path = Path(args.resume_from)
        if not result_path.is_absolute():
            result_path = repo_root / result_path
        completed_operators = load_completed_operators(result_path)
        original_count = len(selected)
        selected = [
            operator
            for operator in selected
            if operator["name"] not in completed_operators
        ]
        print(
            f"resume: skipped {original_count - len(selected)} completed operators"
        )
        file_mode = "a"
    else:
        if not selected:
            raise SystemExit("no operators matched the selection")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = slugify(selected[0]["name"]) if len(selected) == 1 else "batch"
        result_path = results_dir / f"operator-validation-{timestamp}-{suffix}.jsonl"
        file_mode = "w"

    if not selected:
        print(f"nothing to run; all selected operators are recorded in {result_path}")
        return 0

    result_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with result_path.open(file_mode, encoding="utf-8") as handle:
        for operator in selected:
            result = run_operator(
                operator,
                repo_root=repo_root,
                results_dir=results_dir,
                source_env=args.source_env,
                dry_run=args.dry_run,
                timeout_seconds=args.timeout_seconds,
            )
            result_json = json.dumps(asdict(result), sort_keys=True)
            handle.write(result_json + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            if result.exit_code not in (0, None):
                failures += 1

    try:
        display_path = result_path.relative_to(repo_root)
    except ValueError:
        display_path = result_path
    print(f"wrote {display_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

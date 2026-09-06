#!/usr/bin/env python3
"""Run discovered Triton-RISCV validation targets.

This runner consumes the JSON produced by `discover.py`. It can run a filtered
subset of targets, save per-target logs, and append structured JSONL results for
later failure classification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_TARGETS = Path("agent-results/targets.json")
DEFAULT_RESULTS_DIR = Path("agent-results")


@dataclass
class ValidationResult:
    path: str
    kind: str
    likely_area: str
    command: str
    dry_run: bool
    exit_code: int | None
    duration_seconds: float
    log_path: str | None


def load_targets(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [*data.get("pytest_targets", []), *data.get("lit_targets", [])]


def slugify(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def build_shell_command(command: str, source_env: bool) -> str:
    if not source_env:
        return command
    return f"source scripts/triton-riscv-env.sh && {command}"


def target_matches(
    target: dict,
    *,
    kind: str | None,
    area: str | None,
    path_contains: str | None,
) -> bool:
    if kind and target.get("kind") != kind:
        return False
    if area and target.get("likely_area") != area:
        return False
    if path_contains and path_contains not in target.get("path", ""):
        return False
    return True


def run_target(
    target: dict,
    *,
    repo_root: Path,
    results_dir: Path,
    source_env: bool,
    dry_run: bool,
    timeout_seconds: int,
) -> ValidationResult:
    command = build_shell_command(target["command"], source_env)
    log_path: Path | None = None
    start = time.monotonic()

    if dry_run:
        exit_code = None
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{timestamp}-{slugify(target['path'])}.log"

        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        exit_code = completed.returncode

    duration = time.monotonic() - start
    return ValidationResult(
        path=target["path"],
        kind=target["kind"],
        likely_area=target.get("likely_area", "unknown"),
        command=command,
        dry_run=dry_run,
        exit_code=exit_code,
        duration_seconds=round(duration, 3),
        log_path=log_path.as_posix() if log_path else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Triton-RISCV validation targets discovered by discover.py."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--targets",
        default=DEFAULT_TARGETS.as_posix(),
        help="Discovery JSON produced by codex_agent/discover.py.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR.as_posix(),
        help="Directory for logs and JSONL results.",
    )
    parser.add_argument("--kind", choices=["pytest", "lit"], default=None)
    parser.add_argument("--area", default=None, help="Filter by likely_area.")
    parser.add_argument(
        "--path-contains",
        default=None,
        help="Only run targets whose path contains this text.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source-env",
        action="store_true",
        help="Source scripts/triton-riscv-env.sh before each command.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Per-target timeout. Defaults to 15 minutes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and record planned commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    targets_path = Path(args.targets)
    if not targets_path.is_absolute():
        targets_path = repo_root / targets_path
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        target
        for target in load_targets(targets_path)
        if target_matches(
            target,
            kind=args.kind,
            area=args.area,
            path_contains=args.path_contains,
        )
    ]
    if args.limit is not None:
        targets = targets[: args.limit]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = results_dir / f"validation-{timestamp}.jsonl"

    failures = 0
    with results_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            result = run_target(
                target,
                repo_root=repo_root,
                results_dir=results_dir,
                source_env=args.source_env,
                dry_run=args.dry_run,
                timeout_seconds=args.timeout_seconds,
            )
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            print(
                f"{target['kind']} {target['path']} "
                f"exit={result.exit_code} log={result.log_path}"
            )
            if result.exit_code not in (0, None):
                failures += 1

    print(f"wrote {results_path.relative_to(repo_root)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

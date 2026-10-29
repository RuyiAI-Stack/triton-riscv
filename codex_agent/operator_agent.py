#!/usr/bin/env python3
"""Run the complete autonomous operator discovery and validation workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .discover_operators import discover_operators
from .operator_status import write_status_reports
from .summarize_operator_results import (
    latest_by_operator,
    load_result_files,
    read_results,
    render_markdown,
)
from .validate_operator import run_operator


RETRYABLE_STAGES = {"timeout", "unknown"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_latest_results(results_dir: Path) -> dict[str, dict]:
    results = read_results(load_result_files(results_dir, []))
    return {item["operator"]: item for item in latest_by_operator(results)}


def run_preflight(repo_root: Path, source_env: bool, dry_run: bool) -> dict:
    if dry_run:
        return {
            "status": "skipped",
            "exit_code": None,
            "reason": "dry-run",
            "output": "",
        }
    checks = (
        "python -c 'import triton; print(triton.__version__)' && "
        "command -v triton-shared-opt && command -v buddy-opt"
    )
    command = checks
    if source_env:
        command = f"source scripts/triton-riscv-env.sh && {checks}"
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return {
            "status": "failed",
            "exit_code": 124,
            "reason": "operator environment check timed out",
            "output": output.strip(),
        }
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "reason": None if completed.returncode == 0 else "operator environment check failed",
        "output": completed.stdout.strip(),
    }


def select_operators(
    operators: list[dict],
    latest: dict[str, dict],
    *,
    names: list[str],
    visibility: str,
    selection: str,
    contains: str | None,
    offset: int,
    limit: int | None,
) -> list[dict]:
    by_name = {item["name"]: item for item in operators}
    if names:
        missing = [name for name in names if name not in by_name]
        if missing:
            raise SystemExit(f"unknown operators: {', '.join(missing)}")
        selected = [by_name[name] for name in names]
    else:
        selected = []
        for operator in operators:
            name = operator["name"]
            operator_visibility = operator.get(
                "visibility",
                "internal" if name.startswith("_") else "public",
            )
            if visibility != "all" and operator_visibility != visibility:
                continue
            if contains and contains not in name:
                continue
            previous = latest.get(name)
            if selection == "unvalidated" and previous is not None:
                if previous.get("status") in {"passed", "failed", "skipped"}:
                    continue
            if selection == "failed" and (
                previous is None or previous.get("status") != "failed"
            ):
                continue
            selected.append(operator)

    selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def prune_old_logs(
    results_dir: Path,
    keep_logs: int,
    latest: dict[str, dict],
) -> list[str]:
    if keep_logs <= 0:
        return []
    log_dir = (results_dir / "logs").resolve()
    if not log_dir.exists():
        return []

    protected = {
        Path(item["log_path"]).resolve()
        for item in latest.values()
        if item.get("log_path")
    }
    logs = sorted(
        (
            path
            for path in log_dir.iterdir()
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    protected_logs = {path for path in logs if path.resolve() in protected}
    available_slots = max(keep_logs - len(protected_logs), 0)
    unprotected_logs = [path for path in logs if path not in protected_logs]
    retained = protected_logs | set(unprotected_logs[:available_slots])
    removed: list[str] = []
    for path in logs:
        resolved = path.resolve()
        if path in retained or resolved in protected:
            continue
        if resolved.parent != log_dir:
            continue
        path.unlink()
        removed.append(path.as_posix())
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover operators, validate a safe batch, and regenerate reports."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-dir", default="agent-results")
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument(
        "--visibility",
        choices=("public", "internal", "all"),
        default="public",
    )
    parser.add_argument(
        "--selection",
        choices=("unvalidated", "failed", "all"),
        default="unvalidated",
    )
    parser.add_argument("--contains", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--source-env",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--keep-logs", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-invalid-names", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not 0 <= args.retries <= 3:
        raise SystemExit("--retries must be between 0 and 3")
    if args.keep_logs < 0:
        raise SystemExit("--keep-logs must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = Path(args.repo_root).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = results_dir / "operators.json"
    inventory = discover_operators(
        repo_root,
        include_invalid_names=args.include_invalid_names,
    )
    write_json(inventory_path, inventory)

    preflight = run_preflight(repo_root, args.source_env, args.dry_run)
    latest = load_latest_results(results_dir)
    if preflight["status"] == "failed":
        selected = []
        print("preflight failed; operator validation was not started")
        print(preflight["output"])
    else:
        selected = select_operators(
            inventory["operators"],
            latest,
            names=args.operator,
            visibility=args.visibility,
            selection=args.selection,
            contains=args.contains,
            offset=args.offset,
            limit=args.limit,
        )

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_path = results_dir / f"operator-validation-{timestamp}-agent.jsonl"
    failures = 0
    attempts_written = 0

    if selected:
        with result_path.open("w", encoding="utf-8") as handle:
            for index, operator in enumerate(selected, start=1):
                final_result = None
                for attempt in range(1, args.retries + 2):
                    result = run_operator(
                        operator,
                        repo_root=repo_root,
                        results_dir=results_dir,
                        source_env=args.source_env,
                        dry_run=args.dry_run,
                        timeout_seconds=args.timeout_seconds,
                    )
                    record = asdict(result)
                    record["attempt"] = attempt
                    record["max_attempts"] = args.retries + 1
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    attempts_written += 1
                    final_result = result
                    print(
                        f"[{index}/{len(selected)}] {operator['name']}: "
                        f"{result.status}"
                        + (f" ({result.failure_stage})" if result.failure_stage else ""),
                        flush=True,
                    )
                    if result.status != "failed":
                        break
                    if result.failure_stage not in RETRYABLE_STAGES:
                        break
                    if attempt <= args.retries:
                        print(
                            f"retrying {operator['name']} after {result.failure_stage}",
                            flush=True,
                        )
                if final_result and final_result.status == "failed":
                    failures += 1
    elif preflight["status"] != "failed":
        result_path = None
        print("no operators matched the requested selection")
    else:
        result_path = None

    all_results = read_results(load_result_files(results_dir, []))
    summary_path = results_dir / "operator-summary.md"
    summary_path.write_text(
        render_markdown(all_results, latest_only=True),
        encoding="utf-8",
    )
    status_report = write_status_reports(
        inventory_path,
        results_dir,
        results_dir / "operator-status.json",
        results_dir / "operator-status.md",
    )
    latest = load_latest_results(results_dir)
    removed_logs = prune_old_logs(results_dir, args.keep_logs, latest)

    run_report = {
        "schema_version": 1,
        "preflight": preflight,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection": args.selection,
        "visibility": args.visibility,
        "selected_operators": [item["name"] for item in selected],
        "selected_count": len(selected),
        "attempts_written": attempts_written,
        "failures": failures,
        "result_path": result_path.as_posix() if result_path else None,
        "removed_logs": removed_logs,
        "operator_status_summary": status_report["summary"],
    }
    write_json(results_dir / "operator-agent-last-run.json", run_report)

    print(json.dumps(run_report, indent=2, sort_keys=True))
    if preflight["status"] == "failed":
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

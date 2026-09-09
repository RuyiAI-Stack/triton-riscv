#!/usr/bin/env python3
"""Build a complete validation-status report for discovered operators."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .summarize_operator_results import (
    latest_by_operator,
    load_result_files,
    read_results,
)


DEFAULT_INVENTORY = Path("agent-results/operators.json")
DEFAULT_RESULTS_DIR = Path("agent-results")


def load_inventory(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def operator_status(operator: dict, result: dict | None) -> tuple[str, str]:
    if result is not None:
        status = result.get("status", "unknown")
        if status == "passed":
            return status, "basic-tests-passed"
        if status == "failed":
            return status, "basic-tests-failed"
        if status == "skipped":
            return status, "not-runnable"
        return status, "not-established"
    if not operator.get("validation_command"):
        return "unverified", "not-established"
    return "unvalidated", "not-established"


def build_status_report(inventory: dict, results: list[dict]) -> dict:
    latest = {item["operator"]: item for item in latest_by_operator(results)}
    operators: list[dict] = []
    for operator in inventory.get("operators", []):
        result = latest.get(operator["name"])
        status, correctness = operator_status(operator, result)
        operators.append(
            {
                "name": operator["name"],
                "visibility": operator.get(
                    "visibility",
                    "internal" if operator["name"].startswith("_") else "public",
                ),
                "implementation_file": operator["implementation_file"],
                "test_files": operator.get("test_files", []),
                "test_nodes": operator.get("test_nodes", []),
                "tl_ops": operator.get("tl_ops", []),
                "risk_hints": operator.get("risk_hints", []),
                "status": status,
                "correctness": correctness,
                "failure_stage": result.get("failure_stage") if result else None,
                "likely_reason": result.get("likely_reason") if result else None,
                "error_excerpt": result.get("error_excerpt", []) if result else [],
                "duration_seconds": result.get("duration_seconds") if result else None,
                "log_path": result.get("log_path") if result else None,
            }
        )

    status_counts = Counter(item["status"] for item in operators)
    public_status_counts = Counter(
        item["status"] for item in operators if item["visibility"] == "public"
    )
    validated = status_counts.get("passed", 0) + status_counts.get("failed", 0)
    public_validated = public_status_counts.get("passed", 0) + public_status_counts.get(
        "failed", 0
    )
    total = len(operators)
    return {
        "schema_version": 1,
        "summary": {
            "operators": total,
            "public_operators": sum(
                1 for item in operators if item["visibility"] == "public"
            ),
            "internal_operators": sum(
                1 for item in operators if item["visibility"] == "internal"
            ),
            "validated": validated,
            "validation_percent": round((validated / total * 100), 2) if total else 0.0,
            "passed": status_counts.get("passed", 0),
            "failed": status_counts.get("failed", 0),
            "public_validated": public_validated,
            "public_passed": public_status_counts.get("passed", 0),
            "public_failed": public_status_counts.get("failed", 0),
            "skipped": status_counts.get("skipped", 0),
            "planned": status_counts.get("planned", 0),
            "unvalidated": status_counts.get("unvalidated", 0),
            "unverified": status_counts.get("unverified", 0),
        },
        "operators": operators,
    }


def markdown_escape(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_status_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Operator Validation Status",
        "",
        "A passed status means that the mapped basic tests passed in the recorded",
        "environment. It does not prove correctness for inputs not covered by those tests.",
        "",
        f"- Operators discovered: {summary['operators']}",
        f"- Public operators: {summary['public_operators']}",
        f"- Internal operators: {summary['internal_operators']}",
        f"- Validated: {summary['validated']} ({summary['validation_percent']}%)",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Public validated: {summary['public_validated']}",
        f"- Public passed: {summary['public_passed']}",
        f"- Public failed: {summary['public_failed']}",
        f"- Skipped: {summary['skipped']}",
        f"- Unvalidated: {summary['unvalidated']}",
        f"- Unverified (no command): {summary['unverified']}",
        "",
        "## Operators",
        "",
        "| Operator | Visibility | Tests | Status | Failure Stage | Reason |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item in sorted(report["operators"], key=lambda value: value["name"]):
        lines.append(
            "| {name} | {visibility} | {tests} | {status} | {stage} | {reason} |".format(
                name=markdown_escape(item["name"]),
                visibility=markdown_escape(item["visibility"]),
                tests=len(item["test_nodes"]),
                status=markdown_escape(item["status"]),
                stage=markdown_escape(item["failure_stage"]),
                reason=markdown_escape(item["likely_reason"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_status_reports(
    inventory_path: Path,
    results_dir: Path,
    json_output: Path,
    markdown_output: Path,
) -> dict:
    inventory = load_inventory(inventory_path)
    result_paths = load_result_files(results_dir, [])
    results = read_results(result_paths)
    report = build_status_report(inventory, results)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_status_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report every operator's validation status.")
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY.as_posix())
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR.as_posix())
    parser.add_argument(
        "--json-output",
        default="agent-results/operator-status.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="agent-results/operator-status.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_status_reports(
        Path(args.inventory),
        Path(args.results_dir),
        Path(args.json_output),
        Path(args.markdown_output),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge actual board records only when case and frozen BIN hash both match."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def candidate(path, case):
    try:
        row = load(path)
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("case") != case:
        return None
    final = row.get("final")
    if not isinstance(final, dict):
        return None
    return row


def actual_success(row, expected_sha):
    final = row["final"]
    return (
        row.get("image_sha256") == expected_sha
        and final.get("image_sha256", expected_sha) == expected_sha
        and row.get("server", final.get("server")) == "fpga112"
        and final.get("state") == "completed"
        and final.get("execution_status") == "exited"
        and final.get("app_exit_code") == 0
        and final.get("launch_returned") is True
        and isinstance(final.get("launch_cycles"), int)
        and final["launch_cycles"] > 0
        and final.get("image_hash_match") is True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--board",
        type=Path,
        action="append",
        required=True,
        help="directory containing <case>/summary.json; later directories take precedence",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptable-seconds", type=int, default=60)
    args = parser.parse_args()
    manifest = load(args.manifest)
    if (
        manifest.get("status") != "READY"
        or manifest.get("formal_run_allowed") is not True
    ):
        raise ValueError("expected a frozen READY 99-case manifest")
    cases = manifest.get("cases", [])
    if len(cases) != 99 or len({case["case"] for case in cases}) != 99:
        raise ValueError("manifest must contain99 unique operator cases")
    records = []
    for case in cases:
        attempts = []
        for board_index, board in enumerate(args.board):
            path = board.resolve() / case["case"] / "summary.json"
            row = candidate(path, case["case"]) if path.is_file() else None
            if row is not None:
                attempts.append(
                    {
                        "source": str(path),
                        "row": row,
                        "board_index": board_index,
                        "exact_bin": row.get("image_sha256") == case["image_sha256"],
                    }
                )
        matching = [
            item
            for item in attempts
            if item["exact_bin"] and actual_success(item["row"], case["image_sha256"])
        ]
        matching.sort(
            key=lambda item: (
                item["board_index"],
                item["row"]["final"].get("client_finished_at_utc", ""),
                item["row"]["final"].get("task_id", ""),
            )
        )
        if matching:
            selected = matching[-1]
            final = selected["row"]["final"]
            seconds = final["launch_cycles"] / 5529600.0
            classification = (
                "OVER_60S_HANG_THRESHOLD_RETURNED"
                if seconds > args.acceptable_seconds
                else "RETURNED_WITHIN_60S"
            )
        else:
            selected = None
            final = {}
            stale = [item for item in attempts if not item["exact_bin"]]
            same_hash_failures = [item for item in attempts if item["exact_bin"]]
            if same_hash_failures:
                latest = same_hash_failures[-1]["row"]["final"]
                final = latest
                if latest.get("execution_status") == "timeout":
                    classification = "HANG_UNRESOLVED_BY_DEADLINE"
                elif (
                    latest.get("execution_status") == "platform_error"
                    and latest.get("application_started") is False
                ):
                    classification = "PLATFORM_FAILURE_NOT_EXECUTED"
                else:
                    classification = "EXECUTION_OR_EVIDENCE_FAILURE"
            elif stale:
                classification = "STALE_BIN_ONLY"
            else:
                classification = "PENDING"
        records.append(
            {
                "case": case["case"],
                "family": case["family"],
                "image_sha256": case["image_sha256"],
                "classification": classification,
                "launch_cycles": final.get("launch_cycles"),
                "launch_seconds": (
                    final["launch_cycles"] / 5529600.0
                    if isinstance(final.get("launch_cycles"), int)
                    else None
                ),
                "scheduler_seconds": final.get("scheduler_seconds"),
                "task_id": final.get("task_id"),
                "fpga": final.get("fpga"),
                "execution_status": final.get("execution_status"),
                "app_exit_code": final.get("app_exit_code"),
                "source": selected["source"] if selected else None,
                "exact_success_records": len(matching),
                "stale_hash_records": sum(not item["exact_bin"] for item in attempts),
                "numerical_current_full_pass": case.get("numerical_current_full_pass"),
                "numerical_evidence": case.get("numerical_evidence"),
            }
        )
    classifications = {}
    for row in records:
        classifications[row["classification"]] = (
            classifications.get(row["classification"], 0) + 1
        )
    report = {
        "schema": 1,
        "server": "fpga112",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "unique_operator_cases": len(records),
        "acceptable_seconds": args.acceptable_seconds,
        "classification_counts": classifications,
        "exact_returned_cycles": sum(
            row["launch_cycles"] is not None and row["execution_status"] == "exited"
            for row in records
        ),
        "current_full_numerical_pass_cases": sum(
            row.get("numerical_current_full_pass") is True for row in records
        ),
        "complete": all(
            row["classification"]
            in ("RETURNED_WITHIN_60S", "OVER_60S_HANG_THRESHOLD_RETURNED")
            for row in records
        ),
        "scope": "one exact-BIN fpga112 completed/exited0 launch record per operator; 60s labels user hang threshold, not proof of permanent deadlock",
        "cases": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    fields = [
        "case",
        "family",
        "classification",
        "launch_cycles",
        "launch_seconds",
        "scheduler_seconds",
        "task_id",
        "fpga",
        "image_sha256",
        "execution_status",
        "app_exit_code",
        "exact_success_records",
        "stale_hash_records",
        "source",
    ]
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in records)
    lines = [
        "# EfficientNet-B3 exact-image FPGA112 results",
        "",
        f"Unique operator cases: {len(records)}. Returned cycles: {report['exact_returned_cycles']}/99.",
        "A result over60s meets the user hang threshold but did return. Scheduler deadlines without a returned launch counter remain unresolved.",
        "",
        "| Case | Classification | Cycles | Seconds | Task / FPGA |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in records:
        cycles = str(row["launch_cycles"]) if row["launch_cycles"] is not None else "—"
        seconds = (
            f"{row['launch_seconds']:.3f}" if row["launch_seconds"] is not None else "—"
        )
        lines.append(
            f"| `{row['case']}` | {row['classification']} | {cycles} | {seconds} | {row['task_id'] or '—'} / {row['fpga'] or '—'} |"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "unique_operator_cases",
                    "exact_returned_cycles",
                    "classification_counts",
                    "current_full_numerical_pass_cases",
                    "complete",
                )
            },
            indent=2,
        )
    )
    return int(not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())

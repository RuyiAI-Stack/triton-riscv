#!/usr/bin/env python3
"""Summarize actual frozen suite launches; never infer cycles for unrun cases."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    numerical = {}
    for path in args.numerical:
        report = json.loads(path.read_text())
        for row in report.get("cases", []):
            numerical[row.get("name", row.get("case"))] = {
                "status": row["status"],
                "evidence": str(path),
                "programs": row.get("programs"),
                "total_programs": row.get("total_programs"),
                "outputs": row.get("outputs"),
                "max_abs": row.get("max_abs"),
                "collected_existing_actual_evidence": row.get(
                    "suite_reused_authoritative_collect", False
                ),
            }
    rows = []
    for case in manifest["cases"]:
        name = case["case"]
        path = args.board / name / "summary.json"
        final = {}
        attempts = []
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("image_sha256") != case["image_sha256"]:
                raise RuntimeError(
                    "board evidence image differs from frozen suite: " + name
                )
            final = saved.get("final", {})
            attempts = saved.get("attempts", [])
        if not final:
            classification = "PENDING"
        elif final.get("protocol_failure"):
            classification = "PROTOCOL_FAILURE"
        elif final.get("timing_passed"):
            classification = (
                "OVER_60S_RETURNED"
                if final.get("exceeds_acceptable_60s_by_cycles")
                else "RETURNED_WITHIN_60S"
            )
        elif final.get("execution_status") == "timeout":
            classification = "DEADLINE_TIMEOUT_UNRESOLVED"
        elif (
            final.get("execution_status") == "platform_error"
            and final.get("application_started") is False
        ):
            classification = "PLATFORM_FAILURE_NOT_EXECUTED"
        else:
            classification = "EXECUTION_OR_EVIDENCE_FAILURE"
        rows.append(
            {
                "case": name,
                "family": case["family"],
                "classification": classification,
                "launch_cycles": final.get("launch_cycles"),
                "launch_seconds": final.get("launch_seconds_at_5529600hz"),
                "scheduler_seconds": final.get("scheduler_seconds"),
                "execution_status": final.get("execution_status"),
                "state": final.get("state"),
                "app_exit_code": final.get("app_exit_code"),
                "exit_code": final.get("exit_code"),
                "task_id": final.get("task_id"),
                "fpga": final.get("fpga"),
                "image_sha256": case["image_sha256"],
                "attempts": len(attempts),
                "had_60s_timeout": any(
                    a.get("timeout_seconds") == 60
                    and a.get("execution_status") == "timeout"
                    for a in attempts
                ),
                "numerical": numerical.get(name),
                "evidence": str(path) if path.is_file() else None,
            }
        )
    counts = dict(Counter(row["classification"] for row in rows))
    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": "fpga112",
        "manifest": str(args.manifest),
        "cases": len(rows),
        "classifications": counts,
        "actual_returned_cycles": sum(row["launch_cycles"] is not None for row in rows),
        "numerical_pass_cases": sum(
            (row["numerical"] or {}).get("status") == "PASS" for row in rows
        ),
        "complete": all(
            row["classification"] in ("RETURNED_WITHIN_60S", "OVER_60S_RETURNED")
            for row in rows
        ),
        "timing_scope": "full C launch including fixture setup adapter and launcher postprocessing",
        "threshold_note": "60 seconds is the user performance threshold. A returned over-threshold case is slow; a timeout alone does not prove permanent hang.",
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
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
        "attempts",
        "had_60s_timeout",
    ]
    with (args.output / "report.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    lines = [
        "# EfficientNet-B3 actual operator suite",
        "",
        "Server: fpga112. Full C launch cycles use 5,529,600 Hz.",
        "",
        "Returned cycles: "
        + str(report["actual_returned_cycles"])
        + "/"
        + str(len(rows))
        + ".",
        "Numerical evidence is distinct from board completion. Timeouts do not establish permanent hangs.",
        "",
        "| Case | Classification | Cycles | Launch seconds | Task / FPGA |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        seconds = (
            format(row["launch_seconds"], ".6f")
            if row["launch_seconds"] is not None
            else "—"
        )
        lines.append(
            "| `"
            + row["case"]
            + "` | "
            + row["classification"]
            + " | "
            + str(row["launch_cycles"] if row["launch_cycles"] is not None else "—")
            + " | "
            + seconds
            + " | "
            + str(row["task_id"] or "—")
            + " / "
            + str(row["fpga"] or "—")
            + " |"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "cases",
                    "actual_returned_cycles",
                    "classifications",
                    "numerical_pass_cases",
                    "complete",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

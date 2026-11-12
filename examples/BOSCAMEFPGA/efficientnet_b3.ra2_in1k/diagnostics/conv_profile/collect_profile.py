#!/usr/bin/env python3
"""Summarize board phase measurements; keep sampled and full runs distinct."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    rows = []
    for path in sorted(root.glob("**/summary.json")):
        if path.parent.name.startswith("attempt-"):
            continue
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or "profile" not in record:
            continue
        evidence = Path(record["evidence_directory"])
        manifest_path = evidence / "profile-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        task = json.loads((evidence / "task.json").read_text())
        details = (task.get("result") or {}).get("details") or {}
        assert task["id"] == record["task_id"]
        assert details.get("image_sha256") == record["image_sha256"]
        assert digest(Path(record["image_snapshot"])) == record["image_sha256"]
        profile = record["profile"]
        stages = profile.get("stages", {})
        successful = (
            record.get("state") == "completed"
            and record.get("execution_status") == "exited"
            and record.get("app_exit_code") == 0
            and record.get("exit_code") == 0
            and record.get("profile_counter_check_passed")
        )
        full_calls = manifest.get("original_full_grid_stage_calls", {})
        extrapolated = {}
        for name, value in stages.items():
            if value.get("calls") and name in full_calls:
                extrapolated[name] = value["cycles"] / value["calls"] * full_calls[name]
        rows.append(
            {
                "case": record["case"],
                "mode": record["mode"],
                "selected_stage": record["selected_stage"],
                "coverage_set": str(path.relative_to(root).parts[0]),
                "coverage": manifest.get("coverage"),
                "selected_tiles": manifest.get("selected_tiles"),
                "full_grid_tiles": manifest.get("full_grid_tiles"),
                "successful": bool(successful),
                "state": record.get("state"),
                "execution_status": record.get("execution_status"),
                "fpga": record.get("fpga"),
                "task_id": record["task_id"],
                "image_sha256": record["image_sha256"],
                "manifest_sha256": digest(manifest_path),
                "fixture": manifest.get("fixture"),
                "stages": stages,
                "adapter_total_cycles": profile.get("adapter_total_cycles"),
                "summed_stage_cycles": profile.get("summed_stage_cycles"),
                "adapter_minus_stage_cycles": profile.get("adapter_minus_stage_cycles"),
                "launch_cycles": profile.get("launch_cycles"),
                "timer_pair_mean_cycles": profile.get("timer_pair_mean_cycles"),
                "full_stage_cycles_estimate": extrapolated,
                "estimate_semantics": "mean measured raw call cycles times original full call count; excludes full loop cost, not a full-grid measurement",
                "evidence": str(path),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.runs.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"runs": rows}, indent=2) + "\n"
    )
    fields = [
        "case",
        "mode",
        "selected_stage",
        "coverage_set",
        "selected_tiles",
        "full_grid_tiles",
        "successful",
        "stage",
        "calls",
        "cycles",
        "mean_cycles_per_call",
        "adapter_total_cycles",
        "task_id",
        "fpga",
    ]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for stage, data in row["stages"].items():
                writer.writerow({**row, **data, "stage": stage})
    print(
        json.dumps(
            {"runs": len(rows), "successful": sum(row["successful"] for row in rows)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

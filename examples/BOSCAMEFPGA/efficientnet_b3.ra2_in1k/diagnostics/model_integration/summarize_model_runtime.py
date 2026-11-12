#!/usr/bin/env python3
"""Group independently audited full-model DDR call counters by implementation."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREQUENCY_HZ = 5_529_600


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--plan-dir", type=Path, default=ROOT / "build/model-integration/physical-plan"
    )
    args = parser.parse_args()
    run = args.run_dir.resolve()
    audit_path = run / "independent-board-audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("execution_pass"):
        raise ValueError("requires a completed full-model execution integrity audit")
    manifest = json.loads((run / "image-manifest.json").read_text())
    plan_path = args.plan_dir / "physical-plan.json"
    if sha(plan_path) != manifest["plan_sha256"]:
        raise ValueError("physical plan differs from the audited image")
    plan = json.loads(plan_path.read_text())
    with (run / "audited-call-cycles.csv").open() as stream:
        calls = list(csv.DictReader(stream))
    if len(calls) != 503 or len(plan["calls"]) != 503:
        raise ValueError("expected 503 physical call counters")
    families = {
        item["case"]: item["family"] for item in manifest["existing_case_inputs"]
    }
    connectors = {spec["name"]: spec["kernel"] for spec in plan["connectors"]}
    grouped = defaultdict(list)
    for row, call in zip(calls, plan["calls"]):
        if int(row["call"]) != call["index"] + 1 or row["case"] != call["target"]:
            raise ValueError("call counter order differs from the physical plan")
        family = (
            families[call["target"]]
            if call["target_kind"] == "existing_case"
            else "connector:" + connectors[call["target"]]
        )
        grouped[family].append(int(row["cycles"]))
    total = sum(sum(values) for values in grouped.values())
    if total != audit["sum_call_cycles"]:
        raise ValueError("grouped counters differ from the independent audit")
    groups = []
    for family, values in grouped.items():
        cycles = sum(values)
        groups.append(
            {
                "family": family,
                "calls": len(values),
                "cycles": cycles,
                "seconds": cycles / FREQUENCY_HZ,
                "fraction_of_call_cycles": cycles / total,
                "max_call_cycles": max(values),
                "calls_over_60_seconds": sum(x > FREQUENCY_HZ * 60 for x in values),
            }
        )
    groups.sort(key=lambda item: item["cycles"], reverse=True)
    report = {
        "task_id": audit["task_id"],
        "server": audit["server"],
        "fpga": audit["fpga"],
        "scope": "503 executed physical calls, grouped by final downloaded counters",
        "timing_source": "independently audited DDR counters; excludes launcher initialization",
        "frequency_hz": FREQUENCY_HZ,
        "independent_audit_sha256": sha(audit_path),
        "physical_plan_sha256": sha(plan_path),
        "physical_calls": len(calls),
        "sum_call_cycles": total,
        "sum_call_seconds": total / FREQUENCY_HZ,
        "model_cycles": audit["model_cycles"],
        "model_seconds": audit["model_cycles"] / FREQUENCY_HZ,
        "trace_hash_and_other_overhead_cycles": audit["model_cycles"] - total,
        "groups": groups,
        "execution_pass": audit["execution_pass"],
        "selected_reference_pass": audit["selected_reference_pass"],
        "primary_torch_reference_pass": audit["primary_torch_reference_pass"],
        "slowest_calls": audit["slowest_calls"],
    }
    output = run / "runtime-breakdown.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read saved compilation, ABI, strict numerical and board evidence for all cases.

This command only aggregates existing evidence. It never builds, simulates,
submits a board task, or executes a complete model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "build/attention-primitive-model-20261029"
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402
from collect_quantized_attention import (  # noqa: E402
    builds,
    evidence,
    normalize,
    read,
    rel,
    sha,
    source_inventory,
    verify_hashes,
)


def abi(case, kernels, directory):
    record = read(directory / case["name"] / "result.json")
    valid = (
        record.get("status") == "ABI_SMOKE_PASS"
        and record.get("launch_sha256") == sha(case["launch_source"])
        and record.get("adapter_sha256") == sha(kernels / case["name"] / "adapter.c")
        and record.get("mock_sha256") == sha(directory / case["name"] / "mock.c")
        and record.get("numeric_oracle_executed") is False
        and record.get("simulator_executed") is False
    )
    valid &= (
        record.get("programs")
        == record.get("expected_programs")
        == math.prod(case["grid"])
    )
    return {
        "status": "PASS" if valid else "INCOMPLETE",
        "evidence": evidence(directory / case["name"] / "result.json"),
        "scope": "native descriptor/grid smoke; no candidate numerical execution",
        "saved_scope": record.get("scope"),
        "programs": record.get("programs"),
    }


def simulation(case, kernels):
    try:
        if case["family"] == "patch_embed":
            from check_patch_chain import collect_patch_case

            record = collect_patch_case(case, kernels)
        else:
            from simcheck import collect_case

            record = collect_case(case, kernels)
    except Exception as error:
        record = {"status": "INCOMPLETE", "reason": f"{type(error).__name__}: {error}"}
    return {
        "status": record.get("status"),
        "reason": record.get("reason"),
        "evidence": evidence(kernels / case["name"] / "simcheck-result.json"),
        "execution": record.get("execution", "strict emitted IR"),
        "programs": record.get("programs", 0),
        "outputs": record.get("outputs", 0),
        "comparison_rounds": record.get("comparison_rounds"),
        "max_abs": record.get("max_abs"),
        "module_sha256": record.get("module_sha256"),
        "runs": [
            {
                key: run.get(key)
                for key in (
                    "name",
                    "order",
                    "programs",
                    "outputs",
                    "output_sha256",
                    "reference_sha256",
                    "fp32_comparison",
                    "guard_errors",
                    "input_errors",
                )
                if key in run
            }
            for run in record.get("runs", [])
        ],
    }


def board(case, base, blocked, forbidden=False):
    name = case["name"]
    directory = base / "board-runs" / name
    saved = read(directory / "result.json")
    task = read(directory / "task.json")
    result = task.get("result") or {}
    details = result.get("details") or {}
    uart = result.get("uart_output")
    if uart is None:
        uart = result.get("output", "")
    cycles = [
        int(value, 16) for value in re.findall(r"kernel_cycles=0x([0-9a-fA-F]+)", uart)
    ]
    match = re.search(
        r"kernel_returned=(\d+) validation_executed=(\d+) checked=(\d+) errors=(\d+)",
        uart,
    )
    counts = (
        dict(
            zip(
                ("returned", "validation_executed", "checked", "errors"),
                map(int, match.groups()),
            )
        )
        if match
        else {}
    )
    config = saved.get("board_config") or read(directory / "board_config.json")
    actual_sha = sha(directory / (name + ".bin"))
    image_ok = bool(
        actual_sha
        and actual_sha == saved.get("image_sha256") == details.get("image_sha256")
    )
    mismatches = verify_hashes(
        directory / "build-evidence", saved.get("build_evidence_sha256", {})
    )
    execution = result.get("execution_status")
    valid = (
        execution == "exited"
        and result.get("app_exit_code") == 0
        and image_ok
        and not mismatches
        and counts.get("validation_executed") == 1
        and counts.get("errors") == 0
        and counts.get("checked") == config.get("expected_checked")
        and counts.get("returned") == config.get("expected_invocations") == len(cycles)
        and details.get("reset_confirmed") is True
        and details.get("application_started") is True
        and details.get("ddr_readback_matches") is True
        and task.get("id") == saved.get("task_id")
    )
    status = "PASS" if valid else "INCOMPLETE"
    if execution == "timeout":
        status = "TIMEOUT" if uart.strip() else "HANG"
    elif execution == "trap":
        status = "TRAP"
    elif execution == "exited" and result.get("app_exit_code") != 0:
        status = "FAIL"
    elif execution == "cancelled":
        status = (
            "BLOCKED_PLATFORM_CANCELLED_BEFORE_START"
            if blocked and task.get("started") is None
            else "CANCELLED"
        )
    elif not task:
        status = (
            "NOT_RUN_USER_FPGA_BROKEN"
            if forbidden
            else "BLOCKED_PLATFORM_NOT_SUBMITTED"
            if blocked
            else "NOT_SUBMITTED"
        )
    return {
        "status": status,
        "result": evidence(directory / "result.json"),
        "task": evidence(directory / "task.json"),
        "task_id": task.get("id", saved.get("task_id")),
        "execution_status": execution,
        "kernel_cycles": cycles,
        "checks": counts,
        "image_sha256": actual_sha,
        "task_image_sha256": details.get("image_sha256"),
        "image_hashes_match": image_ok,
        "build_hash_mismatches": mismatches,
        "node": task.get("node"),
        "fpga": task.get("fpga"),
        "cycle_scope": config.get("cycle_scope"),
    }


def prepared_images(base, names):
    directory = base / "local-images"
    path = directory / "manifest.json"
    manifest = read(path)
    entries = {entry["case"]: entry for entry in manifest.get("entries", [])}
    records = {}
    for name in names:
        entry = entries.get(name, {})
        local = directory / name
        mismatches = verify_hashes(local, entry.get("files_sha256", {}))
        audit = read(local / "elf-audit.json")
        valid = (
            bool(entry.get("files_sha256"))
            and not mismatches
            and audit.get("status") == "PASS"
            and audit.get("elf_sha256") == sha(local / (name + ".elf"))
            and entry.get("files_sha256", {}).get(name + ".bin")
            == sha(local / (name + ".bin"))
        )
        records[name] = {
            "status": "PASS" if valid else "INCOMPLETE",
            "elf_audit": evidence(local / "elf-audit.json"),
            "image": evidence(local / (name + ".bin")),
            "hash_mismatches": mismatches,
            "ame_instructions": audit.get("ame_instructions"),
            "scope": "frozen image and ELF audit only; no FPGA execution implied",
        }
    return {
        "manifest": evidence(path),
        "status": "PASS"
        if all(r["status"] == "PASS" for r in records.values())
        else "INCOMPLETE",
        "passed": sum(r["status"] == "PASS" for r in records.values()),
        "required": len(names),
        "cases": records,
    }


def markdown(report):
    lines = [
        "# All operator validation",
        "",
        f"Status: **{report['status']}**. Scope: independent operator validation; no complete model execution.",
        "",
        f"Inventory {report['inventory']['case_count']}; compile {report['counts']['compile_pass']}; "
        f"ABI {report['counts']['abi_pass']}; ELF/frozen image {report['counts']['prepared_image_pass']}; strict numerical {report['counts']['numerical_pass']}; "
        f"FPGA {report['counts']['board_pass']}/{report['required_cases']}.",
        "",
        "| Operator | Compile | ABI | ELF/image | Strict numerical | Outputs checked | FPGA | Cycles |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['name']} | {case['compile']['status']} | {case['abi']['status']} | {case['prepared_image']['status']} | "
            f"{case['numerical']['status']} | {case['numerical']['outputs']} | {case['board']['status']} | "
            f"{', '.join(map(str, case['board']['kernel_cycles'])) or '—'} |"
        )
    lines += [
        "",
        "Numerical output counts include every retained stress round. ABI smoke uses a mock kernel "
        "and establishes no numerical correctness. Patch embedding executes the actual generated C adapter "
        "natively with strict emitted-IR callbacks for im2col and four child kernels; its bias/layout run in that C adapter.",
        "",
        "Quantized-oracle equality and comparison with the host FP32 baseline remain separate. "
        "No tolerance is changed by this collector. FPGA timeout is the 120-second execution deadline; "
        "queue wait and unstarted platform-blocked tasks are not kernel TIMEOUT/HANG.",
        "",
    ]
    if report["platform"].get("reason"):
        lines += ["Platform blocker: " + report["platform"]["reason"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=DEFAULT_REPORT_DIR / "kernels")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    kernels, base = args.kernels.resolve(), args.report_dir.resolve()
    os.environ.setdefault("BUDDY_ROOT", str(ROOT.parents[2] / "buddy-mlir"))
    os.environ.setdefault(
        "BUDDY_OPT", str(ROOT.parents[2] / "buddy-mlir/build/bin/buddy-opt")
    )
    cases = inventory()
    build = builds(cases, kernels)
    build_cases = {case["case"]: case for case in build["cases"]}
    platform_path = base / "board-platform-final.json"
    platform = read(platform_path) or {
        "status": "NOT_RUN_USER_FPGA_BROKEN",
        "reason": "User reports FPGA broken and forbids board connection and execution.",
    }
    blocked = platform.get("status") == "BLOCKED_PLATFORM"
    forbidden = platform.get("status") == "NOT_RUN_USER_FPGA_BROKEN"
    prepared = prepared_images(base, [case["name"] for case in cases])
    records = [
        {
            "name": case["name"],
            "family": case["family"],
            "compile": build_cases[case["name"]],
            "abi": abi(case, kernels, base / "launchcheck"),
            "prepared_image": prepared["cases"][case["name"]],
            "numerical": simulation(case, kernels),
            "board": board(case, base, blocked, forbidden),
        }
        for case in cases
    ]
    counts = {
        label + "_pass": sum(case[key]["status"] == "PASS" for case in records)
        for label, key in (
            ("compile", "compile"),
            ("abi", "abi"),
            ("numerical", "numerical"),
            ("board", "board"),
        )
    }
    counts["prepared_image_pass"] = prepared["passed"]
    inventory_record = source_inventory(cases)
    complete = inventory_record["status"] == "PASS" and all(
        counts[key] == len(cases)
        for key in ("compile_pass", "abi_pass", "numerical_pass", "prepared_image_pass")
    )
    status = (
        "OPERATOR_CHECKS_PASS_FPGA_NOT_RUN"
        if complete and forbidden
        else "OPERATOR_CHECKS_PASS_BOARD_BLOCKED"
        if complete and blocked
        else "OPERATOR_CHECKS_PASS_BOARD_INCOMPLETE"
        if complete and counts["board_pass"] != len(cases)
        else "ALL_OPERATOR_EVIDENCE_COMPLETE"
        if complete
        else "INCOMPLETE"
    )
    report = normalize(
        {
            "schema": 1,
            "status": status,
            "scope": "per-operator validation only",
            "full_model_executed": False,
            "required_cases": len(cases),
            "kernels": rel(kernels),
            "collector_source": evidence(Path(__file__)),
            "inventory": inventory_record,
            "prepared_board_manifest": prepared["manifest"],
            "counts": counts,
            "total_strict_programs": sum(c["numerical"]["programs"] for c in records),
            "total_strict_outputs_checked": sum(
                c["numerical"]["outputs"] for c in records
            ),
            "board_status_counts": dict(
                sorted(Counter(c["board"]["status"] for c in records).items())
            ),
            "execution_timeout_seconds": 120,
            "platform": {
                "evidence": evidence(platform_path),
                "status": platform.get("status"),
                "reason": platform.get("reason"),
            },
            "cases": records,
        }
    )
    base.mkdir(parents=True, exist_ok=True)
    (base / "all-operators-validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    table = markdown(report)
    (base / "all-operators-validation.md").write_text(table + "\n")
    print(table)
    # Board unavailability remains explicit, while successful collection of
    # complete host/operator evidence can be used as an offline Make check.
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

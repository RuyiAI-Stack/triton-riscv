#!/usr/bin/env python3
"""Collect saved standalone primitive evidence without running candidates.

Compilation, numerical execution, launcher ABI and prepared images have separate
statuses. This collector checks saved evidence and makes no model or FPGA
execution claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "build/attention-primitive-model-20261029"
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402

RETIRED = (
    "linear_196x768x768",
    "linear_197x768x768",
    "linear_197x3072x768",
    "linear_197x768x3072",
    "linear_1x1000x768",
    "attention_qk_12x197x197x64",
    "attention_pv_12x197x64x197",
)
PRIMITIVES = (
    "quantize_197x64",
    "quantize_197x197",
    "quantize_transpose_64x197",
    "quantize_transpose_197x64",
    "matmul_i8_197x197x64",
    "matmul_i8_197x64x197",
    "dequantize_197x197",
    "dequantize_197x64",
    "pack_heads_12x197x197",
    "unpack_heads_12x197x197",
)
BUILD_ARTIFACTS = {
    "kernel.ttir",
    "kernel.frontend.mlir",
    "kernel.bufferized.mlir",
    "kernel.planned.mlir",
    "kernel.boscame.mlir",
    "kernel.llvm.mlir",
    "kernel.ll",
    "kernel.s",
    "kernel.encoded.s",
    "kernel.nr.s",
    "kernel.o",
}


def rel(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        return {"status": "INVALID_EVIDENCE", "reason": str(error)}


def evidence(path):
    return {"path": rel(path), "sha256": sha(path), "exists": Path(path).is_file()}


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    return value


def verify_hashes(directory, entries):
    return [
        {
            "file": rel(directory / name),
            "expected_sha256": expected,
            "actual_sha256": sha(directory / name),
        }
        for name, expected in sorted(entries.items())
        if sha(directory / name) != expected
    ]


def source_inventory(cases):
    names = {case["name"] for case in cases}
    missing = [
        {"case": c["name"], "dependency": dep}
        for c in cases
        for dep in c.get("dependencies", [])
        if dep not in names
    ]
    remaining = [name for name in RETIRED if (ROOT / name).exists()]
    required_missing = sorted(set(PRIMITIVES) - names)
    prohibited = [
        c["name"]
        for c in cases
        if c["name"] in RETIRED
        or c.get("composition") == "w8a8_attention"
        or c["family"] in ("attention_qk", "attention_pv")
    ]
    return {
        "status": "PASS"
        if not (missing or remaining or required_missing or prohibited)
        else "FAIL",
        "case_count": len(cases),
        "cases": sorted(names),
        "retired_executables": list(RETIRED),
        "retired_executable_directories_absent": not remaining,
        "retired_executable_directories_remaining": remaining,
        "missing_required_primitives": required_missing,
        "prohibited_attention_executables": prohibited,
        "unresolved_dependencies": missing,
        "scope": "current executable inventory and named dependency closure; no execution",
    }


def builds(cases, kernels):
    records = []
    for case in cases:
        directory = kernels / case["name"]
        build = read(directory / "build-result.json")
        recorded = build.get("artifacts", {})
        mismatches = verify_hashes(directory, recorded)
        current_source = sha(case["kernel_source"])
        source_current = bool(
            current_source and current_source == build.get("kernel_module_sha256")
        )
        complete = set(recorded) == BUILD_ARTIFACTS
        valid = (
            build.get("status") == "PASS"
            and build.get("stage") == "asm"
            and complete
            and not mismatches
            and source_current
        )
        records.append(
            {
                "case": case["name"],
                "status": "PASS" if valid else "INCOMPLETE",
                "build_result": evidence(directory / "build-result.json"),
                "recorded_stage": build.get("stage"),
                "artifact_mismatches": mismatches,
                "all_11_artifacts_present": complete,
                "current_kernel_source_sha256": current_source,
                "build_kernel_source_sha256": build.get("kernel_module_sha256"),
                "current_source_matches": source_current,
            }
        )
    return {
        "status": "PASS"
        if all(r["status"] == "PASS" for r in records)
        else "INCOMPLETE",
        "scope": "actual frontend, plan/verify, emit, LLVM and object artifacts; execution is separate",
        "case_count": len(records),
        "passed": sum(r["status"] == "PASS" for r in records),
        "cases": records,
    }


def ame_evidence(kernels):
    records = []
    for name in PRIMITIVES:
        if not name.startswith("matmul_i8_"):
            continue
        directory = kernels / name
        planned_path, emitted_path = (
            directory / "kernel.planned.mlir",
            directory / "kernel.boscame.mlir",
        )
        planned = planned_path.read_text() if planned_path.is_file() else ""
        emitted = emitted_path.read_text() if emitted_path.is_file() else ""
        zero_constants = re.findall(r"(%[\w]+)\s*=\s*arith.constant 0 : i32", emitted)
        zero_stores = [
            line.strip()
            for line in emitted.splitlines()
            if any(
                re.search(r"memref.store\s+" + re.escape(z) + r",", line)
                for z in zero_constants
            )
        ]
        mma = [
            line.strip() for line in emitted.splitlines() if "bosc_ame.tile.mma" in line
        ]
        zero_acc, load_acc = (
            emitted.count("bosc_ame.tile.zero_acc"),
            emitted.count("bosc_ame.tile.load_acc"),
        )
        private = "bosc_ame.ddr.frame" in emitted
        signed_i8 = bool(
            mma and all(line.count("xi8>") == 2 and "xi32>" in line for line in mma)
        )
        backing = bool(zero_acc or (private and load_acc and zero_stores))
        records.append(
            {
                "case": name,
                "status": "PRESENT"
                if signed_i8 and backing and 'engine = "ame"' in planned
                else "INCOMPLETE",
                "planned": evidence(planned_path),
                "emitted": evidence(emitted_path),
                "ame_mma_operations": len(mma),
                "signed_i8_to_i32_mma": signed_i8,
                "zero_acc_operations": zero_acc,
                "load_acc_operations": load_acc,
                "private_ddr_frame": private,
                "scalar_i32_zero_stores": zero_stores,
                "overwrite_initializer": "tile.zero_acc"
                if zero_acc
                else "private DDR zero stores then tile.load_acc",
                "mma_excerpt": mma[:1],
                "scope": "static plan/emitted IR evidence; numerical old-C reads checked separately",
            }
        )
    return records


def simulator_evidence(kernels):
    try:
        from simcheck import collect_case
    except Exception as error:
        return {
            "status": "INCOMPLETE",
            "reason": f"{type(error).__name__}: {error}",
            "cases": [],
        }
    cases = {case["name"]: case for case in inventory()}
    records = []
    for name in PRIMITIVES:
        try:
            saved = collect_case(cases[name], kernels)
        except Exception as error:
            saved = {
                "status": "INCOMPLETE",
                "reason": f"{type(error).__name__}: {error}",
            }
        records.append(
            {
                "case": name,
                "status": saved.get("status"),
                "evidence": evidence(kernels / name / "simcheck-result.json"),
                "reason": saved.get("reason"),
                "programs": saved.get("programs", 0),
                "outputs": saved.get("outputs", 0),
                "rounds": saved.get("comparison_rounds"),
                "max_abs": saved.get("max_abs"),
                "guard_bytes_checked": sum(
                    run.get("guard_bytes_checked", 0) for run in saved.get("runs", [])
                ),
                "readonly_bytes_checked": sum(
                    run.get("readonly_bytes_checked", 0)
                    for run in saved.get("runs", [])
                ),
                "guard_errors": sum(
                    run.get("guard_errors", 0) for run in saved.get("runs", [])
                ),
                "input_errors": sum(
                    run.get("input_errors", 0) for run in saved.get("runs", [])
                ),
                "old_C_bytes_read": sum(
                    run.get("argument_bytes", {}).get("C", {}).get("read", 0)
                    for run in saved.get("runs", [])
                )
                if cases[name]["family"] == "matmul_i8"
                else None,
            }
        )
    return {
        "status": "PASS"
        if all(r["status"] == "PASS" for r in records)
        else "INCOMPLETE",
        "passed": sum(r["status"] == "PASS" for r in records),
        "required": len(PRIMITIVES),
        "scope": "saved standalone primitive strict emitted-IR execution",
        "cases": records,
    }


def elf_evidence(base):
    directory = base / "local-images"
    manifest = read(directory / "manifest.json")
    entries = {entry["case"]: entry for entry in manifest.get("entries", [])}
    records = []
    for name in PRIMITIVES:
        entry, local = entries.get(name, {}), directory / name
        audit = read(local / "elf-audit.json")
        disassembly = (
            (local / "disassembly.txt").read_text()
            if (local / "disassembly.txt").is_file()
            else ""
        )
        custom_words = [
            line.strip() for line in disassembly.splitlines() if "\t.word\t" in line
        ]
        mismatches = verify_hashes(local, entry.get("files_sha256", {}))
        recorded = entry.get("files_sha256", {})
        current = ROOT / name / "kernel.py"
        valid = (
            bool(recorded)
            and not mismatches
            and audit.get("status") == "PASS"
            and audit.get("elf_sha256") == sha(local / (name + ".elf"))
            and recorded.get(name + ".bin") == sha(local / (name + ".bin"))
            and recorded.get("kernel.py") == sha(current)
            and recorded.get("kernel.o") == sha(base / "kernels" / name / "kernel.o")
            and sha(local / "disassembly.txt")
            and audit.get("ame_instructions") == entry.get("ame_instructions")
            and (
                not name.startswith("matmul_i8_")
                or (audit.get("ame_instructions", 0) > 0 and custom_words)
            )
        )
        records.append(
            {
                "case": name,
                "status": "PASS" if valid else "INCOMPLETE",
                "elf_audit": evidence(local / "elf-audit.json"),
                "image": evidence(local / (name + ".bin")),
                "disassembly": evidence(local / "disassembly.txt"),
                "custom_instruction_excerpt": custom_words[:3],
                "hash_mismatches": mismatches,
                "ame_instructions": audit.get("ame_instructions"),
            }
        )
    return {
        "status": "PASS"
        if all(r["status"] == "PASS" for r in records)
        else "INCOMPLETE",
        "passed": sum(r["status"] == "PASS" for r in records),
        "required": len(records),
        "scope": "saved ELF/bin and audit; preparation does not imply FPGA execution",
        "cases": records,
    }


def abi_evidence(base, kernels):
    cases = {case["name"]: case for case in inventory()}
    records = []
    for name in PRIMITIVES:
        directory = base / "launchcheck" / name
        saved, case = read(directory / "result.json"), cases[name]
        valid = (
            saved.get("status") == "ABI_SMOKE_PASS"
            and saved.get("launch_sha256") == sha(case["launch_source"])
            and saved.get("adapter_sha256") == sha(kernels / name / "adapter.c")
            and saved.get("mock_sha256") == sha(directory / "mock.c")
            and saved.get("numeric_oracle_executed") is False
            and saved.get("simulator_executed") is False
            and saved.get("programs")
            == saved.get("expected_programs")
            == math.prod(case["grid"])
        )
        records.append(
            {
                "case": name,
                "status": "PASS" if valid else "INCOMPLETE",
                "evidence": evidence(directory / "result.json"),
                "programs": saved.get("programs"),
            }
        )
    return {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in records)
        else "INCOMPLETE",
        "passed": sum(item["status"] == "PASS" for item in records),
        "required": len(records),
        "scope": "saved native launcher/descriptor/grid mock smoke; no numerical execution",
        "cases": records,
    }


def markdown(report):
    lines = [
        "# ViT standalone primitive evidence",
        "",
        f"Collection: **{report['status']}**. Scope: {report['scope']}.",
        "",
        f"Inventory/dependencies: {report['inventory']['status']} ({report['inventory']['case_count']} cases). "
        f"Current ASM artifacts: {report['build']['passed']}/{report['build']['case_count']}. "
        f"Primitive strict checks: {report['primitive_simulator'].get('passed', 0)}/{len(PRIMITIVES)}. "
        f"Native ABI mock: {report['abi']['passed']}/{report['abi']['required']}; "
        f"local ELF/audit: {report['elf']['passed']}/{report['elf']['required']}.",
        "",
        "The detailed numerical, ABI and image evidence below covers the named standalone primitives, "
        "including pack_heads_12x197x197 and unpack_heads_12x197x197. "
        "Other inventory cases require their own independent checks.",
        "",
        "| Primitive | Compile | ABI mock | Saved strict numerical | ELF/image | Programs | Outputs |",
        "|---|---|---|---|---|---:|---:|",
    ]
    build_cases = {case["case"]: case for case in report["build"]["cases"]}
    abi_cases = {case["case"]: case for case in report["abi"]["cases"]}
    elf_cases = {case["case"]: case for case in report["elf"]["cases"]}
    numerical_cases = {
        case["case"]: case for case in report["primitive_simulator"]["cases"]
    }
    for name in PRIMITIVES:
        numerical = numerical_cases.get(name, {})
        lines.append(
            f"| {name} | {build_cases.get(name, {}).get('status', 'INCOMPLETE')} | "
            f"{abi_cases.get(name, {}).get('status', 'INCOMPLETE')} | {numerical.get('status', 'INCOMPLETE')} | "
            f"{elf_cases.get(name, {}).get('status', 'INCOMPLETE')} | "
            f"{numerical.get('programs', 0)} | {numerical.get('outputs', 0)} |"
        )
    lines += ["", "AME evidence:", ""]
    for item in report["ame"]:
        lines.append(
            f"- {item['case']}: {item['status']}; {item['ame_mma_operations']} emitted signed-i8/i32 "
            f"MMA operations; initializer={item['overwrite_initializer']}."
        )
    lines += [
        "",
        "This collection reads saved artifacts and verifies their hashes. It does not execute candidates. "
        "ABI smoke uses a mock kernel; prepared ELF/images do not establish FPGA execution. "
        "Full model execution and checkpoint accuracy remain unverified.",
        "",
        f"FPGA: **{report['board']['status']}**. No task, UART, cycles or hardware check count was generated. "
        "Historical reports, images and hashes remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--kernels", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    base = args.report_dir.resolve()
    kernels = (args.kernels or base / "kernels").resolve()
    os.environ.setdefault("BUDDY_ROOT", str(ROOT.parents[2] / "buddy-mlir"))
    os.environ.setdefault(
        "BUDDY_OPT", str(ROOT.parents[2] / "buddy-mlir/build/bin/buddy-opt")
    )
    cases = inventory()
    inventory_record, build = source_inventory(cases), builds(cases, kernels)
    ame = ame_evidence(kernels)
    primitive_simulator = simulator_evidence(kernels)
    elf, abi = elf_evidence(base), abi_evidence(base, kernels)
    ready = (
        inventory_record["status"] == build["status"] == "PASS"
        and all(item["status"] == "PRESENT" for item in ame)
        and primitive_simulator["status"] == elf["status"] == abi["status"] == "PASS"
    )
    report = normalize(
        {
            "schema": 3,
            "status": "SAVED_PRIMITIVE_EVIDENCE_COMPLETE" if ready else "INCOMPLETE",
            "scope": "standalone_primitive_evidence",
            "required_primitives": list(PRIMITIVES),
            "full_model_executed": False,
            "checkpoint_accuracy_measured": False,
            "collector_source": evidence(Path(__file__)),
            "inventory": inventory_record,
            "build": build,
            "ame": ame,
            "primitive_simulator": primitive_simulator,
            "elf": elf,
            "abi": abi,
            "board": {
                "status": "NOT_RUN_USER_FPGA_BROKEN",
                "hardware_started": False,
                "tasks": [],
                "uart": [],
                "cycles": [],
                "checks": 0,
                "execution_timeout_seconds": 120,
                "reason": "User reports FPGA broken and forbids board connection, tunnel recovery, service queries and execution.",
            },
            "preservation": {
                "before_change": evidence(base / "before-change.tar.gz"),
                "historical_report_and_image_hashes": evidence(
                    base / "historical-report-preservation.json"
                ),
                "compiler_rollback": evidence(base / "compiler-rollback/summary.json"),
                "retired_sources": rel(base / "retired-source"),
                "retired_tools": rel(base / "retired-tools"),
                "historical_board_evidence_reused_as_current_pass": False,
            },
        }
    )
    base.mkdir(parents=True, exist_ok=True)
    (base / "primitive-validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    table = markdown(report)
    (base / "primitive-validation.md").write_text(table + "\n")
    if args.markdown:
        args.markdown.write_text(table + "\n")
    print(table)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

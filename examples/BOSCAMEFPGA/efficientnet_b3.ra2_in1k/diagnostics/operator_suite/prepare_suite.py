#!/usr/bin/env python3
"""Inventory production EfficientNet images without changing or compiling them.

Numerical evidence is reported separately from board completion. Historical
cycles are accepted as current only when their exact BIN SHA256 matches.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
sys.path.insert(0, str(MODEL))
from cases import inventory  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def short(path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def historical_runs():
    records = []
    for filename in (
        "FPGA112_RESULTS.json",
        "W8A8_FPGA112_RESULTS.json",
        "W8A8_DEPTHWISE_FPGA112_LONGRUN_RESULTS.json",
    ):
        source = MODEL / filename
        report = read_json(source)
        if not report:
            continue
        for row in report.get("cases", report.get("results", [])):
            records.append(
                {
                    "case": row.get("case", row.get("name")),
                    "image_sha256": row.get("image_sha256"),
                    "launch_cycles": row.get("launch_cycles", row.get("cycles")),
                    "scheduler_seconds": row.get(
                        "scheduler_seconds", row.get("seconds")
                    ),
                    "execution_status": row.get("execution_status"),
                    "state": row.get("state"),
                    "task_id": row.get("task_id"),
                    "fpga": row.get("fpga"),
                    "source_report": short(source),
                    "evidence": row.get("evidence"),
                }
            )
    return records


def numerical_evidence(case, directory, hashes):
    evidence = []
    saved = read_json(directory / "simcheck-result.json")
    if saved:
        runs = saved.get("runs", [])
        current = saved.get("module_sha256") == hashes.get("kernel.boscame.mlir")
        source = Path(case["kernel_source"])
        source_current = saved.get("kernel_module_sha256") in (None, sha256(source))
        coverage = (
            saved.get("programs") == saved.get("total_programs")
            and saved.get("programs", 0) > 0
            and all(run.get("coverage", "full") == "full" for run in runs)
        )
        evidence.append(
            {
                "kind": "simcheck",
                "path": short(directory / "simcheck-result.json"),
                "status": saved.get("status"),
                "emitted_module_matches": current,
                "case_source_matches": source_current,
                "full_recorded_grid": coverage,
                "programs": saved.get("programs"),
                "outputs": saved.get("outputs"),
                "comparison_rounds": saved.get("comparison_rounds"),
                "accepted_current": saved.get("status") == "PASS"
                and current
                and source_current
                and coverage,
            }
        )
    sharded = MODEL / "build/w8a8-validation/depthwise-full/report.json"
    saved = read_json(sharded)
    if saved and saved.get("name") == case["name"]:
        checks = {}
        for name, expected in saved.get("artifact_sha256", {}).items():
            path = Path(name) if Path(name).is_absolute() else REPO / name
            checks[name] = path.is_file() and sha256(path) == expected
        accepted = (
            saved.get("status") == "PASS"
            and checks
            and all(checks.values())
            and saved.get("tile_coverage_exactly_once") is True
            and saved.get("output_coverage_exactly_once") is True
            and saved.get("bit_exact_w8a8") is True
        )
        evidence.append(
            {
                "kind": "disjoint_actual_ir_shards",
                "path": short(sharded),
                "status": saved.get("status"),
                "coverage": saved.get("coverage"),
                "outputs": saved.get("outputs"),
                "artifact_hash_checks": checks,
                "accepted_current": bool(accepted),
            }
        )
    return evidence


def static_metrics(case, directory):
    text = (directory / "kernel.boscame.mlir").read_text()
    asm = (directory / "kernel.nr.s").read_text()
    families = {
        "conv2d_f32": (
            "quantized_convolution",
            "im2col plus activation quantization, many raw calls; MMA GEMM",
        ),
        "dwconv2d_f32": (
            "depthwise_quantized_convolution",
            "many per-channel im2col calls and BM=1 MMA occupancy",
        ),
        "conv2d_i8": (
            "packed_mma_pointwise",
            "packed INT8 MMA with shape-dependent launch/packing costs",
        ),
        "matmul_i8": ("packed_mma", "INT8 MMA with scalar/DDR launch overhead"),
        "linear_f32": (
            "w8a8_classifier",
            "FP32 public ABI; activation/weight quantization, preserved INT8 MMA and dequantization plus bias",
        ),
        "gap": (
            "ordered_f32_reduction",
            "per-channel reduction, scalar ordered F32 reduction policy",
        ),
        "per_token_quantization": (
            "quantization_reduction",
            "abs/max reduction, scale division and integer rounding",
        ),
        "silu": (
            "nonlinear_elementwise",
            "exponential approximation and elementwise work",
        ),
        "add": ("memory_elementwise", "full-array memory and vector elementwise work"),
        "dequantization": (
            "dequantization_elementwise",
            "integer conversion, scale and vector memory work",
        ),
    }
    category, reason = families[case["family"]]
    instructions = Counter()
    for line in asm.splitlines():
        match = re.match(r"\s+([a-z][a-z0-9_.]*)\s", line)
        if match:
            mnemonic = match.group(1)
            if mnemonic.startswith(("f", "v", "ame", "m")) or mnemonic in (
                "flw",
                "fsw",
                "lb",
                "sb",
                "lw",
                "sw",
            ):
                instructions[mnemonic] += 1
    return {
        "category": category,
        "potential_cost": reason,
        "grid": case["grid"],
        "raw_public_grid_programs": math.prod(case["grid"]),
        "logical_shape_elements": math.prod(case["shape"]),
        "static_ir_counts": {
            name: text.count(name)
            for name in (
                "scf.for",
                "arith.divf",
                "arith.mulf",
                "math.absf",
                "bosc_ame.direct_ddr",
                "bosc_ame.mma",
                "vector<",
                "memref.load",
                "memref.store",
            )
        },
        "assembly_mnemonic_counts": dict(instructions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=MODEL / "build/kernels")
    parser.add_argument(
        "--overlay-kernels",
        type=Path,
        action="append",
        default=[],
        help="use a case directory from this frozen build root when present; later roots win",
    )
    parser.add_argument(
        "--output", type=Path, default=MODEL / "build/operator-suite/preparation"
    )
    parser.add_argument(
        "--ready",
        action="store_true",
        help="record a frozen ready suite after the optimization phase completes",
    )
    args = parser.parse_args()
    cases = inventory()
    if len(cases) != 99:
        raise RuntimeError(f"expected99 actual case configs, observed{len(cases)}")
    history = historical_runs()
    records = []
    for case in cases:
        directory = args.kernels.resolve() / case["name"]
        for overlay in args.overlay_kernels:
            candidate = overlay.resolve() / case["name"]
            if candidate.is_dir():
                directory = candidate
        names = (
            "kernel.boscame.mlir",
            "kernel.o",
            "adapter.c",
            "frontend.json",
            "contract-manifest.json",
            "elf-audit.json",
            case["name"] + ".elf",
            case["name"] + ".bin",
        )
        hashes = {
            name: sha256(directory / name)
            for name in names
            if (directory / name).is_file()
        }
        image = directory / (case["name"] + ".bin")
        if not image.is_file() or len(hashes) != len(names):
            raise RuntimeError(
                "missing compiled production artifact: " + str(directory)
            )
        image_bytes = image.read_bytes()
        protocol_markers = {
            marker: marker.encode() in image_bytes
            for marker in (
                "@@FPGA_START",
                "@@FPGA_RESULT",
                "launch_returned=",
                "launch_cycles=0x",
            )
        }
        if args.ready and not all(protocol_markers.values()):
            raise RuntimeError(
                "image lacks current scheduler/timer protocol; relink with current board.mk: "
                + str(image)
            )
        board = [item for item in history if item["case"] == case["name"]]
        current = [item for item in board if item["image_sha256"] == hashes[image.name]]
        successful = [
            item
            for item in current
            if item["launch_cycles"] is not None and item["state"] == "completed"
        ]
        historical_seconds = [
            item["scheduler_seconds"]
            for item in board
            if item["launch_cycles"] is not None
            and item["scheduler_seconds"] is not None
        ]
        prior_seconds = max(historical_seconds, default=0)
        # A changed BIN retains prior time only as an estimate, never as current evidence.
        deadline = (
            1200 if prior_seconds >= 550 else (600 if prior_seconds >= 50 else 60)
        )
        evidence = numerical_evidence(case, directory, hashes)
        frontend = read_json(directory / "frontend.json") or {}
        source_hashes = {
            case["kernel_source"]: sha256(Path(case["kernel_source"])),
            case["launch_source"]: sha256(Path(case["launch_source"])),
        }
        for relative, expected in frontend.get("source_sha256", {}).items():
            path = Path(relative) if Path(relative).is_absolute() else REPO / relative
            source_hashes[str(path)] = expected
        for stage in frontend.get("stages", []):
            if stage.get("kernel_source") and stage.get("kernel_source_sha256"):
                source_hashes[stage["kernel_source"]] = stage["kernel_source_sha256"]
        record = {
            "case": case["name"],
            "family": case["family"],
            "image": str(image),
            "image_sha256": hashes[image.name],
            "image_bytes": image.stat().st_size,
            "protocol_markers_in_image": protocol_markers,
            "artifact_directory": str(directory),
            "artifact_sha256": hashes,
            "source_sha256": source_hashes,
            "board_current_completed": bool(successful),
            "board_current_results": current,
            "board_historical_results": board,
            "numerical_evidence": evidence,
            "numerical_current_full_pass": any(
                item["accepted_current"] for item in evidence
            ),
            "board_launcher_validation": case["validation"],
            "planned_initial_timeout_seconds": deadline,
            "acceptable_seconds": 60,
            "max_timeout_seconds": 1200,
            "historical_runtime_estimate_seconds": prior_seconds or None,
            "timeout_basis": "historical observed runtime; stale hashes are estimates only",
            "static": static_metrics(case, directory),
        }
        records.append(record)
    report = {
        "schema": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": "fpga112",
        "suite": "EfficientNet-B3 99 production operator specializations",
        "status": "READY" if args.ready else "PREPARATION_ONLY",
        "formal_run_allowed": args.ready,
        "artifact_roots": {
            "default": str(args.kernels.resolve()),
            "overlays_in_priority_order": [
                str(path.resolve()) for path in args.overlay_kernels
            ],
        },
        "note": "Regenerate after optimized convolution artifacts freeze; actual formal runner must execute every selected BIN.",
        "counts": {
            "cases": len(records),
            "families": dict(Counter(item["family"] for item in records)),
            "current_board_cycles": sum(
                item["board_current_completed"] for item in records
            ),
            "current_full_numerical_pass": sum(
                item["numerical_current_full_pass"] for item in records
            ),
            "any_historical_board_evidence": sum(
                bool(item["board_historical_results"]) for item in records
            ),
            "planned_initial_timeouts": dict(
                Counter(item["planned_initial_timeout_seconds"] for item in records)
            ),
        },
        "cases": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "suite-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    fields = [
        "case",
        "family",
        "image_sha256",
        "image_bytes",
        "board_current_completed",
        "numerical_current_full_pass",
        "planned_initial_timeout_seconds",
        "historical_runtime_estimate_seconds",
    ]
    with (args.output / "inventory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: record[name] for name in fields} for record in records)
    lines = [
        "# Operator suite preparation",
        "",
        "No formal board tasks are submitted by this script.",
        "",
        "```json",
        json.dumps(report["counts"], indent=2),
        "```",
        "",
        "| Case | Family | Current board cycles | Current full numerical evidence | Planned first deadline |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for record in records:
        lines.append(
            f"| `{record['case']}` | {record['family']} | {record['board_current_completed']} | {record['numerical_current_full_pass']} | {record['planned_initial_timeout_seconds']}s |"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

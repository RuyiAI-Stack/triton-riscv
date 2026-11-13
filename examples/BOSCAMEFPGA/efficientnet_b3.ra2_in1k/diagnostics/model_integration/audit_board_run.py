#!/usr/bin/env python3
"""Audit all downloaded logits and call cycles from a completed full-model run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "build/model-integration/physical-plan",
    )
    parser.add_argument(
        "--primary-reference-dir",
        type=Path,
        default=ROOT / "build/model-integration/physical-plan",
        help="preserved original Torch W8A8 reference, reported independently",
    )
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "image-manifest.json").read_text())
    plan_file = args.reference_dir / "physical-plan.json"
    plan = json.loads(plan_file.read_text())
    expected_calls = len(plan["calls"])
    checks = {
        "full_model": summary["full_model"] and manifest["full_model"],
        "completed": summary["state"] == "completed",
        "exited_zero": (
            summary["execution_status"] == "exited"
            and summary["app_exit_code"] == 0
            and summary["exit_code"] == 0
        ),
        "fpga112": summary["server"] == "fpga112",
        "plan_hash": manifest["plan_sha256"] == sha(plan_file),
        "physical_calls": (
            summary["physical_calls"]
            == manifest["physical_calls"]
            == expected_calls
            == 503
        ),
        "uploaded_image_hash": summary["scheduler_image_hash_match"],
        "snapshot_image_hash": (
            sha(directory / "efficientnet_b3_model.bin")
            == manifest["image_sha256"]
            == summary["image_sha256"]
        ),
        "ddr_readback": summary["details"].get("ddr_readback_matches") is True,
        "application_started": summary["details"].get("application_started") is True,
    }
    downloads = {item["name"]: item for item in summary["downloads"]}
    required = {"logits_f32", "call_cycles_u64"}
    checks["artifacts_present"] = required <= downloads.keys()
    if not checks["artifacts_present"]:
        raise ValueError("full-model DDR artifacts are missing")

    def artifact(name):
        item = downloads[name]
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO / path
        checks[name + "_download_hash"] = (
            sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        )
        return path

    logits_file = artifact("logits_f32")
    cycles_file = artifact("call_cycles_u64")
    actual = np.fromfile(logits_file, dtype="<f4")
    expected = np.fromfile(args.reference_dir / "w8a8-logits.f32", dtype="<f4")
    primary_file = args.primary_reference_dir / "w8a8-logits.f32"
    primary = np.fromfile(primary_file, dtype="<f4")
    cycles = np.fromfile(cycles_file, dtype="<u8")
    checks["logit_shape"] = actual.shape == expected.shape == primary.shape == (1000,)
    checks["cycle_shape"] = cycles.shape == (expected_calls,)
    if not checks["logit_shape"] or not checks["cycle_shape"]:
        raise ValueError("unexpected full-model DDR artifact shape")
    checks["finite_logits"] = bool(np.isfinite(actual).all())
    checks["nonzero_cycles"] = bool((cycles > 0).all())
    reference_offset = (
        int(manifest["elf_layout_symbols"]["model_reference_blob"], 16) - 0x80000000
    )
    snapshot = (directory / "efficientnet_b3_model.bin").read_bytes()
    checks["reference_matches_embedded_bytes"] = (
        reference_offset >= 0
        and snapshot[reference_offset : reference_offset + expected.nbytes]
        == expected.tobytes()
    )
    provenance = manifest.get("reference_provenance", {})
    checks["reference_provenance_hash"] = manifest.get(
        "reference_logits_sha256", sha(args.reference_dir / "w8a8-logits.f32")
    ) == sha(args.reference_dir / "w8a8-logits.f32")
    checks["primary_reference_hash"] = provenance.get(
        "primary_golden_sha256", sha(primary_file)
    ) == sha(primary_file)
    sum_call_cycles = sum(int(value) for value in cycles)
    checks["model_cycles_cover_calls"] = (
        isinstance(summary["model_cycles"], int)
        and summary["model_cycles"] >= sum_call_cycles > 0
    )
    uart_cycles = {item["call"]: item for item in summary["per_call_cycles"]}
    checks["uart_call_coverage"] = set(uart_cycles) == set(range(expected_calls))
    uart = (directory / "uart.log").read_text()
    footer = uart.rfind("[efficientnet-model] calls=")
    cycle_pattern = r"\[efficientnet-cycle\] call=(\d+) cycles=(0x[0-9a-fA-F]+|[0-9]+)"
    final_pairs = (
        [
            (int(index), int(value, 0))
            for index, value in re.findall(cycle_pattern, uart[footer:])
        ]
        if footer >= 0
        else []
    )
    final_cycles = dict(final_pairs)
    checks["final_uart_call_coverage"] = len(final_pairs) == expected_calls and set(
        final_cycles
    ) == set(range(expected_calls))
    checks["uart_ddr_cycles_match"] = checks["final_uart_call_coverage"] and all(
        int(cycles[index]) == value for index, value in final_pairs
    )
    live_pairs = (
        [
            (int(index), int(value, 0))
            for index, value in re.findall(cycle_pattern, uart[:footer])
        ]
        if footer >= 0
        else []
    )
    threshold = manifest["diagnostic_validation_threshold"]
    checks["declared_numerical_threshold"] = bool(
        np.allclose(actual, expected, atol=threshold["atol"], rtol=threshold["rtol"])
    )
    checks["top1_matches"] = int(np.argmax(actual)) == int(np.argmax(expected))
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    primary_error = np.abs(actual.astype(np.float64) - primary.astype(np.float64))
    primary_close = np.isclose(
        actual, primary, atol=threshold["atol"], rtol=threshold["rtol"]
    )
    selected_numerical_close = bool(
        checks["finite_logits"] and checks["declared_numerical_threshold"]
    )
    primary_numerical_close = bool(checks["finite_logits"] and primary_close.all())
    selected_reference_integrity = bool(
        checks["reference_matches_embedded_bytes"]
        and checks["reference_provenance_hash"]
    )
    primary_reference_integrity = bool(checks["primary_reference_hash"])
    original = np.fromfile(args.reference_dir / "original-logits.f32", dtype="<f4")
    if original.shape != (1000,) or not np.isfinite(original).all():
        raise ValueError("invalid original FP32 comparison reference")
    original_error = np.abs(actual.astype(np.float64) - original.astype(np.float64))
    rows = []
    for call, count in zip(plan["calls"], cycles):
        rows.append(
            {
                "call": call["index"] + 1,
                "logical_node": call["logical_node"],
                "target_kind": call["target_kind"],
                "case": call["target"],
                "cycles": int(count),
            }
        )
    with (directory / "audited-call-cycles.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "audited-logits.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "index",
                "actual",
                "selected_reference",
                "primary_torch_w8a8",
                "original_fp32",
                "selected_abs_error",
                "primary_abs_error",
            ]
        )
        writer.writerows(
            (
                index,
                float(actual[index]),
                float(expected[index]),
                float(primary[index]),
                float(original[index]),
                float(error[index]),
                float(primary_error[index]),
            )
            for index in range(actual.size)
        )
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "full-model FPGA run; all 1000 logits and 503 call cycles downloaded",
        "task_id": summary["task_id"],
        "server": summary["server"],
        "fpga": summary["fpga"],
        "image_sha256": manifest["image_sha256"],
        "checks": checks,
        "selected_reference_provenance": provenance,
        "numerical_validation_location": manifest.get(
            "numerical_validation_location", "inline and host"
        ),
        "execution_pass": all(
            value
            for key, value in checks.items()
            if key
            not in ("declared_numerical_threshold", "top1_matches", "finite_logits")
        ),
        "selected_reference_numerical_close": selected_numerical_close,
        "primary_torch_numerical_close": primary_numerical_close,
        "selected_reference_integrity_pass": selected_reference_integrity,
        "primary_reference_integrity_pass": primary_reference_integrity,
        "selected_reference_pass": selected_numerical_close
        and selected_reference_integrity,
        "primary_torch_reference_pass": primary_numerical_close
        and primary_reference_integrity,
        "pass_scope": "reference pass requires numerical closeness and matching reference integrity; numerical_close fields report arithmetic separately; execution integrity has its own pass flag",
        "scheduler_seconds": summary["scheduler_seconds"],
        "model_cycles": summary["model_cycles"],
        "sum_call_cycles": sum_call_cycles,
        "live_uart_observation": {
            "samples": len(live_pairs),
            "zero_cycle_samples": sum(value == 0 for _, value in live_pairs),
            "mismatches_against_final_ddr": sum(
                index >= expected_calls or value != int(cycles[index])
                for index, value in live_pairs
            ),
            "timing_source": "final DDR artifact, independently matched to the final UART report",
            "meaning": "stale live reads are invalid timing observations; final report and downloaded counters are checked separately",
        },
        "logits": {
            "elements": actual.size,
            "sha256": sha(logits_file),
            "bit_exact": bool(
                np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
            ),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "allclose_2e_4": bool(np.allclose(actual, expected, atol=2e-4, rtol=2e-4)),
            "allclose_2e_6": bool(np.allclose(actual, expected, atol=2e-6, rtol=2e-6)),
            "actual_top5": np.argsort(actual)[-5:][::-1].tolist(),
            "w8a8_reference_top5": np.argsort(expected)[-5:][::-1].tolist(),
            "declared_atol": threshold["atol"],
            "declared_rtol": threshold["rtol"],
        },
        "difference_from_original_fp32": {
            "max_abs_error": float(original_error.max()),
            "mean_abs_error": float(original_error.mean()),
            "top5": np.argsort(original)[-5:][::-1].tolist(),
            "meaning": "includes W8A8 quantization; one declared regression input, not ImageNet accuracy",
        },
        "primary_torch_w8a8_comparison": {
            "sha256": sha(primary_file),
            "max_abs_error": float(primary_error.max()),
            "mean_abs_error": float(primary_error.mean()),
            "outside_declared_threshold": int((~primary_close).sum()),
            "outside_declared_threshold_indices": np.flatnonzero(
                ~primary_close
            ).tolist(),
            "top5": np.argsort(primary)[-5:][::-1].tolist(),
            "meaning": "preserved Torch mean/activation semantics; reported separately from any explicitly selected target-order CPU reference",
        },
        "call_cycles_sha256": sha(cycles_file),
        "slowest_calls": sorted(rows, key=lambda row: row["cycles"], reverse=True)[:20],
    }
    (directory / "independent-board-audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise ValueError(
            "full-model board audit failed; see independent-board-audit.json"
        )


if __name__ == "__main__":
    main()

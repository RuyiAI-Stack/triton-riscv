#!/usr/bin/env python3
"""Check dual-reference audit reporting with synthetic, explicitly non-model data.

No FPGA execution is represented by these fixtures. Each fixture has 1000
logits and 503 counters only to exercise the strict shape checks cheaply;
full_model=False must keep execution_pass and overall status false.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(
    directory,
    *,
    selected_offset=0,
    primary_offset=0,
    primary_elements=1000,
    original_elements=1000,
    bad_primary_hash=False,
    bad_selected_hash=False,
    bad_embedded_reference=False,
):
    directory.mkdir(parents=True, exist_ok=True)
    selected = directory / "selected-reference"
    primary = directory / "primary-reference"
    run = directory / "run"
    for path in (selected, primary, run):
        path.mkdir(exist_ok=True)
    actual = np.linspace(-1, 1, 1000, dtype="<f4")
    selected_values = (actual + np.float32(selected_offset)).astype("<f4")
    primary_values = (actual[:primary_elements] + np.float32(primary_offset)).astype(
        "<f4"
    )
    original = actual[:original_elements].copy()
    actual.tofile(run / "logits_f32")
    selected_values.tofile(selected / "w8a8-logits.f32")
    primary_values.tofile(primary / "w8a8-logits.f32")
    original.tofile(selected / "original-logits.f32")
    plan = {
        "fixture": "synthetic selfcheck, not a physical model plan",
        "calls": [
            {
                "index": index,
                "logical_node": "synthetic_" + str(index),
                "target_kind": "synthetic",
                "target": "synthetic",
            }
            for index in range(503)
        ],
    }
    dump(selected / "physical-plan.json", plan)
    cycles = np.arange(1, 504, dtype="<u8")
    cycles.tofile(run / "call_cycles_u64")
    embedded = bytearray(selected_values.tobytes())
    if bad_embedded_reference:
        embedded[0] ^= 1
    (run / "efficientnet_b3_model.bin").write_bytes(embedded)
    manifest = {
        "full_model": False,
        "physical_calls": 503,
        "plan_sha256": sha(selected / "physical-plan.json"),
        "image_sha256": sha(run / "efficientnet_b3_model.bin"),
        "elf_layout_symbols": {"model_reference_blob": "0x80000000"},
        "reference_logits_sha256": "0" * 64
        if bad_selected_hash
        else sha(selected / "w8a8-logits.f32"),
        "reference_provenance": {
            "kind": "synthetic diagnostic selfcheck only",
            "primary_golden_sha256": "0" * 64
            if bad_primary_hash
            else sha(primary / "w8a8-logits.f32"),
        },
        "diagnostic_validation_threshold": {"atol": 2e-6, "rtol": 2e-6},
    }
    dump(run / "image-manifest.json", manifest)
    summary = {
        "task_id": "synthetic_selfcheck_no_board_submission",
        "server": "fpga112",
        "fpga": None,
        "state": "completed",
        "execution_status": "exited",
        "app_exit_code": 0,
        "exit_code": 0,
        "full_model": False,
        "physical_calls": 503,
        "scheduler_image_hash_match": True,
        "image_sha256": manifest["image_sha256"],
        "scheduler_seconds": None,
        "details": {"ddr_readback_matches": True, "application_started": True},
        "model_cycles": int(cycles.sum()),
        "per_call_cycles": [
            {"call": index, "cycles": int(cycle)} for index, cycle in enumerate(cycles)
        ],
        "downloads": [
            {
                "name": name,
                "path": str((run / name).resolve()),
                "sha256": sha(run / name),
                "bytes": (run / name).stat().st_size,
            }
            for name in ("logits_f32", "call_cycles_u64")
        ],
    }
    dump(run / "summary.json", summary)
    (run / "uart.log").write_text(
        "[efficientnet-model] calls=503\n"
        + "".join(
            f"[efficientnet-cycle] call={index} cycles={int(cycle)}\n"
            for index, cycle in enumerate(cycles)
        )
    )
    return run, selected, primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL
        / "build/model-integration/dual-reference-audit-selfcheck-20261104",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    auditor = HERE / "audit_board_run.py"
    initial_auditor_hash = sha(auditor)
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(auditor), __file__], check=True
    )
    results = []
    scenarios = [
        ("selected_exact_primary_diff", {"primary_offset": 0.25}, True, False),
        ("selected_diff_primary_exact", {"selected_offset": 0.25}, False, True),
        ("both_exact_but_not_full_model", {}, True, True),
        ("primary_hash_mismatch", {"bad_primary_hash": True}, True, False),
        ("selected_hash_mismatch", {"bad_selected_hash": True}, False, True),
        ("selected_embedded_mismatch", {"bad_embedded_reference": True}, False, True),
        (
            "both_hashes_mismatch",
            {"bad_selected_hash": True, "bad_primary_hash": True},
            False,
            False,
        ),
        ("primary_shape_999_rejected", {"primary_elements": 999}, None, None),
        ("original_shape_999_rejected", {"original_elements": 999}, None, None),
    ]
    for name, configuration, selected_pass, primary_pass in scenarios:
        directory = args.output / name
        run, selected, primary = fixture(directory, **configuration)
        command = [
            sys.executable,
            str(auditor),
            "--run-dir",
            str(run),
            "--reference-dir",
            str(selected),
            "--primary-reference-dir",
            str(primary),
        ]
        process = subprocess.run(command, text=True, capture_output=True, check=False)
        (directory / "audit.stdout").write_text(process.stdout)
        (directory / "audit.stderr").write_text(process.stderr)
        result = {
            "scenario": name,
            "synthetic_no_board_submission": True,
            "full_model": False,
            "exit_code": process.returncode,
            "selected_reference_elements": 1000,
            "primary_reference_elements": configuration.get("primary_elements", 1000),
            "original_reference_elements": configuration.get("original_elements", 1000),
        }
        assert process.returncode != 0, (
            name + ": non-model fixture unexpectedly accepted"
        )
        report_path = run / "independent-board-audit.json"
        if selected_pass is None:
            expected_error = (
                "unexpected full-model"
                if "primary_elements" in configuration
                else "invalid original FP32 comparison reference"
            )
            assert expected_error in process.stderr, (
                name + ": missing explicit shape rejection"
            )
            result["shape_rejection_verified"] = True
        else:
            report = json.loads(report_path.read_text())
            assert report["status"] == "FAIL" and report["execution_pass"] is False
            assert report["checks"]["full_model"] is False
            assert report["selected_reference_pass"] is selected_pass
            assert report["primary_torch_reference_pass"] is primary_pass
            assert report["selected_reference_numerical_close"] is (
                configuration.get("selected_offset", 0) == 0
            )
            assert report["primary_torch_numerical_close"] is (
                configuration.get("primary_offset", 0) == 0
            )
            assert report["selected_reference_integrity_pass"] is not (
                configuration.get("bad_selected_hash", False)
                or configuration.get("bad_embedded_reference", False)
            )
            assert report["primary_reference_integrity_pass"] is not configuration.get(
                "bad_primary_hash", False
            )
            assert report["checks"][
                "reference_provenance_hash"
            ] is not configuration.get("bad_selected_hash", False)
            assert report["checks"][
                "reference_matches_embedded_bytes"
            ] is not configuration.get("bad_embedded_reference", False)
            if configuration.get("bad_primary_hash"):
                assert report["checks"]["primary_reference_hash"] is False
            else:
                assert report["checks"]["primary_reference_hash"] is True
            csv_lines = (run / "audited-logits.csv").read_text().splitlines()
            assert (
                len(csv_lines) == 1001
                and "selected_reference,primary_torch_w8a8" in csv_lines[0]
            )
            result.update(
                status=report["status"],
                execution_pass=report["execution_pass"],
                selected_reference_pass=report["selected_reference_pass"],
                primary_torch_reference_pass=report["primary_torch_reference_pass"],
                selected_reference_numerical_close=report[
                    "selected_reference_numerical_close"
                ],
                primary_torch_numerical_close=report["primary_torch_numerical_close"],
                selected_reference_integrity_pass=report[
                    "selected_reference_integrity_pass"
                ],
                primary_reference_integrity_pass=report[
                    "primary_reference_integrity_pass"
                ],
                pass_scope=report.get("pass_scope"),
                primary_max_abs_error=report["primary_torch_w8a8_comparison"][
                    "max_abs_error"
                ],
                selected_max_abs_error=report["logits"]["max_abs_error"],
                logits_csv_rows=1000,
                audit_report=str(report_path),
            )
        result["check_passed"] = True
        results.append(result)
    assert sha(auditor) == initial_auditor_hash, (
        "auditor changed during selfcheck; rerun"
    )
    report = {
        "status": "PASS",
        "scope": "synthetic dual-reference reporting checks only; no FPGA jobs",
        "auditor_source": str(auditor),
        "auditor_sha256": initial_auditor_hash,
        "production_modified": False,
        "python_compile_pass": True,
        "scenarios": len(results),
        "results": results,
    }
    dump(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

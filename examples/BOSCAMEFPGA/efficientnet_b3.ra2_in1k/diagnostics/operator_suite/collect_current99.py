#!/usr/bin/env python3
"""Audit current default objects against preserved, actual fpga112 executions.

This is a collector, not a new board run. Objects, BINs, scheduler task files,
and downloaded outputs are checked separately so a diagnostic harness cannot
be mistaken for the default smoke image or for whole-model execution.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

MODEL = Path(__file__).resolve().parents[2]
HZ = 5529600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def indexed(document, key="cases"):
    return {row["case"]: row for row in document[key]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(case):
    name = case["name"]
    directory = MODEL / "build/kernels" / name
    files = {
        f: sha(directory / f)
        for f in ("kernel.o", "adapter.c", "kernel.boscame.mlir", name + ".bin")
    }
    sources = {
        str(MODEL / name / f): sha(MODEL / name / f)
        for f in ("kernel.py", "launch.c", "makefile")
    }
    return {
        "artifact_directory": str(directory),
        "artifact_sha256": files,
        "source_sha256": sources,
    }


def board_record(row, task_directory, image):
    """Verify original scheduler terminal state and immutable submitted image."""
    task_directory = Path(task_directory)
    task_path = task_directory / "task.json"
    task = read(task_path)
    result = task["result"]
    image_hash = row["image_sha256"]
    require(task["id"] == row["task_id"], "task identity mismatch")
    require(task["state"] == row["state"] == "completed", "task did not complete")
    require(
        result["execution_status"] == row["execution_status"] == "exited",
        "application did not exit",
    )
    require(result["app_exit_code"] == row["app_exit_code"] == 0, "application failed")
    require(result["exit_code"] == row["exit_code"] == 0, "scheduler execution failed")
    require(
        result["details"]["image_sha256"] == image_hash == sha(image),
        "submitted image hash mismatch",
    )
    uart = result.get("uart_output", result.get("output", ""))
    require(
        "@@FPGA_START" in uart and "@@FPGA_RESULT" in uart,
        "missing scheduler protocol markers",
    )
    launch_match = re.search(r"launch_cycles=(0x[0-9a-fA-F]+|[0-9]+)", uart)
    oracle_match = re.search(r"verify [^:]+: PASS errors=([0-9a-fA-F]+)", uart)
    downloads = []
    for item in row.get("downloads", []):
        path = Path(item["path"])
        expected = item.get("downloaded_sha256", item.get("sha256"))
        require(expected == sha(path), "download hash mismatch: " + str(path))
        downloads.append(
            {
                "name": item["name"],
                "path": str(path),
                "sha256": expected,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "task_id": task["id"],
        "fpga": task["fpga"],
        "state": task["state"],
        "execution_status": result["execution_status"],
        "app_exit_code": 0,
        "exit_code": 0,
        "scheduler_seconds": result["seconds"],
        "evidence_image_sha256": image_hash,
        "evidence_image": str(image),
        "scheduler_task": str(task_path),
        "scheduler_task_sha256": sha(task_path),
        "uart_launch_cycles": int(launch_match.group(1), 0) if launch_match else None,
        "uart_launch_returned": bool(re.search(r"launch_returned=1(?:\s|$)", uart)),
        "uart_C_oracle_errors": int(oracle_match.group(1), 16)
        if oracle_match
        else None,
        "downloaded_outputs": downloads,
    }


def manifest_match(current, record):
    """Object and adapter identity are mandatory; IR/BIN identity is explicit."""
    expected = record["artifact_sha256"]
    for key in ("kernel.o", "adapter.c"):
        require(current["artifact_sha256"][key] == expected[key], key + " differs")
        require(
            sha(
                Path(
                    record.get(
                        "artifact_directory", record.get("source_artifact_directory")
                    )
                )
                / key
            )
            == expected[key],
            "preserved evidence artifact differs: " + key,
        )
    return {
        "evidence_kernel_o_sha256": expected["kernel.o"],
        "evidence_adapter_sha256": expected["adapter.c"],
        "current_adapter_matches_evidence": True,
        "current_emitted_ir_matches_evidence_manifest": current["artifact_sha256"][
            "kernel.boscame.mlir"
        ]
        == expected.get("kernel.boscame.mlir"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=MODEL / "build/operator-suite/current99-20261104"
    )
    parser.add_argument(
        "--markdown", type=Path, default=MODEL / "CURRENT99_RUNTIME_COVERAGE.md"
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("current_cases", MODEL / "cases.py")
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    inventory = cases.inventory()
    require(len(inventory) == 99, "expected 99 current operator specializations")
    before = {case["name"]: snapshot(case) for case in inventory}
    paths = {
        "conv_publication": "build/operator-suite/conv-patch-phase-current-20261104/publication-manifest.json",
        "conv_board": "build/llvm-phase-all18-validation-20261104/board/result-summary.json",
        "conv_numeric": "build/llvm-phase-all18-validation-20261104/numerical/numerical-report.json",
        "silu_publication": "build/operator-suite/silu-direct-output-20261104/publication-manifest.json",
        "silu_board": "build/board-runs/silu-direct-output-20261104/result-summary.json",
        "silu_numeric": "build/operator-suite/silu-direct-output-20261104/board-verification.json",
        "gap_publication": "build/operator-suite/gap-torch-order-current-20261104/publication-manifest.json",
        "gap_board": "build/model-integration/gap-torch-order-board/result-summary.json",
        "historical_manifest": "build/operator-suite/final-protocol-frozen/suite-manifest.json",
        "historical_merged": "build/board-runs/all99-final-merged/result-summary.json",
        "pointwise_manifest": "build/operator-suite/pointwise37-memory-frozen/suite-manifest.json",
        "quantize_publication": "build/operator-suite/quantize-scalar-max-current-20261104/publication-manifest.json",
        "quantize_numeric": "build/operator-suite/quantize-scalar-max-current-20261104/board-verification.json",
        "quantize_nonzero_manifest": "build/operator-suite/quantize-scalar-max-current-20261104/nonzero-suite/suite-manifest.json",
    }
    documents = {key: read(MODEL / path) for key, path in paths.items()}
    cp = indexed(documents["conv_publication"])
    cb = indexed(documents["conv_board"], "results")
    cn = {row["case"]: row for row in documents["conv_numeric"]}
    sp = indexed(documents["silu_publication"], "results")
    sb = indexed(documents["silu_board"], "results")
    sn = indexed(documents["silu_numeric"], "results")
    gp = indexed(documents["gap_publication"])
    gb = indexed(documents["gap_board"], "results")
    hm = indexed(documents["historical_manifest"])
    hb = indexed(documents["historical_merged"])
    pm = indexed(documents["pointwise_manifest"])
    rows, failures = [], []
    for case in inventory:
        name, family = case["name"], case["family"]
        current = before[name]
        current_hash = current["artifact_sha256"]["kernel.o"]
        row = {"case": name, "family": family, "current": current}
        try:
            if name in cp:
                publication, board, numeric = cp[name], cb[name], cn[name]
                profile = read(board["manifest_path"])
                require(
                    current_hash
                    == publication["kernel_o_sha256"]
                    == numeric["production_kernel_object_sha256"]
                    == profile["kernel_object_sha256"]
                    == sha(
                        MODEL
                        / "build/diagnostics/conv-im2col-phase-all18"
                        / name
                        / "kernel.o"
                    ),
                    "convolution object mismatch",
                )
                require(
                    current["artifact_sha256"]["adapter.c"]
                    == publication["adapter_sha256"]
                    == sha(
                        MODEL
                        / "build/diagnostics/conv-im2col-phase-all18"
                        / name
                        / "adapter.c"
                    ),
                    "convolution default adapter mismatch",
                )
                require(
                    board["task_id"] == numeric["task_id"] == publication["board_task"],
                    "convolution numeric/task mismatch",
                )
                require(
                    current["artifact_sha256"][name + ".bin"]
                    == publication["BIN_sha256"],
                    "convolution default BIN changed",
                )
                row.update(
                    board_record(
                        board, board["evidence_directory"], board["image_snapshot"]
                    )
                )
                require(
                    numeric["status"] == "PASS"
                    and numeric["full_grid"]
                    and numeric["bit_mismatches"] == 0
                    and numeric["guard_check"]
                    and numeric["original_f32_quantization"]["bound_pass"],
                    "convolution numeric failure",
                )
                require(
                    next(x for x in row["downloaded_outputs"] if x["name"] == "Out")[
                        "sha256"
                    ]
                    == numeric["output_sha256"]
                    == numeric["oracle_sha256"],
                    "convolution output mismatch",
                )
                row.update(
                    evidence_group="current_per_patch_ddr_phase_conv18",
                    evidence_kernel_o_sha256=current_hash,
                    evidence_adapter_sha256=profile["profile_adapter_sha256"],
                    harness="nonzero full-grid convolution profile probe; common profile adapter",
                    cycle_kind="launch_cycles",
                    cycles=numeric["launch_cycles"],
                    adapter_cycles=numeric["adapter_cycles"],
                    numeric_scope="full DDR output bit-exact W8A8 oracle; guards; original F32 quantization bound",
                    board_full_numeric_pass=True,
                    numeric_elements=numeric["elements"],
                    numeric_max_abs_error=0,
                    numeric_report=str(MODEL / paths["conv_numeric"]),
                )
            elif name in sp:
                publication, board, numeric = sp[name], sb[name]["final"], sn[name]
                row.update(manifest_match(current, publication))
                row.update(
                    board_record(
                        board, board["evidence_directory"], board["image_snapshot"]
                    )
                )
                require(
                    numeric["status"] == "PASS"
                    and numeric["board_C_oracle_errors"] == 0,
                    "SiLU C oracle failed",
                )
                require(
                    board["task_id"]
                    == numeric["task_id"]
                    == publication["board_task_id"]
                    and row["uart_C_oracle_errors"] == 0,
                    "SiLU numeric/task mismatch",
                )
                row.update(
                    evidence_group="current_direct_output_native_silu20",
                    harness="standard make bin C launcher",
                    cycle_kind="launch_cycles",
                    cycles=board["launch_cycles"],
                    numeric_scope="all outputs checked on FPGA by C oracle; deterministic x=(i%31)/128; no DDR output download",
                    board_full_numeric_pass=True,
                    numeric_max_abs_error=numeric["max_abs_error"],
                    numeric_report=str(MODEL / paths["silu_numeric"]),
                )
            elif name in gp:
                publication, board = gp[name], gb[name]
                require(
                    current_hash
                    == publication["kernel_o_sha256"]
                    == board["kernel_o_sha256"],
                    "GAP object mismatch",
                )
                require(
                    sha(Path(publication["source"]) / "kernel.o") == current_hash
                    and board["task_id"] == publication["board_task"],
                    "GAP publication mismatch",
                )
                directory = Path(board["downloads"][0]["path"]).parent.parent
                row.update(
                    board_record(
                        board, directory, directory / "efficientnet_b3_model.bin"
                    )
                )
                comparison = next(
                    x for x in board["comparisons"] if x["name"] == "logits_f32"
                )
                require(
                    comparison["bit_exact"] and comparison["max_abs_error"] == 0,
                    "GAP numeric failure",
                )
                require(
                    next(
                        x
                        for x in row["downloaded_outputs"]
                        if x["name"] == "logits_f32"
                    )["sha256"]
                    == comparison["expected_sha256"],
                    "GAP output mismatch",
                )
                cycles_output = next(
                    x
                    for x in row["downloaded_outputs"]
                    if x["name"] == "call_cycles_u64"
                )
                require(
                    int.from_bytes(Path(cycles_output["path"]).read_bytes(), "little")
                    == board["call_cycles"],
                    "GAP downloaded cycle mismatch",
                )
                row.update(
                    evidence_group="current_torch_order_native_gap13",
                    evidence_kernel_o_sha256=current_hash,
                    harness="independent full-shape one-op GAP model probe",
                    cycle_kind="call_cycles",
                    cycles=board["call_cycles"],
                    model_probe_cycles=board["model_cycles"],
                    numeric_scope="full DDR output bit-exact pinned Torch CPU mean; no board guard artifact",
                    board_full_numeric_pass=True,
                    numeric_elements=comparison["elements"],
                    numeric_max_abs_error=0,
                    numeric_report=str(MODEL / paths["gap_board"]),
                )
            elif family == "per_token_quantization":
                publication = indexed(documents["quantize_publication"], "results")[
                    name
                ]
                numeric = documents["quantize_numeric"]
                board, nonzero = numeric["standard_run"], numeric["nonzero_run"]
                row.update(manifest_match(current, publication))
                row.update(
                    board_record(
                        board, board["evidence_directory"], board["image_snapshot"]
                    )
                )
                nonzero_manifest = indexed(documents["quantize_nonzero_manifest"])[name]
                manifest_match(current, nonzero_manifest)
                nonzero_evidence = board_record(
                    nonzero, nonzero["evidence_directory"], nonzero["image_snapshot"]
                )
                require(
                    numeric["status"] == "PASS"
                    and numeric["errors"] == 0
                    and numeric["scenario_count"] == 4
                    and numeric["outputs"] == 6144
                    and numeric["scales"] == 4
                    and nonzero_evidence["uart_C_oracle_errors"] == 0,
                    "current scalar-max quantize nonzero C oracle failed",
                )
                require(
                    current_hash
                    == numeric["kernel_sha256"]
                    == documents["quantize_publication"]["kernel_sha256"]
                    and board["task_id"] == publication["board_task_id"]
                    and nonzero["task_id"] == publication["numeric_board_task_id"],
                    "current scalar-max quantize task/object mismatch",
                )
                require(
                    current["artifact_sha256"][name + ".bin"] == board["image_sha256"],
                    "quantize standard image is not the current default BIN",
                )
                row.update(
                    evidence_group="current_scalar_max_quantize1",
                    harness="fresh standard C launcher; separate same-object nonzero C oracle",
                    cycle_kind="launch_cycles",
                    cycles=board["launch_cycles"],
                    numeric_scope="same current object; 4 full-length fixtures, 6144 INT8 outputs and 4 scales checked by C oracle; no DDR output artifact",
                    board_full_numeric_pass=True,
                    numeric_elements=6144,
                    numeric_scenarios=4,
                    numeric_scales=4,
                    numeric_board_evidence=nonzero_evidence,
                    numeric_report=str(MODEL / paths["quantize_numeric"]),
                )
            else:
                merged = hb[name]
                summary = read(merged["source"])
                board = summary["final"]
                record = pm[name] if family == "conv2d_i8" else hm[name]
                require(
                    record["image_sha256"] == board["image_sha256"],
                    "manifest/image mismatch",
                )
                row.update(manifest_match(current, record))
                row.update(
                    board_record(
                        board, board["evidence_directory"], board["image_snapshot"]
                    )
                )
                if family == "add":
                    require(row["uart_C_oracle_errors"] == 0, "add C oracle failed")
                row.update(
                    evidence_group="historical_exact_current_pointwise37"
                    if family == "conv2d_i8"
                    else "historical_same_object_suite10",
                    harness="standard C launcher"
                    if family == "conv2d_i8"
                    else "protocol-relinked C suite launcher",
                    cycle_kind="launch_cycles",
                    cycles=board["launch_cycles"],
                    numeric_scope="all outputs checked by launcher C oracle"
                    if family == "add"
                    else "runtime completion smoke; separate emitted-IR/backend numeric evidence",
                    board_full_numeric_pass=family == "add",
                    historical_ir_numeric_evidence=summary.get(
                        "numerical_evidence", []
                    ),
                    evidence_manifest=str(
                        MODEL
                        / paths[
                            "pointwise_manifest"
                            if family == "conv2d_i8"
                            else "historical_manifest"
                        ]
                    ),
                )
            row["current_kernel_object_matches_evidence"] = (
                row["evidence_kernel_o_sha256"] == current_hash
            )
            if row["cycle_kind"] == "launch_cycles":
                require(
                    row["uart_launch_returned"]
                    and row["cycles"] == row["uart_launch_cycles"],
                    "recorded launch cycles differ from original scheduler UART",
                )
            row["current_default_bin_sha256"] = current["artifact_sha256"][
                name + ".bin"
            ]
            row["current_default_bin_matches_executed_image"] = (
                row["current_default_bin_sha256"] == row["evidence_image_sha256"]
            )
            row["seconds_at_5529600hz"] = row["cycles"] / HZ
            row["user_60s_threshold_exceeded"] = row["cycles"] > 60 * HZ
            row["classification"] = (
                "OVER_60S_RETURNED"
                if row["user_60s_threshold_exceeded"]
                else "RETURNED_WITHIN_60S"
            )
            row["status"] = "PASS"
        except (KeyError, ValueError, OSError, StopIteration) as error:
            row.update(status="EVIDENCE_GAP", error=str(error))
            failures.append({"case": name, "error": str(error)})
        rows.append(row)
    after = {case["name"]: snapshot(case) for case in inventory}
    pointwise = [case["name"] for case in inventory if case["family"] == "conv2d_i8"]
    require(
        len(pointwise) == 37 and all(before[name] == after[name] for name in pointwise),
        "pointwise files changed during collection",
    )
    require(
        before == after,
        "default files changed during collection; rerun on stable artifacts",
    )
    passed = [r for r in rows if r["status"] == "PASS"]
    slow = [r for r in passed if r["user_60s_threshold_exceeded"]]
    historical48 = [
        {
            "case": row["case"],
            "family": row["family"],
            "current_kernel_o_sha256": row["current"]["artifact_sha256"]["kernel.o"],
            "old_frozen_kernel_o_sha256": hm[row["case"]]["artifact_sha256"][
                "kernel.o"
            ],
            "current_object_equals_old_frozen": row["current"]["artifact_sha256"][
                "kernel.o"
            ]
            == hm[row["case"]]["artifact_sha256"]["kernel.o"],
            "current_BIN_equals_old_frozen": row["current"]["artifact_sha256"][
                row["case"] + ".bin"
            ]
            == hm[row["case"]]["image_sha256"],
            "actual_current_object_runtime_evidence": row.get("evidence_group"),
            "task_id": row.get("task_id"),
        }
        for row in rows
        if row["case"] not in cp and row["case"] not in sp and row["case"] not in gp
    ]
    require(len(historical48) == 48, "expected 48 remaining operator specializations")
    report = {
        "schema": 1,
        "status": "PASS" if not failures else "EVIDENCE_GAPS",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": "fpga112",
        "scope": "current default object runtime coverage assembled from actual preserved jobs; mixed harnesses; not uniform 99 default BIN rerun; not whole-model execution",
        "frequency_hz": HZ,
        "acceptable_seconds": 60,
        "unique_current_cases": len(rows),
        "current_object_runtime_covered": len(passed),
        "current_default_exact_bin_runtime_covered": sum(
            r["current_default_bin_matches_executed_image"] for r in passed
        ),
        "same_object_different_harness_bin": sum(
            not r["current_default_bin_matches_executed_image"] for r in passed
        ),
        "classification_counts": dict(Counter(r["classification"] for r in passed)),
        "evidence_group_counts": dict(Counter(r["evidence_group"] for r in passed)),
        "numerical_scope_counts": dict(Counter(r["numeric_scope"] for r in passed)),
        "conv_full_ddr_numeric_elements": sum(
            r.get("numeric_elements", 0) for r in passed if r["case"] in cp
        ),
        "gap_full_ddr_numeric_elements": sum(
            r.get("numeric_elements", 0) for r in passed if r["case"] in gp
        ),
        "fresh_board_submissions_by_collector": 0,
        "pointwise37_preserved": True,
        "historical_frozen48_comparison": historical48,
        "pointwise37_before_and_after": {name: before[name] for name in pointwise},
        "input_reports": {
            key: {"path": str(MODEL / value), "sha256": sha(MODEL / value)}
            for key, value in paths.items()
        },
        "slow_cases": [
            {
                key: row[key]
                for key in (
                    "case",
                    "cycles",
                    "cycle_kind",
                    "seconds_at_5529600hz",
                    "task_id",
                    "fpga",
                    "harness",
                )
            }
            for row in slow
        ],
        "failures": failures,
        "cases": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "case",
        "family",
        "status",
        "evidence_group",
        "classification",
        "cycles",
        "cycle_kind",
        "seconds_at_5529600hz",
        "scheduler_seconds",
        "task_id",
        "fpga",
        "current_kernel_object_matches_evidence",
        "current_default_bin_matches_executed_image",
        "current_default_bin_sha256",
        "evidence_image_sha256",
        "harness",
        "board_full_numeric_pass",
        "numeric_elements",
        "numeric_scope",
    ]
    with (args.output / "report.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    lines = [
        "# Current 99 operator runtime coverage",
        "",
        f"Current default `kernel.o` runtime coverage: **{len(passed)}/99**. "
        f"Exact current default BIN executions: **{report['current_default_exact_bin_runtime_covered']}/99**. "
        f"The other {report['same_object_different_harness_bin']} cases use the exact current object with a different diagnostic/protocol launcher. "
        "This collector reuses actual jobs and submits no new jobs; this is neither a uniform 99-BIN rerun nor whole-model evidence.",
        "",
        "Convolution evidence uses the current per-patch DDR phase objects; GAP evidence uses the current pinned Torch-order mean objects; "
        "quantize uses the current scalar-max object. SiLU uses the corrected direct-output objects and 20 fresh fpga112 standard-BIN runs. "
        "The previous report before these convolution/GAP/quantize updates is preserved in "
        "`build/operator-suite/current99-pre-conv-ddr-phase-20261104/`; the older pre-SiLU snapshot is in "
        "`build/operator-suite/current99-pre-direct-output-20261104/`. "
        "The separate preserved-full-layout first-SiLU capture and its limited scope are documented in "
        "[SILU_CONTEXT_DIAGNOSIS.md]("
        + os.path.relpath(
            MODEL / "SILU_CONTEXT_DIAGNOSIS.md", args.markdown.resolve().parent
        )
        + ").",
        "",
        "Reproduce from the model directory:",
        "",
        "```bash",
        "python diagnostics/operator_suite/collect_current99.py",
        "```",
        "",
        "Detailed current hashes, original scheduler terminal states, task IDs, image snapshots, "
        "download hashes and harness scopes are in "
        "`" + str(args.output / "report.json") + "` and `report.csv`.",
        "",
        "| Evidence group | Cases | Scope |",
        "| --- | ---: | --- |",
        "| Current per-patch DDR phase convolution | 18 | Same current kernel.o; nonzero full-grid profile probe; all 6,725,232 DDR outputs bit-exact; guards and original F32 quantization bounds pass |",
        "| Current direct-output native RVV SiLU | 20 | Fresh exact standard make BIN; all launcher outputs checked by C oracle; no external DDR output artifact |",
        "| Current native Torch-order GAP | 13 | Same kernel.o; independent full-shape one-op model probe; all 8,608 DDR outputs bit-exact to pinned Torch CPU mean; no board guard artifact |",
        "| Existing INT8 pointwise | 37 | Exact current kernel.o, adapter and BIN; historical actual executions, including 28 memory lowering overlay jobs |",
        "| Add / classifier / matmul / dequantize | 10 | Exact current objects; classifier exact BIN, remaining 9 protocol-relinked launcher BINs |",
        "| Activation quantize | 1 | Fresh scalar-max default object/BIN; same object also checks 4 full-length nonzero/rounding fixtures, 6,144 INT8 outputs and 4 scales by C oracle |",
        "",
        "Cycles use 5,529,600 Hz. Convolution and SiLU `launch_cycles` include launcher work. "
        "GAP `call_cycles` measure one complete native adapter call; its model trace wrapper has separate `model_probe_cycles`. "
        "The columns are labeled individually and should not be compared as uniform pure-kernel timing.",
        "",
        f"**{len(slow)} cases exceed the user’s 60-second threshold; all eventually returned and exited 0.** "
        "They remain above-threshold/hang under the requested 60-second convention; a later return is not proof of an infinite hardware hang.",
        "",
        "| Case | Launch cycles | Seconds | Task | FPGA |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    lines += [
        f"| `{r['case']}` | {r['cycles']:,} | {r['seconds_at_5529600hz']:.3f} | `{r['task_id']}` | {r['fpga']} |"
        for r in slow
    ]
    lines += [
        "",
        "The 37 pointwise source, Makefile and default object/adapter/IR/BIN hashes were checked before and after collection and preserved. "
        "Runtime coverage does not imply identical numerical coverage for every operator. Separate IR/backend numerical evidence is retained for the other families. "
        "The quantize fixture is bound to the current scalar-max object; it covers its four declared fixtures and does not establish arbitrary-input model accuracy.",
        "",
    ]
    if failures:
        lines += ["Evidence gaps:", ""] + [
            f"- `{r['case']}`: {r['error']}" for r in failures
        ]
    args.markdown.write_text("\n".join(lines))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "current_object_runtime_covered",
                    "current_default_exact_bin_runtime_covered",
                    "classification_counts",
                    "failures",
                )
            },
            indent=2,
        )
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

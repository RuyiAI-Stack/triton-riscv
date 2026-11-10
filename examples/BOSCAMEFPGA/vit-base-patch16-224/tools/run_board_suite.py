#!/usr/bin/env python3
"""Run existing board binaries sequentially through the user-owned scheduler.

Every actual submission is python ../run_bin.py IMAGE --timeout SECONDS. SDK use below is
read-only: it retrieves task evidence after that process finishes.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402


def execution_timeout(value):
    seconds = int(value)
    if not 1 <= seconds <= 600:
        raise argparse.ArgumentTypeError("timeout must be an integer between 1 and 600")
    return seconds


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def snapshot_validation_evidence(image, build_dir, destination):
    """Recognize embedded references, including after GOLDEN mode changes."""
    record = {"validation_mode": "on_board_scalar_oracle"}
    accumulator = build_dir / "accumulator-manifest.json"
    if accumulator.exists():
        manifest = json.loads(accumulator.read_text())
        offset = 0x100000
        matched = True
        with image.open("rb") as binary:
            for entry in manifest["rounds"]:
                golden = (build_dir / entry["file"]).read_bytes()
                binary.seek(offset)
                matched &= (
                    binary.read(len(golden)) == golden
                    and hashlib.sha256(golden).hexdigest() == entry["output_sha256"]
                )
                offset = (offset + len(golden) + 63) & ~63
        if matched:
            shutil.copyfile(accumulator, destination / accumulator.name)
            by_name = {entry["name"]: entry for entry in manifest["rounds"]}
            round_ids = manifest.get("round_ids", [0, 1, 2])
            return dict(
                validation_mode="accumulator_three_round_bit_exact",
                validation_tolerance=0,
                validation_round_ids=round_ids,
                expected_output_fnv64=[
                    by_name["normal" if index == 0 else "stress"]["output_fnv64"]
                    for index in round_ids
                ],
            )
    manifest_path = build_dir / "golden_manifest.json"
    golden_path = build_dir / "golden.bin"
    if not manifest_path.exists() or not golden_path.exists():
        return record
    golden = golden_path.read_bytes()
    # board_golden.ld places .vit_golden at 0x80100000; flat bins start
    # at 0x80000000. A stale manifest alone cannot establish GOLDEN=1.
    with image.open("rb") as binary:
        binary.seek(0x100000)
        embedded = binary.read(len(golden))
    if not golden or embedded != golden:
        return record
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["rounds"][0]["outputs"][0]
    digest = hashlib.sha256(golden).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("embedded golden differs from its provenance manifest")
    shutil.copyfile(manifest_path, destination / "golden_manifest.json")
    record.update(
        validation_mode="precomputed_original_scalar_oracle",
        embedded_golden_sha256=digest,
        validation_tolerance=expected["tolerance"],
    )
    return record


def snapshot_build_evidence(kernels, name, destination):
    """Keep generated route/code and linker inputs beside the submitted image."""
    case = kernels / name
    board = case / "board"
    files = {
        filename: case / filename
        for filename in (
            "kernel.planned.mlir",
            "kernel.boscame.mlir",
            "kernel.nr.s",
            "kernel.o",
            "frontend.json",
            ".compiler-inputs.json",
            "build-result.json",
        )
    }
    for filename in (
        "adapter.c",
        "board_wrapper.c",
        "board_config.json",
        "accumulator-launch.c",
        "launch_board.c",
        "board_launch.c",
        "launch.o",
        "adapter.o",
        "wrapper.o",
        name + ".elf",
        name + ".map",
    ):
        files["board-" + filename] = board / filename
    for filename in (
        "identity-before.json",
        "identity-after.json",
        "frozen-tools.json",
        "images.json",
        "build-validation.json",
    ):
        files[filename] = kernels.parent / filename
    snapshot = kernels.parent / "source-snapshot"
    if snapshot.is_dir():
        for path in snapshot.iterdir():
            if path.is_file():
                files["source-snapshot-" + path.name] = path
    evidence = destination / "build-evidence"
    evidence.mkdir(exist_ok=True)
    hashes = {}
    for label, original in files.items():
        if original.is_file():
            shutil.copyfile(original, evidence / label)
            hashes[label] = hashlib.sha256((evidence / label).read_bytes()).hexdigest()
    atomic_json(evidence / "sha256.json", hashes)
    return hashes


def run_case(name, destination, kernels=None, timeout=120):
    from fpga_scheduler import Client

    case_dir = destination / name
    case_dir.mkdir(parents=True, exist_ok=True)
    kernels = Path(kernels).resolve() if kernels is not None else ROOT / "build/kernels"
    original = kernels / name / "board" / (name + ".bin")
    record = dict(
        case=name,
        status="BUILD_MISSING",
        kernel_cycles=[],
        execution_timeout_seconds=timeout,
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    if not original.is_file() or original.stat().st_size == 0:
        atomic_json(case_dir / "result.json", record)
        return record
    image = case_dir / (name + ".bin")
    shutil.copyfile(original, image)
    record["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    record["build_evidence_sha256"] = snapshot_build_evidence(kernels, name, case_dir)
    config = original.parent / "board_config.json"
    record["board_config"] = (
        json.loads(config.read_text())
        if config.exists()
        else {
            "expected_invocations": 1,
            "expected_checked": 768,
            "cycle_scope": "full typed adapter, including DMA and final fence; excludes initialization, oracle and UART",
        }
    )
    record.update(snapshot_validation_evidence(image, original.parent, case_dir))
    for filename in ("elf-audit.json", "board_config.json", ".board-inputs.json"):
        path = original.parent / filename
        if path.exists():
            shutil.copyfile(path, case_dir / filename)
    command = [
        sys.executable,
        str(ROOT.parent / "run_bin.py"),
        str(image),
        "--timeout",
        str(timeout),
    ]
    record["command"] = command
    record["status"] = "SUBMITTING"
    atomic_json(case_dir / "result.json", record)
    print("RUN " + name, flush=True)
    started = time.monotonic()
    lines = []
    with (
        (case_dir / "console.log").open("w") as log,
        (case_dir / "events.jsonl").open("w") as events,
    ):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            events.write(
                json.dumps(dict(seconds=time.monotonic() - started, text=line)) + "\n"
            )
            events.flush()
            task_match = re.search(r"\[FPGA\] Task ([0-9a-f]{32})", line)
            if task_match:
                record.update(task_id=task_match[1], status="SUBMITTED")
                atomic_json(case_dir / "result.json", record)
            print(name + ": " + line.rstrip(), flush=True)
        record["process_returncode"] = process.wait()
    record["wall_seconds_with_queue"] = time.monotonic() - started
    output = "".join(lines)
    match = re.search(r"\[FPGA\] Task ([0-9a-f]{32})", output)
    if match:
        record["task_id"] = match[1]
        with Client(
            "http://127.0.0.1:18765",
            token_file=str(ROOT.parent / "client.token"),
            request_timeout=30,
        ) as client:
            task = client.job(match[1])
        atomic_json(case_dir / "task.json", asdict(task))
        result = task.result
        record.update(
            node=task.node,
            fpga=task.fpga,
            execution_status=result.execution_status,
            app_exit_code=result.app_exit_code,
            scheduler_seconds=result.seconds,
            details=result.details,
        )
        uart = result.uart_output if result.uart_output is not None else result.output
        (case_dir / "uart.log").write_text(uart)
        assert result.details.get("image_sha256") == record["image_sha256"], (
            "submitted image hash mismatch"
        )
        record["kernel_cycles"] = [
            int(x, 16) for x in re.findall(r"kernel_cycles=0x([0-9a-fA-F]+)", uart)
        ]
        record["kernel_cycles_raw"] = list(record["kernel_cycles"])
        rounds = re.findall(
            r"validation_round=(\d+) checked=(\d+) errors=(\d+) guard_errors=(\d+) input_errors=(\d+) output_fnv64=0x([0-9a-fA-F]+)",
            uart,
        )
        record["validation_rounds"] = [
            dict(
                round=int(r[0]),
                checked=int(r[1]),
                errors=int(r[2]),
                guard_errors=int(r[3]),
                input_errors=int(r[4]),
                output_fnv64=r[5].lower().zfill(16),
            )
            for r in rounds
        ]
        check = re.search(
            r"kernel_returned=(\d+)(?: validation_executed=(\d+))? checked=(\d+) errors=(\d+) max_abs_error_f32_bits=0x([0-9a-fA-F]+)",
            uart,
        )
        if check:
            record.update(
                kernel_returned=int(check[1]),
                validation_executed=int(check[2] or 1),
                checked=int(check[3]),
                errors=int(check[4]),
                max_abs_error=struct.unpack("!f", bytes.fromhex(check[5].zfill(8)))[0],
            )
            record["kernel_cycles"] = record["kernel_cycles"][
                : record["kernel_returned"]
            ]
        expected = record["board_config"]
        if result.execution_status == "timeout":
            record["status"] = "TIMEOUT" if uart.strip() else "HANG"
            record["last_output"] = (
                uart.strip().splitlines()[-1] if uart.strip() else ""
            )
        elif result.execution_status == "trap":
            record["status"] = "TRAP"
        elif (
            result.ok
            and result.execution_status == "exited"
            and result.app_exit_code == 0
        ):
            complete = (
                record.get("validation_executed") == 1
                and record.get("errors") == 0
                and record.get("checked") == expected["expected_checked"]
                and record.get("kernel_returned") == expected["expected_invocations"]
                and len(record["kernel_cycles_raw"]) == expected["expected_invocations"]
                and len(record["kernel_cycles"]) == expected["expected_invocations"]
            )
            if expected.get("validation_mode") == "accumulator_three_round_bit_exact":
                expected_ids = expected.get("validation_round_ids", [0, 1, 2])
                expected_hashes = record.get("expected_output_fnv64", [])
                complete &= (
                    record["validation_mode"] == expected["validation_mode"]
                    and record.get("validation_round_ids") == expected_ids
                    and [r["round"] for r in record["validation_rounds"]]
                    == expected_ids
                    and len(expected_hashes) == len(expected_ids)
                )
                for validation, expected_hash in zip(
                    record["validation_rounds"], expected_hashes
                ):
                    complete &= (
                        validation["checked"] == expected["expected_checked_per_round"]
                        and validation["errors"]
                        == validation["guard_errors"]
                        == validation["input_errors"]
                        == 0
                        and validation["output_fnv64"] == expected_hash
                    )
            record["status"] = "PASS" if complete else "INCOMPLETE_EVIDENCE"
        elif result.execution_status == "exited":
            record["status"] = "FAIL"
        else:
            record["status"] = result.execution_status.upper()
    else:
        record.update(status="SUBMISSION_ERROR", diagnostic=output[-4000:])
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(case_dir / "result.json", record)
    print(
        "RESULT "
        + name
        + " "
        + record["status"]
        + " cycles="
        + str(record["kernel_cycles"]),
        flush=True,
    )
    return record


def write_report(destination):
    records = [
        json.loads(p.read_text()) for p in sorted(destination.glob("*/result.json"))
    ]
    counts = {}
    for record in records:
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    timeouts = sorted(
        {
            record["execution_timeout_seconds"]
            for record in records
            if "execution_timeout_seconds" in record
        }
    )
    missing_timeout = sum(
        "execution_timeout_seconds" not in record for record in records
    )
    atomic_json(
        destination / "report.json",
        dict(
            counts=counts,
            operators=records,
            execution_timeouts_seconds=timeouts,
            timeout_metadata_missing=missing_timeout,
        ),
    )
    timeout_text = ", ".join(str(seconds) for seconds in timeouts) or "not recorded"
    if missing_timeout:
        timeout_text += f"; {missing_timeout} saved record(s) lack deadline metadata"
    lines = [
        "# ViT FPGA operator board results",
        "",
        f"Execution: user-provided `run_bin.py`, sequential submissions; execution deadline(s) in seconds: {timeout_text}.",
        "Timeout with UART output is TIMEOUT; timeout without UART is HANG.",
        "Scheduler budget includes image load/reset; queue wait is separate.",
        "Cycles measure complete typed adapter plus DMA and final fence, excluding input initialization and oracle.",
        "",
        "| Operator | Status | Deadline (s) | Cycles per invocation | Checked | Errors |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in records:
        cycles = ", ".join(f"{c:,}" for c in r["kernel_cycles"]) or "unavailable"
        lines.append(
            f"| {r['case']} | {r['status']} | {r.get('execution_timeout_seconds', '—')} | "
            f"{cycles} | {r.get('checked', '—')} | {r.get('errors', '—')} |"
        )
    lines += [
        "",
        "Each operator directory preserves the submitted binary/hash, UART, task details and result JSON.",
        "Precomputed references, when embedded, are original launch.c scalar oracle outputs; the board compares every result.",
        "Partial cycles on TIMEOUT/TRAP do not establish correctness.",
    ]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        type=Path,
        default=ROOT / "build/kernels",
        help="Use binaries from this fresh build root",
    )
    parser.add_argument(
        "--timeout",
        type=execution_timeout,
        default=120,
        help="scheduler execution deadline in seconds (1..600; queue wait is separate)",
    )
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    destination = args.report_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not args.collect:
        for name in args.case or [c["name"] for c in inventory()]:
            describe(name)
            run_case(name, destination, args.kernels, args.timeout)
            write_report(destination)
    write_report(destination)


if __name__ == "__main__":
    main()

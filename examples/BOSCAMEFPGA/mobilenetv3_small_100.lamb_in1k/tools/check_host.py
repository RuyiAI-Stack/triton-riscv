#!/usr/bin/env python3
"""Run the emitted LLVM kernels and their adapters against each C oracle.

This checks compiler numerics on the host; it does not validate FPGA codegen,
instructions, runtime, or timing. No reference implementation replaces a kernel.
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

from validation import input_hashes, operator_pass

ROOT = Path(__file__).resolve().parents[1]


def check(case, kernels, output, clang):
    directory = output / case
    directory.mkdir(parents=True, exist_ok=True)
    inputs = [
        kernels / case / "kernel.ll",
        kernels / case / "adapter.c",
        ROOT / case / "launch.c",
        ROOT / "support.c",
        ROOT / "support.h",
    ]
    record = {"case": case, "status": "FAIL", "execution": "host-emitted-LLVM"}
    try:
        record["inputs_sha256"] = input_hashes(inputs)
        main = directory / "main.c"
        main.write_text(
            "extern int launch(void);\nint main(void) { return launch(); }\n"
        )
        command = [
            clang,
            "--target=x86_64-unknown-linux-gnu",
            "-O2",
            "-ffp-contract=off",
            "-DHOST_TEST",
            "-I",
            str(ROOT),
            *map(str, inputs[:-1]),
            str(main),
            "-lm",
            "-o",
            str(directory / "check"),
        ]
        record["compile_command"] = command
        compile_result = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
        (directory / "compile.log").write_text(
            compile_result.stdout + compile_result.stderr
        )
        record["compile_exit_code"] = compile_result.returncode
        if compile_result.returncode == 0:
            result = subprocess.run(
                [str(directory / "check")], capture_output=True, text=True, timeout=180
            )
            (directory / "run.log").write_text(result.stdout + result.stderr)
            record.update(exit_code=result.returncode, output=result.stdout)
            if result.returncode == 0 and operator_pass(result.stdout, case):
                record["status"] = "PASS"
        if record["inputs_sha256"] != input_hashes(inputs):
            record.update(status="FAIL", error="Input files changed during validation")
    except Exception as error:
        record["error"] = str(error)
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--output", type=Path, default=ROOT / "build/host-check")
    parser.add_argument(
        "--clang", default=str(Path(os.environ.get("LLVM_BIN", "/usr/bin")) / "clang")
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cases = sorted(args.case or [p.parent.name for p in ROOT.glob("*/kernel.py")])
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [
            pool.submit(
                check, c, args.kernels.resolve(), args.output.resolve(), args.clang
            )
            for c in cases
        ]
        for task in concurrent.futures.as_completed(tasks):
            record = task.result()
            records.append(record)
            print(record["status"], record["case"], flush=True)
    failed = sum(r["status"] != "PASS" for r in records)
    report = {
        "execution": "host-emitted-LLVM",
        "board_validated": False,
        "total": len(records),
        "passed": len(records) - failed,
        "failed": failed,
        "cases": sorted(records, key=lambda r: r["case"]),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(records) - failed}/{len(records)} host checks passed")
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())

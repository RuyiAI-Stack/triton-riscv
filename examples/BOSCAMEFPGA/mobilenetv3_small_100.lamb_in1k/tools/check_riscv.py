#!/usr/bin/env python3
"""Validate emitted RISC-V kernel.o files with Linux adapters under QEMU.

The original machine-code kernels are linked without recompilation. C oracles,
adapters and model runners are cross-compiled for a static Linux executable.
These checks provide numerical evidence for the RISC-V objects, not FPGA
runtime, AME hardware, firmware or performance validation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from validation import compare_logits, operator_pass, validate_model_inputs

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def input_hashes(paths):
    return {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
        for p in paths
    }


def run(command, log, timeout, environment=None):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, env=environment
    )
    Path(log).write_text(result.stdout + result.stderr)
    return result


def compiler_flags(cc):
    return [
        cc,
        "-march=rv64imafdc_zicsr_zifencei",
        "-mabi=lp64d",
        "-O2",
        "-ffp-contract=off",
        "-fno-tree-vectorize",
        "-DHOST_TEST",
        "-I" + str(ROOT),
        "-I" + str(ROOT / "model"),
    ]


def check_operator(case, args):
    output = args.output / case
    output.mkdir(parents=True, exist_ok=True)
    record = {"case": case, "scope": "QEMU", "board_execution": False, "status": "FAIL"}
    try:
        inputs = [
            args.kernels / case / "kernel.o",
            args.kernels / case / "adapter.c",
            ROOT / case / "launch.c",
            ROOT / "support.c",
            ROOT / "support.h",
        ]
        record["inputs_sha256"] = input_hashes(inputs)
        main = output / "main.c"
        main.write_text(
            "extern int launch(void);\nint main(void) { return launch(); }\n"
        )
        executable = output / "check-riscv"
        command = compiler_flags(args.cc) + [
            "-static",
            "-no-pie",
            "-Wl,--no-relax",
            *map(str, inputs[:-1]),
            str(main),
            "-lm",
            "-o",
            str(executable),
        ]
        record["compile_command"] = command
        result = run(command, output / "compile.log", args.timeout)
        record["compile_exit_code"] = result.returncode
        if result.returncode == 0:
            record["executable_sha256"] = digest(executable)
            command = [args.qemu, str(executable)]
            record["run_command"] = command
            start = time.monotonic()
            result = run(command, output / "run.log", args.timeout)
            record.update(
                exit_code=result.returncode,
                output=result.stdout,
                qemu_wall_seconds=time.monotonic() - start,
            )
            if result.returncode == 0 and operator_pass(result.stdout, case):
                record["status"] = "PASS"
        if record["inputs_sha256"] != input_hashes(inputs):
            record.update(status="FAIL", error="Input files changed during validation")
    except Exception as error:
        record["error"] = str(error)
    write_json(output / "result.json", record)
    return record


def check_model(model, args):
    model = model.resolve()
    output = model / "qemu"
    output.mkdir(exist_ok=True)
    report_file = model / "qemu-model-report.json"
    report_file.unlink(missing_ok=True)
    record = {
        "scope": "QEMU",
        "execution": "original RISC-V kernel.o with Linux adapters",
        "board_execution": False,
        "passed": False,
    }
    try:
        manifest_path = model / "model-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        validate_model_inputs(model, manifest)
        record["model"] = manifest["model"]
        sources = [
            ROOT / "support.c",
            ROOT / "model/model_support.c",
            model / "runner.c",
            model / "model_blob.S",
        ]
        objects = []
        for case in manifest["used_cases"]:
            objects.append(args.kernels / case / "kernel.o")
            sources.append(args.kernels / case / "adapter.c")
        tracked = (
            sources
            + objects
            + [
                ROOT / "support.h",
                ROOT / "model/model_support.h",
                manifest_path,
                model / "model-data.bin",
                model / "golden-logits.f32",
            ]
        )
        record["inputs_sha256"] = input_hashes(tracked)
        common = compiler_flags(args.cc)
        commands = []
        for index, source in enumerate(sources):
            target = output / f"{index:03d}-{source.stem}.o"
            command = common + ["-c", str(source), "-o", str(target)]
            commands.append((command, output / f"{index:03d}-compile.log"))
            objects.append(target)
        record["compile_commands"] = [c for c, _ in commands]

        def compile_one(item):
            command, log = item
            result = run(command, log, args.timeout)
            if result.returncode:
                raise RuntimeError(f"Compilation failed; see {log}")

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(compile_one, commands))
        executable = output / "mobilenet-riscv"
        command = common + [
            "-static",
            "-no-pie",
            "-Wl,--no-relax",
            *map(str, objects),
            "-lm",
            "-o",
            str(executable),
        ]
        record["link_command"] = command
        compile_one((command, output / "link.log"))
        record["executable_sha256"] = digest(executable)
        actual_path = output / "actual-logits.f32"
        actual_path.unlink(missing_ok=True)
        environment = {**os.environ, "MODEL_LOGITS_FILE": str(actual_path)}
        command = [args.qemu, str(executable)]
        record["run_command"] = command
        start = time.monotonic()
        result = run(command, output / "run.log", args.timeout, environment)
        record.update(
            returncode=result.returncode, qemu_wall_seconds=time.monotonic() - start
        )
        if result.returncode == 0 and actual_path.exists():
            record.update(
                compare_logits(actual_path, model / "golden-logits.f32", manifest)
            )
        else:
            record["output"] = result.stdout + result.stderr
        if record["inputs_sha256"] != input_hashes(tracked):
            record.update(passed=False, error="Input files changed during validation")
    except Exception as error:
        record["error"] = str(error)
    record["tools"] = args.tool_identity
    write_json(report_file, record)
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ("inputs_sha256", "compile_commands", "link_command")
            },
            indent=2,
        )
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--output", type=Path, default=ROOT / "build/qemu-check")
    parser.add_argument("--cc", default="riscv64-unknown-linux-gnu-gcc")
    parser.add_argument("--qemu", default="qemu-riscv64")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=[],
        help="Also validate this exported full model; repeat for multiple inputs",
    )
    parser.add_argument("--skip-operators", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    if args.skip_operators and not args.model:
        parser.error("--skip-operators requires at least one --model")
    args.kernels, args.output = args.kernels.resolve(), args.output.resolve()
    args.tool_identity = {}
    for name in ("cc", "qemu"):
        path = shutil.which(getattr(args, name))
        if not path:
            parser.error(f"Cannot find {getattr(args, name)}")
        setattr(args, name, path)
        args.tool_identity[name] = {
            "path": path,
            "sha256": digest(path),
            "version": subprocess.check_output(
                [path, "--version"], text=True
            ).splitlines()[0],
        }
    failures = 0
    if not args.skip_operators:
        args.output.mkdir(parents=True, exist_ok=True)
        cases = sorted(args.case or [p.parent.name for p in ROOT.glob("*/kernel.py")])
        records = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            tasks = [pool.submit(check_operator, case, args) for case in cases]
            for task in as_completed(tasks):
                record = task.result()
                records.append(record)
                print(record["status"], record["case"], flush=True)
        failures = sum(r["status"] != "PASS" for r in records)
        report = {
            "scope": "QEMU",
            "execution": "original RISC-V kernel.o with Linux adapters",
            "board_execution": False,
            "total": len(records),
            "passed": len(records) - failures,
            "failed": failures,
            "tools": args.tool_identity,
            "cases": sorted(records, key=lambda r: r["case"]),
        }
        write_json(args.output / "summary.json", report)
        print(
            f"{len(records) - failures}/{len(records)} QEMU operator checks passed",
            flush=True,
        )
    for model in args.model:
        failures += not check_model(model, args)["passed"]
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

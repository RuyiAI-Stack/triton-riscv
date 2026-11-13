#!/usr/bin/env python3
"""Execute the complete model using native code from actual Triton LLVM IR.

This validates the compiler output, ABI adapters, memory reuse and pretrained
model logits. It is host numerical evidence, not an FPGA execution claim.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from validation import compare_logits, input_hashes, validate_model_inputs

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_model(args):
    model = args.model.resolve()
    host = model / "host"
    host.mkdir(parents=True, exist_ok=True)
    report_file = model / "host-model-report.json"
    report_file.unlink(missing_ok=True)
    report = {
        "scope": "host native execution of compiled Triton LLVM IR and original adapters",
        "board_execution": False,
        "passed": False,
    }
    try:
        manifest_path = model / "model-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        validate_model_inputs(model, manifest)
        report["model"] = manifest["model"]
        common = [
            args.clang,
            "--target=x86_64-unknown-linux-gnu",
            "-O2",
            "-ffp-contract=off",
            "-DHOST_TEST",
            "-I" + str(ROOT),
            "-I" + str(ROOT / "model"),
        ]
        commands, objects, sources = [], [], []
        for case in manifest["used_cases"]:
            for suffix, source in [
                ("kernel", args.kernels / case / "kernel.ll"),
                ("adapter", args.kernels / case / "adapter.c"),
            ]:
                target = host / f"{case}.{suffix}.o"
                commands.append(common + ["-c", str(source), "-o", str(target)])
                objects.append(target)
                sources.append(source)
        for source in [
            ROOT / "support.c",
            ROOT / "model/model_support.c",
            model / "runner.c",
            model / "model_blob.S",
        ]:
            target = host / (source.stem + ".o")
            commands.append(common + ["-c", str(source), "-o", str(target)])
            objects.append(target)
            sources.append(source)
        tracked = sources + [
            ROOT / "support.h",
            ROOT / "model/model_support.h",
            manifest_path,
            model / "model-data.bin",
            model / "golden-logits.f32",
        ]
        report["sources_sha256"] = input_hashes(sources)
        report["inputs_sha256"] = input_hashes(tracked)
        report["compile_commands"] = commands

        def compile(command):
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=args.timeout
            )
            log = host / (Path(command[-1]).name + ".compile.log")
            log.write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError("Compilation failed; see " + str(log))

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(compile, commands))
        executable = host / "mobilenet-host"
        report["link_command"] = (
            common + list(map(str, objects)) + ["-lm", "-o", str(executable)]
        )
        compile(report["link_command"])
        actual_path = host / "actual-logits.f32"
        actual_path.unlink(missing_ok=True)
        environment = {**os.environ, "MODEL_LOGITS_FILE": str(actual_path)}
        start = time.monotonic()
        result = subprocess.run(
            [str(executable)],
            env=environment,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        (host / "run.log").write_text(result.stdout + result.stderr)
        report.update(
            returncode=result.returncode,
            wall_seconds=time.monotonic() - start,
            model_manifest_sha256=digest(manifest_path),
            model_blob_sha256=digest(model / "model-data.bin"),
            golden_logits_sha256=digest(model / "golden-logits.f32"),
            compiler=subprocess.check_output(
                [args.clang, "--version"], text=True, timeout=args.timeout
            ).splitlines()[0],
            executable_sha256=digest(executable),
        )
        if result.returncode == 0 and actual_path.exists():
            report.update(
                compare_logits(actual_path, model / "golden-logits.f32", manifest)
            )
        else:
            report["output"] = result.stdout + result.stderr
        if report["inputs_sha256"] != input_hashes(tracked):
            report.update(passed=False, error="Input files changed during validation")
    except Exception as error:
        report.update(passed=False, error=str(error))
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "sources_sha256",
                    "inputs_sha256",
                    "compile_commands",
                    "link_command",
                )
            },
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "build/model")
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    args.kernels = args.kernels.resolve()
    return 0 if check_model(args)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

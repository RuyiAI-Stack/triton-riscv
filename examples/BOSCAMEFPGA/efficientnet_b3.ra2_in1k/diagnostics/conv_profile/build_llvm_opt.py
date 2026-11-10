#!/usr/bin/env python3
"""Build isolated scalar LLVM-O2 candidates from frozen convolution IR.

The source Triton/Contract/BOSCAME modules and public adapter remain identical.
No production artifact is replaced. Full nonzero FPGA readback is required
before using one of these candidates in the model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
LLVM = REPO / "buddy-mlir/llvm/build/bin"
HELPERS = REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, log, stdin=None):
    proc = subprocess.run(
        list(map(str, args)), input=stdin, capture_output=True, text=True
    )
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError("command failed; see " + str(log))
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument(
        "--baseline", type=Path, default=MODEL / "build/w8a8-opt/precleared-integrated"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name in args.case:
        source = args.baseline.resolve() / name
        output = args.output.resolve() / name
        output.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name == "stages":
                shutil.copytree(item, output / item.name, dirs_exist_ok=True)
            elif item.name in (
                "adapter.c",
                "kernel.ttir",
                "kernel.frontend.mlir",
                "kernel.bufferized.mlir",
                "kernel.planned.mlir",
                "kernel.boscame.mlir",
                "contract-manifest.json",
            ):
                shutil.copy2(item, output / item.name)
        run(
            [
                LLVM / "opt",
                "-S",
                "-passes=default<O2>",
                "-vectorize-loops=false",
                "-vectorize-slp=false",
                source / "kernel.ll",
                "-o",
                output / "kernel.ll",
            ],
            output / "opt.log",
        )
        run(
            [
                LLVM / "llc",
                output / "kernel.ll",
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                output / "kernel.s",
            ],
            output / "llc.log",
        )
        encoded = run(
            [
                sys.executable,
                HELPERS / "ame_to_word.py",
                "--gpr-mode=fixed",
                "--preserve-matrix-registers",
            ],
            output / "encode.log",
            (output / "kernel.s").read_text(),
        )
        (output / "kernel.encoded.s").write_text(encoded)
        restricted = run(
            [
                sys.executable,
                HELPERS / "restrict_fpga_assembly.py",
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
            output / "restrict.log",
            encoded,
        )
        (output / "kernel.nr.s").write_text(restricted)
        run(
            [
                LLVM / "clang",
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdcv_zicsr_zifencei",
                "-mabi=lp64d",
                "-mno-relax",
                "-c",
                output / "kernel.nr.s",
                "-o",
                output / "kernel.o",
            ],
            output / "object.log",
        )
        record = {
            "case": name,
            "status": "BUILD_PASS_BOARD_NUMERICS_PENDING",
            "baseline": str(source),
            "output": str(output),
            "baseline_object_sha256": sha(source / "kernel.o"),
            "candidate_object_sha256": sha(output / "kernel.o"),
            "input_llvm_ir_sha256": sha(source / "kernel.ll"),
            "optimized_llvm_ir_sha256": sha(output / "kernel.ll"),
            "adapter_sha256": sha(output / "adapter.c"),
            "opt_sha256": sha(LLVM / "opt"),
            "flags": ["default<O2>", "vectorize-loops=false", "vectorize-slp=false"],
            "production_replaced": False,
        }
        (output / "candidate.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(record["status"], name, record["candidate_object_sha256"], flush=True)
    (args.output / "build-report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

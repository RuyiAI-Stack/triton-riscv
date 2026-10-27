#!/usr/bin/env python3
"""Lower the captured graph plumbing around separately compiled ViT kernels.

Numerical kernels use common.mk's contract -> plan -> verify -> emit
pipeline. This script only lowers the saved Buddy graph, whose numerical
operations have already become external kernel calls. It is not a kernel
compiler, simulator, or numerical acceptance check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} exited {result.returncode}; see {log}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--buddy-opt",
        default=os.environ.get("BUDDY_OPT", "buddy-opt"),
    )
    parser.add_argument(
        "--buddy-translate",
        default=os.environ.get("BUDDY_TRANSLATE", "buddy-translate"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "stage": "Buddy graph plumbing lowering",
        "status": "FAIL",
        "numerical_validation": False,
    }
    tosa_pipeline = (
        "builtin.module("
        + ",".join(
            f"func.func({name})"
            for name in (
                "tosa-to-linalg-named",
                "tosa-to-linalg",
                "tosa-to-tensor",
                "tosa-to-arith",
            )
        )
        + ")"
    )
    passes = [
        "--canonicalize",
        "--cse",
        "--empty-tensor-to-alloc-tensor",
        "--one-shot-bufferize=bufferize-function-boundaries "
        "allow-return-allocs-from-loops=true buffer-alignment=64",
        "--convert-linalg-to-loops",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-index-to-llvm",
        "--memref-expand",
        "--finalize-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    try:
        replacement = json.loads(
            (args.replacement / "triton-call-replacement.json").read_text()
        )
        if replacement.get("status") != "PASS":
            raise ValueError("a complete PASS replacement report is required")
        source = args.replacement / "subgraph0.triton.mlir"
        main_source = args.replacement / "forward.triton.mlir"
        for required in (source, main_source):
            if not required.is_file():
                raise FileNotFoundError(required)
        # Preserve the ordinary lowered ABI of @vit. Only the public wrapper
        # @forward gets a C interface for a separately linked model launcher.
        main_text = main_source.read_text()
        match = re.search(
            r"(func\.func\s+@forward\(.*?\)\s*->\s*memref<1x1000xf32>)\s*\{",
            main_text,
            flags=re.S,
        )
        if match is None:
            raise ValueError("expected a fixed-shape @forward memref wrapper")
        main_input = args.output / "main.ciface.mlir"
        main_input.write_text(
            main_text[: match.start()]
            + match[1]
            + " attributes {llvm.emit_c_interface} {"
            + main_text[match.end() :]
        )
        outputs = {}
        for stem, input_path in (("forward", source), ("main", main_input)):
            tosa = args.output / f"{stem}.tosa-lowered.mlir"
            lowered = args.output / f"{stem}.lowered.mlir"
            llvm = args.output / f"{stem}.ll"
            run(
                [
                    args.buddy_opt,
                    input_path,
                    "--pass-pipeline=" + tosa_pipeline,
                    "-o",
                    tosa,
                ],
                args.output / f"{stem}.tosa.log",
            )
            run(
                [args.buddy_opt, tosa, *passes, "-o", lowered],
                args.output / f"{stem}.lower.log",
            )
            run(
                [
                    args.buddy_translate,
                    "--buddy-to-llvmir",
                    lowered,
                    "-o",
                    llvm,
                ],
                args.output / f"{stem}.translate.log",
            )
            outputs[stem] = {
                "file": llvm.name,
                "sha256": hashlib.sha256(llvm.read_bytes()).hexdigest(),
            }
        if (
            "define void @_mlir_ciface_forward("
            not in (args.output / "main.ll").read_text()
        ):
            raise RuntimeError("main graph did not export _mlir_ciface_forward")
        report.update(
            status="PASS",
            artifacts=outputs,
            external_calls=replacement["graphs"]["forward"]["external_calls"],
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report["reason"] = str(error)
    (args.output / "lower.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""BOSCAME FPGA example 'lower_model.py' under model 'vit-base-patch16-224'.

Inputs: neighboring kernel, launch, and Make files in this example tree.

Outputs: support for AOT compile, simcheck, or board execution.

When used: examples/BOSCAMEFPGA bring-up and numeric validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from model_contract import (
    require_attention_manifest,
    require_no_matrix_fallback,
    require_production_replacement,
)
from model_buffers import verify_bufferized_workspace_dataflow

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
        raise RuntimeError(f"{command[0]} exited {result.returncode}; see {log}")


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
        "full_model_executed": False,
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
    buffer_passes = [
        "--canonicalize",
        "--cse",
        "--empty-tensor-to-alloc-tensor",
        "--one-shot-bufferize=bufferize-function-boundaries "
        "allow-return-allocs-from-loops=true buffer-alignment=64",
    ]
    scalar_passes = [
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
        require_production_replacement(replacement)
        if (
            replacement.get("workspace_buffer_policy_version") != 1
            or replacement.get("graph_mlir_compiled") is not True
        ):
            raise ValueError(
                "a current graph with materialized mutable workspace memrefs is required"
            )
        contract_identity = replacement.get("workspace_buffer_contract", {})
        contract_path = args.replacement / contract_identity.get("file", "")
        if contract_path.name != "workspace-buffers.json" or hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest() != contract_identity.get("sha256"):
            raise ValueError("mutable workspace contract identity is missing or stale")
        buffer_contract = json.loads(contract_path.read_text())
        if (
            buffer_contract.get("status") != "PASS"
            or buffer_contract.get("wrapper_abi", {}).get("status") != "PASS"
        ):
            raise ValueError("mutable workspace or wrapper ABI was not materialized")
        argument_roots = buffer_contract["argument_roots"]
        workspace_names = set(buffer_contract["workspace_roots"])
        if len(argument_roots) != len(set(argument_roots)):
            raise ValueError(
                "caller-owned workspace roots are aliased across function arguments"
            )
        identity = replacement["attention_primitive_manifest"]
        manifest_path = args.replacement / identity["file"]
        if (
            manifest_path.name != "attention-primitive-calls.json"
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != identity["sha256"]
        ):
            raise ValueError("attention primitive manifest identity is stale")
        manifest = json.loads(manifest_path.read_text())
        require_attention_manifest(replacement, manifest)
        if (
            manifest.get("adapter_sha256")
            != hashlib.sha256(
                (args.replacement / "vit_triton_adapters.c").read_bytes()
            ).hexdigest()
        ):
            raise ValueError(
                "generated primitive descriptor adapters differ from manifest"
            )
        source = args.replacement / "subgraph0.triton.mlir"
        main_source = args.replacement / "forward.triton.mlir"
        for path in (source, main_source):
            require_no_matrix_fallback(path.read_text())
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
            bufferized = args.output / f"{stem}.bufferized.mlir"
            lowered = args.output / f"{stem}.lowered.mlir"
            llvm = args.output / f"{stem}.ll"
            run(
                [
                    args.buddy_opt,
                    input_path,
                    "--pass-pipeline=" + tosa_pipeline,
                    "--mlir-print-debuginfo",
                    "-o",
                    tosa,
                ],
                args.output / f"{stem}.tosa.log",
            )
            # The TOSA conversion must not introduce a matrix product that
            # convert-linalg-to-loops could silently scalarize.
            require_no_matrix_fallback(tosa.read_text())
            run(
                [
                    args.buddy_opt,
                    tosa,
                    *buffer_passes,
                    "--mlir-print-debuginfo",
                    "-o",
                    bufferized,
                ],
                args.output / f"{stem}.bufferize.log",
            )
            if stem == "forward":
                audit = verify_bufferized_workspace_dataflow(
                    bufferized.read_text(),
                    "vit",
                    argument_roots,
                    workspace_names,
                    replacement["replaced"],
                )
                audit["bufferized_ir_sha256"] = hashlib.sha256(
                    bufferized.read_bytes()
                ).hexdigest()
                (args.output / "workspace-dataflow.json").write_text(
                    json.dumps(audit, indent=2) + "\n"
                )
                report["workspace_dataflow"] = audit
            run(
                [args.buddy_opt, bufferized, *scalar_passes, "-o", lowered],
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

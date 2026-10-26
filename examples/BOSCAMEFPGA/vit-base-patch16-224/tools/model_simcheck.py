#!/usr/bin/env python3
"""Audit prerequisites for a complete ViT BOSCAME simulation, failing closed.

A standalone kernel PASS, graph replacement PASS, successful host check, or
successful static graph link cannot establish a model simulation PASS. This
entry records why full-model simulation must be skipped while any required
kernel is blocked. It never substitutes NumPy/PyTorch arithmetic for emitted
kernel execution and never labels this prerequisite audit a model PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Exact numerical call shapes in the 12-block graph. Patch projection is a
# composed im2col/quantize/AME/dequantize operator in the upstream example.
SHARED_CASES = {
    "patch_embed_196x768",
    "quantize_196x768",
    "quantize_768x768",
    "matmul_i8_overwrite_196x768x768",
    "dequantize_196x768",
    "layernorm_197x768",
    "attention_qk_12x197x197x64",
    "softmax_12x197x197",
    "attention_pv_12x197x64x197",
    "gelu_197x3072",
    "bias_add_197x768",
    "bias_add_197x3072",
    "bias_add_1x1000",
    "add_197x768",
}
FP32_LINEAR_CASES = {
    "linear_197x768x768",
    "linear_197x3072x768",
    "linear_197x768x3072",
    "linear_1x1000x768",
}
W8A8_LINEAR_CASES = {
    "quantize_197x768",
    "quantize_197x3072",
    "quantize_1x768",
    "matmul_i8_overwrite_197x768x768",
    "matmul_i8_overwrite_197x3072x768",
    "matmul_i8_overwrite_197x768x3072",
    "matmul_i8_overwrite_1x1000x768",
    "dequantize_197x768",
    "dequantize_197x3072",
    "dequantize_1x1000",
}


def read_report(path, reasons):
    try:
        data = json.loads(path.read_text())
        if data.get("schema") != 1:
            raise ValueError("expected schema 1")
        cases = data["cases"]
        names = [record["name"] for record in cases]
        if len(names) != len(set(names)):
            raise ValueError("duplicate case names")
        return {record["name"]: record for record in cases}
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append(
            {"stage": "prerequisite", "reason": f"{path.name}: {error}"}
        )
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fp32", "w8a8"),
        default="w8a8",
        help="linear mode; patch projection remains W8A8",
    )
    parser.add_argument(
        "--build-report", type=Path, default=ROOT / "build/build-report.json"
    )
    parser.add_argument(
        "--simcheck-report",
        type=Path,
        default=ROOT / "build/simcheck-report.json",
    )
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "assets/model.safetensors"
    )
    parser.add_argument(
        "--replacement",
        type=Path,
        help="optional graph replacement report directory",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "build/model-simcheck-report.json"
    )
    args = parser.parse_args()
    reasons = []
    builds = read_report(args.build_report, reasons)
    checks = read_report(args.simcheck_report, reasons)
    required = SHARED_CASES | (
        W8A8_LINEAR_CASES if args.mode == "w8a8" else FP32_LINEAR_CASES
    )
    if args.replacement is not None:
        try:
            graph = json.loads(
                (args.replacement / "triton-call-replacement.json").read_text()
            )
            if graph.get("status") != "PASS":
                raise ValueError("replacement is not complete")
            if bool(graph.get("w8a8_enabled")) != (args.mode == "w8a8"):
                raise ValueError("replacement linear mode differs from --mode")
            required |= set(graph["distinct_cases"])
        except (OSError, ValueError, KeyError) as error:
            reasons.append({"stage": "graph", "reason": str(error)})
    evidence = []
    for name in sorted(required):
        build = builds.get(name, {})
        check = checks.get(name, {})
        record = {
            "name": name,
            "build": build.get("status", "MISSING"),
            "simcheck": check.get("status", "MISSING"),
        }
        evidence.append(record)
        for stage, item in (("build", build), ("simcheck", check)):
            if item.get("status") != "PASS":
                reasons.append(
                    {
                        "case": name,
                        "stage": stage,
                        "reason": item.get("reason", "missing PASS evidence"),
                    }
                )
        if build.get("status") != "PASS" or check.get("status") != "PASS":
            continue
        if check.get("programs") != check.get(
            "total_programs"
        ) or not check.get("programs"):
            reasons.append(
                {
                    "case": name,
                    "stage": "simcheck",
                    "reason": "simcheck does not cover the complete launch grid",
                }
            )
        module = args.kernels / name / "kernel.boscame.mlir"
        try:
            digest = hashlib.sha256(module.read_bytes()).hexdigest()
            if check.get("module_sha256") != digest:
                raise ValueError(
                    "simcheck module digest differs from current emitted kernel"
                )
            record["module_sha256"] = digest
        except (OSError, ValueError) as error:
            reasons.append(
                {"case": name, "stage": "artifact", "reason": str(error)}
            )
    # No compiled-graph simulator exists yet. This must remain explicit even
    # if all standalone kernels pass: those results prove a different scope.
    if not reasons:
        if not args.checkpoint.is_file():
            reasons.append(
                {
                    "stage": "assets",
                    "reason": "pinned model checkpoint is missing",
                }
            )
        reasons.append(
            {
                "stage": "model-execution",
                "reason": "complete graph execution through emitted BOSCAME kernels is not implemented; host graph checks are insufficient",
            }
        )
    report = {
        "schema": 1,
        "status": "SKIP",
        "mode": args.mode,
        "model_executed": False,
        "model_pass": False,
        "required_cases": sorted(required),
        "cases": evidence,
        "reasons": reasons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""BOSCAME FPGA example 'model_simcheck.py' under model 'vit-base-patch16-224'.

Inputs: neighboring kernel, launch, and Make files in this example tree.

Outputs: support for AOT compile, simcheck, or board execution.

When used: examples/BOSCAMEFPGA bring-up and numeric validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from model_contract import (
    ATTENTION_PRIMITIVES,
    require_attention_manifest,
    require_no_matrix_fallback,
    require_production_replacement,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Exact numerical call shapes in the 12-block graph. Patch projection is a
# composed im2col/quantize/AME/dequantize operator in the upstream example.
SHARED_CASES = {
    "patch_embed_196x768",
    "quantize_196x768",
    "quantize_768x768",
    "matmul_i8_196x768x768",
    "dequantize_196x768",
    "layernorm_197x768",
    "softmax_12x197x197",
    "gelu_197x3072",
    "bias_add_197x768",
    "bias_add_197x3072",
    "bias_add_1x1000",
    "add_197x768",
}

ATTENTION_CASES = {name for names in ATTENTION_PRIMITIVES.values() for name in names}
SHARED_CASES |= ATTENTION_CASES

W8A8_LINEAR_CASES = {
    "quantize_197x768",
    "quantize_197x3072",
    "quantize_1x768",
    "matmul_i8_197x768x768",
    "matmul_i8_197x3072x768",
    "matmul_i8_197x768x3072",
    "matmul_i8_1x1000x768",
    "dequantize_197x768",
    "dequantize_197x3072",
    "dequantize_1x1000",
}

PATCH_DEPENDENCIES = (
    "quantize_196x768",
    "quantize_768x768",
    "matmul_i8_196x768x768",
    "dequantize_196x768",
)


def require_simcheck_coverage(case, check):
    """Validate complete rounds, including the patch adapter's child launches."""
    from cases import describe

    rounds = check.get("comparison_rounds", 1)
    runs = check.get("runs", [])
    grid = math.prod(case["grid"])
    patch = case["family"] == "patch_embed"
    child_grids = (
        {name: math.prod(describe(name)["grid"]) for name in PATCH_DEPENDENCIES}
        if patch
        else {}
    )
    per_round = grid + sum(child_grids.values())
    if (
        type(rounds) is not int
        or rounds < 1
        or not isinstance(runs, list)
        or len(runs) != rounds
        or check.get("programs") != per_round * rounds
        or check.get("total_programs") != per_round * rounds
    ):
        raise ValueError(
            "simcheck does not cover every comparison round and complete launch grid"
        )
    for run in runs:
        if not isinstance(run, dict) or run.get("programs") != per_round:
            raise ValueError("simcheck round does not cover the complete launch grid")
        if patch:
            stages = run.get("stages", [])
            if run.get("im2col_programs") != grid or [
                stage.get("case") for stage in stages
            ] != list(PATCH_DEPENDENCIES):
                raise ValueError("patch round lacks its complete im2col/child grids")
            if any(
                stage.get("programs") != child_grids[stage["case"]]
                or stage.get("total_programs") != child_grids[stage["case"]]
                for stage in stages
            ):
                raise ValueError("patch child does not cover its complete launch grid")
        elif run.get("total_programs") != grid:
            raise ValueError("simcheck round does not cover the complete launch grid")


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
        reasons.append({"stage": "prerequisite", "reason": f"{path.name}: {error}"})
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("w8a8",),
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
    required = SHARED_CASES | W8A8_LINEAR_CASES
    from cases import describe

    def close_dependencies():
        pending = list(required)
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            try:
                dependencies = describe(name).get("dependencies", ())
            except ValueError as error:
                reasons.append(
                    {"case": name, "stage": "inventory", "reason": str(error)}
                )
                continue
            required.update(dependencies)
            pending.extend(dependencies)

    if args.replacement is not None:
        try:
            graph = json.loads(
                (args.replacement / "triton-call-replacement.json").read_text()
            )
            require_production_replacement(graph)
            if (
                graph.get("workspace_buffer_policy_version") != 1
                or graph.get("graph_mlir_compiled") is not True
            ):
                raise ValueError(
                    "replacement contains source routing only or predates mutable-workspace graph plumbing"
                )
            contract_identity = graph["workspace_buffer_contract"]
            contract_path = args.replacement / contract_identity["file"]
            if (
                contract_path.name != "workspace-buffers.json"
                or hashlib.sha256(contract_path.read_bytes()).hexdigest()
                != contract_identity["sha256"]
            ):
                raise ValueError("mutable workspace ABI contract is missing or stale")
            buffer_contract = json.loads(contract_path.read_text())
            if (
                buffer_contract.get("status") != "PASS"
                or buffer_contract.get("wrapper_abi", {}).get("status") != "PASS"
            ):
                raise ValueError(
                    "mutable workspace and wrapper ABI have no passing materialization proof"
                )
            identity = graph["attention_primitive_manifest"]
            path = args.replacement / identity["file"]
            if (
                path.name != "attention-primitive-calls.json"
                or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]
            ):
                raise ValueError("attention primitive manifest identity is stale")
            manifest = json.loads(path.read_text())
            require_attention_manifest(graph, manifest)
            if (
                manifest.get("adapter_sha256")
                != hashlib.sha256(
                    (args.replacement / "vit_triton_adapters.c").read_bytes()
                ).hexdigest()
            ):
                raise ValueError(
                    "attention descriptor adapter source differs from manifest"
                )
            for filename in ("subgraph0.triton.mlir", "forward.triton.mlir"):
                require_no_matrix_fallback((args.replacement / filename).read_text())
            required |= set(graph["distinct_cases"])
        except (OSError, ValueError, KeyError) as error:
            reasons.append({"stage": "graph", "reason": str(error)})
    close_dependencies()
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
        case = describe(name)
        try:
            require_simcheck_coverage(case, check)
        except (ValueError, TypeError, KeyError) as error:
            reasons.append(
                {
                    "case": name,
                    "stage": "simcheck",
                    "reason": str(error),
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
            source_digest = hashlib.sha256(
                Path(case["kernel_source"]).read_bytes()
            ).hexdigest()
            if check.get("kernel_module_sha256") != source_digest:
                raise ValueError("simcheck evidence differs from current kernel source")
            if case["family"] == "patch_embed":
                for dependency in PATCH_DEPENDENCIES:
                    dep_module = args.kernels / dependency / "kernel.boscame.mlir"
                    expected = hashlib.sha256(dep_module.read_bytes()).hexdigest()
                    dep_source = hashlib.sha256(
                        Path(describe(dependency)["kernel_source"]).read_bytes()
                    ).hexdigest()
                    if check.get("dependency_sha256", {}).get(dependency) != expected:
                        raise ValueError(
                            "patch dependency module evidence is stale: " + dependency
                        )
                    for run in check.get("runs", []):
                        stage = next(
                            (
                                item
                                for item in run.get("stages", [])
                                if item.get("case") == dependency
                            ),
                            {},
                        )
                        if (
                            stage.get("module_sha256") != expected
                            or stage.get("kernel_module_sha256") != dep_source
                        ):
                            raise ValueError(
                                "patch child round module/source evidence is stale: "
                                + dependency
                            )
            else:
                for run in check.get("runs", []):
                    if (
                        run.get("module_sha256") != digest
                        or run.get("kernel_module_sha256") != source_digest
                    ):
                        raise ValueError(
                            "simcheck round module/source evidence is stale"
                        )
        except (OSError, ValueError) as error:
            reasons.append({"case": name, "stage": "artifact", "reason": str(error)})
    if not args.checkpoint.is_file():
        reasons.append(
            {"stage": "assets", "reason": "pinned model checkpoint is missing"}
        )
    # No compiled-graph simulator exists yet. This must remain explicit even
    # if all standalone kernels pass: those results prove a different scope.
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

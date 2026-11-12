#!/usr/bin/env python3
"""Result rollup for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.

Inputs: per-case check logs, UART captures, or simcheck JSON produced by the
Make flow.

Outputs: a readable summary of pass/fail and error stats across the model.

When used: after a batch host-check or board run.
"""

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import describe, digest, inventory, summary_report_path  # noqa: E402

MODEL = Path(__file__).resolve().parents[1]

STAGE_FILES = {
    "ttir": ["ttir", "frontend.mlir"],
    "boscame": ["bufferized.mlir", "planned.mlir", "boscame.mlir"],
    "llvm": ["llvm.mlir", "ll"],
    "asm": ["s", "encoded.s", "nr.s", "o"],
}


def write_json(path, value, unchanged=False):
    data = json.dumps(value, indent=2) + "\n"
    if unchanged and path.is_file() and path.read_text() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def success(name, output, stage):
    case = describe(name)
    directory = output / name
    manifest = json.loads((directory / "frontend.json").read_text())
    if manifest["kernel_module_sha256"] != digest(case["kernel_source"]):
        raise ValueError("frontend manifest does not match current kernel.py")
    for key in (
        "name",
        "family",
        "kernel",
        "implementation",
        "implementation_source",
        "implementation_symbol",
        "reference_kernel",
        "signature",
        "constexprs",
        "grid",
        "shape",
        "arguments",
        "accumulation",
        "math_accuracy",
    ):
        if manifest.get(key) != case.get(key):
            raise ValueError("frontend manifest does not match current " + key)
    artifacts = {}
    for checkpoint, extensions in STAGE_FILES.items():
        for extension in extensions:
            filename = "kernel." + extension
            artifacts[filename] = digest(directory / filename)
        if checkpoint == stage:
            break
    inputs_path = directory / ".compiler-inputs.json"
    inputs = (
        dict(
            value.split("=", 1)
            for value in json.loads(inputs_path.read_text())
            if "=" in value
        )
        if inputs_path.is_file()
        else {}
    )
    optimization = None
    phases = None
    if "conv-patch-ddr-phases" in inputs:
        from add_conv_patch_ddr_phases import transform

        tool = Path(inputs["conv-patch-ddr-phases"]).resolve()
        phases = {
            "pipeline": "per-patch im2col DDR read/write phases",
            "tool": str(tool),
            "tool_sha256": digest(tool),
            "applied": stage in ("llvm", "asm"),
        }
        if phases["applied"]:
            transformed, expected = transform(
                (directory / "kernel.ll").read_text(), case
            )
            actual = json.loads((directory / "kernel.phase.json").read_text())
            if (
                actual != expected
                or (directory / "kernel.phase.ll").read_text() != transformed
            ):
                raise ValueError(
                    "im2col phase artifacts do not match original LLVM math/current transformer"
                )
            artifacts["kernel.phase.ll"] = digest(directory / "kernel.phase.ll")
            artifacts["kernel.phase.json"] = digest(directory / "kernel.phase.json")
            phases["transformation"] = actual
            phases["input_sha256"] = artifacts["kernel.ll"]
            phases["output_sha256"] = artifacts["kernel.phase.ll"]
    if "llvm-scalar-o2-tool" in inputs:
        tool = Path(inputs["llvm-scalar-o2-tool"])
        resolved = Path(shutil.which(str(tool)) or tool).resolve()
        optimization = {
            "pipeline": "scalar LLVM O2",
            "tool": str(resolved),
            "tool_sha256": digest(resolved),
            "flags": shlex.split(inputs["llvm-scalar-o2-flags"]),
            "applied": stage in ("llvm", "asm"),
        }
        if optimization["applied"]:
            artifacts["kernel.optimized.ll"] = digest(directory / "kernel.optimized.ll")
            optimization["input_sha256"] = (
                artifacts["kernel.phase.ll"]
                if phases is not None
                else artifacts["kernel.ll"]
            )
            optimization["output_sha256"] = artifacts["kernel.optimized.ll"]
    if artifacts["kernel.ttir"] != manifest["ttir_sha256"]:
        raise ValueError("TTIR does not match frontend manifest")
    native = None
    if (
        case.get("implementation") in ("hybrid_rvv_c", "hybrid_ordered_c")
        and stage == "asm"
    ):
        native = json.loads((directory / "native-implementation.json").read_text())
        if (
            native.get("implementation") != case["implementation"]
            or native.get("candidate_source_sha256")
            != digest(MODEL / case["implementation_source"])
            or native.get("object_sha256") != digest(directory / "kernel.o")
            or native.get("adapter_sha256") != digest(directory / "adapter.c")
        ):
            raise ValueError("native build does not match declared source/artifacts")
    record = {
        "name": name,
        "status": "PASS",
        "stage": stage,
        "artifact_dir": str(directory),
        "artifacts": artifacts,
        "kernel_module_sha256": digest(case["kernel_source"]),
    }
    if optimization is not None:
        record["llvm_optimization"] = optimization
    if phases is not None:
        record["conv_patch_ddr_phases"] = phases
    if native is not None:
        record["native_implementation"] = native
        record["kernel_ir_role"] = (
            "Triton/BOSCAME/LLVM artifacts are explicit numerical reference; kernel.o uses declared native C source"
        )
    return record


def failure(name, output, stage, reason):
    return {
        "name": name,
        "status": "BLOCKED",
        "stage": stage,
        "artifact_dir": str(output / name),
        "reason": reason,
    }


def collect(output, names, stage):
    results = []
    for name in names:
        try:
            record = json.loads((output / name / "build-result.json").read_text())
            if record["status"] == "PASS":
                # Recheck current sources and stage hashes before accepting a
                # cached report, including artifacts from a later stage.
                checked = success(name, output, stage)
                for filename, sha in checked["artifacts"].items():
                    if record.get("artifacts", {}).get(filename) != sha:
                        raise ValueError("unrecorded or stale artifact: " + filename)
                if stage in ("llvm", "asm") and record.get(
                    "llvm_optimization"
                ) != checked.get("llvm_optimization"):
                    raise ValueError("unrecorded or stale scalar LLVM optimization")
                if stage in ("llvm", "asm") and record.get(
                    "conv_patch_ddr_phases"
                ) != checked.get("conv_patch_ddr_phases"):
                    raise ValueError("unrecorded or stale im2col DDR phases")
                record = checked
        except (OSError, ValueError, KeyError) as error:
            record = failure(name, output, stage, str(error))
        results.append(record)
    return {
        "schema": 1,
        "status": "PASS" if all(r["status"] == "PASS" for r in results) else "PARTIAL",
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inputs", "success", "failure", "collect"))
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--all", action="store_true", help="write the complete-suite report"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage",
        choices=(
            *STAGE_FILES,
            "bufferized",
            "planned",
            "llvm-ir",
            "llvm-phase",
            "llvm-opt",
            "native-rvv",
            "native-ordered",
            "encode",
            "restrict",
            "object",
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--value", action="append", default=[])
    parser.add_argument("--log", type=Path)
    parser.add_argument("--reason", default="make failed; see the stage log")
    args = parser.parse_args()
    if args.action == "inputs":
        if args.inputs is None:
            parser.error("inputs requires --inputs")
        write_json(args.inputs, args.value, unchanged=True)
        return 0
    if args.output is None or args.stage is None:
        parser.error("result actions require --output and --stage")
    args.output = args.output.resolve()
    if args.action == "collect":
        names = args.cases or [case["name"] for case in inventory()]
        report = collect(args.output, names, args.stage)
        destination = args.report or summary_report_path(
            args.output, None if args.all else args.cases, "build"
        )
    else:
        if not args.cases or len(args.cases) != 1:
            parser.error("per-case results require exactly one --case")
        name = args.cases[0]
        if args.action == "success":
            try:
                record = success(name, args.output, args.stage)
            except (OSError, ValueError, KeyError) as error:
                record = failure(name, args.output, args.stage, str(error))
        else:
            reason = args.reason
            if args.log and args.log.is_file():
                reason = args.log.read_text().strip() or reason
            record = failure(name, args.output, args.stage, reason)
        write_json(args.output / name / "build-result.json", record)
        report = {
            "schema": 1,
            "status": "PASS" if record["status"] == "PASS" else "PARTIAL",
            "cases": [record],
        }
        destination = args.report or summary_report_path(args.output, [name], "build")
        print(f"{record['status']}: {name} ({record['stage']})", flush=True)
    write_json(destination, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

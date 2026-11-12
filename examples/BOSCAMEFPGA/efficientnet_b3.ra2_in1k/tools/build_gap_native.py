#!/usr/bin/env python3
"""Record ordered C GAP compilation separately from optional board evidence.

A normal clean Make build needs only source and compiler inputs. Attaching board
proof is explicit and fail-closed; publication must separately require that
proof for the exact object and fixed specialization.
"""

import argparse
import json
from pathlib import Path
import sys

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[2]
sys.path.insert(0, str(MODEL))
from cases import describe  # noqa: E402
from export import digest  # noqa: E402


def board_evidence(case, source, output, path):
    data = json.loads(path.read_text())
    if "whole_cpu_model_proof" in data:
        if (
            data.get("status") != "PASS"
            or len(data.get("results", [])) != 13
            or data.get("source_sha256") != digest(source)
        ):
            raise ValueError("all13 pinned Torch-order GAP board evidence incomplete")
        entries = [r for r in data["results"] if r.get("case") == case["name"]]
        if len(entries) != 1:
            raise ValueError("expected one exact Torch-order GAP case")
        entry = entries[0]
        if (
            entry.get("state") != "completed"
            or entry.get("execution_status") != "exited"
            or entry.get("app_exit_code") != 0
            or entry.get("exit_code") != 0
            or entry.get("kernel_sha256") != digest(output / "kernel.o")
            or entry.get("bit_exact") is not True
            or entry.get("max_abs_error") != 0
        ):
            raise ValueError("Torch-order GAP source/object lacks exact board evidence")
        folder = path.parent / case["name"] / entry["task_id"]
        summary = json.loads((folder / "summary.json").read_text())
        image = json.loads((folder / "image-manifest.json").read_text())
        existing = image["existing_case_inputs"]
        if (
            len(existing) != 1
            or existing[0]["case"] != case["name"]
            or existing[0]["object_sha256"] != digest(output / "kernel.o")
            or summary.get("image_sha256") != image["image_sha256"]
            or summary.get("task_id") != entry["task_id"]
            or summary.get("state") != "completed"
            or summary.get("execution_status") != "exited"
            or summary.get("app_exit_code") != 0
            or summary.get("exit_code") != 0
            or not summary.get("comparisons")
            or not summary["comparisons"][0].get("bit_exact")
        ):
            raise ValueError("Torch-order GAP board summary/image mismatch")
        adapter = Path(existing[0]["adapter_source"])
        spec = case["constexprs"]
        call = f"{case['implementation_symbol']}((const float *)p0,(float *)p1,{spec['SPATIAL']}u,{spec['C']}u,{spec['BLOCK']}u);"
        if (
            digest(adapter) != existing[0]["adapter_source_sha256"]
            or digest(adapter) != digest(output / "adapter.c")
            or call.replace(" ", "")
            not in adapter.read_text().replace(" ", "").replace("\n", "")
        ):
            raise ValueError("Torch-order GAP adapter specialization mismatch")
        return {
            "status": "PASS",
            "task_id": entry["task_id"],
            "bit_exact": True,
            "reference_semantics": data["reference_semantics"],
            "evidence": str(path.resolve()),
            "evidence_sha256": digest(path),
        }
    if data.get("status") != "PASS" or data.get("completed") != 13:
        raise ValueError("all13 GAP board evidence incomplete")
    entries = [r for r in data["results"] if r.get("case") == case["name"]]
    if len(entries) != 1:
        raise ValueError("expected one exact GAP case in board evidence")
    entry = entries[0]
    if (
        entry.get("state") != "completed"
        or entry.get("execution_status") != "exited"
        or entry.get("app_exit_code") != 0
        or entry.get("exit_code") != 0
        or entry.get("source_sha256") != digest(source)
        or entry.get("kernel_o_sha256") != digest(output / "kernel.o")
        or not entry.get("comparisons")
        or not entry["comparisons"][0].get("bit_exact")
    ):
        raise ValueError("GAP object/source lacks successful exact board evidence")
    # The generic C object takes SPATIAL/C/BLOCK as adapter arguments: object
    # identity alone cannot establish the specialization actually tested.
    image_manifest = json.loads(
        (
            path.parent / case["name"] / entry["task_id"] / "image-manifest.json"
        ).read_text()
    )
    image_input = image_manifest["existing_case_inputs"]
    if len(image_input) != 1 or image_input[0]["case"] != case["name"]:
        raise ValueError("board probe has wrong case binding")
    spec = case["constexprs"]
    # Earlier probe manifests retain the exact native adapter source. Validate
    # fixed native call arguments independently of formatting/header comments.
    adapter_path = Path(image_input[0]["adapter_source"])
    if (
        digest(adapter_path) != image_input[0]["adapter_source_sha256"]
        or image_input[0]["object_sha256"] != digest(output / "kernel.o")
        or image_manifest["image_sha256"] != entry["image_sha256"]
    ):
        raise ValueError("board probe artifacts differ from their recorded hashes")
    adapter = adapter_path.read_text()
    expected_call = f"{case['implementation_symbol']}((const float *)p0,(float *)p1,{spec['SPATIAL']}u,{spec['C']}u,{spec['BLOCK']}u);"
    if expected_call.replace(" ", "") not in adapter.replace(" ", "").replace("\n", ""):
        raise ValueError(
            "board probe adapter specialization differs from current GAP configuration"
        )
    return {
        "status": "PASS",
        "task_id": entry["task_id"],
        "bit_exact": True,
        "evidence": str(path.resolve()),
        "evidence_sha256": digest(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument(
        "--numerical-evidence",
        type=Path,
        help="optional exact all13 board report; absent means NOT_RUN",
    )
    args = parser.parse_args()
    case = describe(args.case)
    output = args.output
    source = MODEL / case["implementation_source"]
    if case.get("implementation") != "hybrid_ordered_c":
        raise ValueError("case is not explicitly configured for ordered C GAP")
    numerical = {
        "status": "NOT_RUN",
        "reason": "compilation does not run board numerical validation",
    }
    if args.numerical_evidence is not None:
        numerical = board_evidence(case, source, output, args.numerical_evidence)
    manifest = {
        "case": case["name"],
        "implementation": "hybrid_ordered_c",
        "reference_implementation": "Triton -> BOSCAME fixed-BLOCK reduction retained as historical reference; kernel.o explicitly adopts pinned Torch CPU mean order",
        "candidate_source": source.relative_to(REPO).as_posix(),
        "candidate_source_sha256": digest(source),
        "kernel_config_sha256": digest(case["kernel_source"]),
        "object_sha256": digest(output / "kernel.o"),
        "adapter_sha256": digest(output / "adapter.c"),
        "kernel_s_sha256": digest(output / "kernel.s"),
        "encoded_s_sha256": digest(output / "kernel.encoded.s"),
        "restricted_s_sha256": digest(output / "kernel.nr.s"),
        "reference_module_sha256": digest(output / "kernel.boscame.mlir"),
        "instruction_restriction": "PASS",
        "compiler": str(args.clang),
        "compiler_sha256": digest(args.clang),
        "compile_mode": "default Make asm path with explicit ordered C selection; no fastmath or vectorizer",
        "policy": "one uncached read phase and one write-through output phase per channel",
        "ordered_sum": "pinned Torch 2.10 CPU AVX2 eight-lane, four-ILP, four-level cascade; scalar F32 arithmetic and true F32 division; BLOCK argument retained for ABI compatibility",
        "build_status": "PASS",
        "numerical_validation": numerical,
        "fpga112_bit_exact": numerical["status"] == "PASS",
    }
    (output / "native-implementation.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        "PASS", args.case, manifest["object_sha256"], "numerical=" + numerical["status"]
    )


if __name__ == "__main__":
    main()

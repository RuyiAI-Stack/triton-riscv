#!/usr/bin/env python3
"""Build scalar and RVV full-shape dequantization connector probes."""

import argparse
import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
from check_dequant_candidate import BUDDY, HERE, LLVM, MODEL, run, sha

BASE = MODEL / "build/model-integration"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "dequant-rvv-probe-compact"
    )
    parser.add_argument("--candidate", type=Path, default=BASE / "dequant-rvv")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = BASE / "physical-plan"
    plan = copy.deepcopy(json.loads((original / "physical-plan.json").read_text()))
    specs = {spec["name"]: spec for spec in plan["connectors"]}
    call = next(
        copy.deepcopy(call)
        for call in plan["calls"]
        if call["target"] == "dequantize_unpack_bias__72_192_72"
    )
    spec = specs[call["target"]]
    channels, spatial = spec["constexprs"]["OC"], spec["grid"][0]
    call["index"] = 0
    old_input = call["arguments"][0]["value"]
    input_spec = copy.deepcopy(plan["tensors"][old_input])
    input_spec["storage"] = {"kind": "external_input"}
    plan["tensors"]["input"] = input_spec
    call["arguments"][0]["value"] = "input"
    call["arguments"][1].update(source="parameter", value="probe.activation_scale")
    call["reads"] = ["input"]
    output = call["writes"][0]
    output_bytes = plan["tensors"][output]["bytes"]
    plan["tensors"][output]["storage"]["offset"] = 0
    plan["workspace"].update(
        provisional_total_bytes=(output_bytes + 63) & ~63,
        composite_scratch_bytes=0,
        scratch_offset=(output_bytes + 63) & ~63,
        activation_high_water_bytes=output_bytes,
    )
    plan.update(
        calls=[call],
        physical_calls=1,
        logical_nodes=1,
        connectors=[spec],
        unique_connector_specializations=1,
        unique_existing_cases=0,
        output={
            "value": output,
            "shape": plan["tensors"][output]["shape"],
            "dtype": "f32",
        },
        target_counts={"connector": 1},
        mode="bounded_connector_probe",
        full_model=False,
    )
    (args.output / "physical-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    original_metadata = json.loads((original / "parameters.json").read_text())
    original_blob = (original / "parameters.bin").read_bytes()
    wanted = {argument["value"] for argument in call["arguments"][2:4]}
    metadata = {
        key: value for key, value in original_metadata.items() if key != "entries"
    }
    metadata["entries"] = []
    blob = bytearray()
    for original_entry in original_metadata["entries"]:
        if original_entry["name"] not in wanted:
            continue
        entry = copy.deepcopy(original_entry)
        destination_offset = (len(blob) + 63) & ~63
        blob.extend(b"\0" * (destination_offset - len(blob)))
        blob.extend(original_blob[entry["offset"] : entry["offset"] + entry["bytes"]])
        entry["offset"] = destination_offset
        metadata["entries"].append(entry)
    activation_scale = ((np.arange(spatial) % 31 + 1) / 256).astype("<f4")
    activation_scale[:3] = [0.0, -0.0, 2**-126]
    offset = (len(blob) + 63) & ~63
    blob.extend(b"\0" * (offset - len(blob)))
    blob.extend(activation_scale.tobytes())
    parameter_path = args.output / "parameters.bin"
    parameter_path.write_bytes(blob)
    metadata["entries"].append(
        {
            "name": "probe.activation_scale",
            "shape": [spatial],
            "dtype": "f32",
            "elements": spatial,
            "bytes": activation_scale.nbytes,
            "offset": offset,
            "source": "declared deterministic diagnostic fixture",
            "sha256": sha_bytes(activation_scale.tobytes()),
        }
    )
    metadata.update(bytes=len(blob), sha256=sha(parameter_path))
    (args.output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    entries = {entry["name"]: entry for entry in metadata["entries"]}

    def values(argument):
        entry = entries[argument["value"]]
        return np.frombuffer(
            blob, dtype="<f4", count=entry["elements"], offset=entry["offset"]
        )

    weight_scale, bias = values(call["arguments"][2]), values(call["arguments"][3])
    accumulator = (
        ((np.arange(spatial * channels, dtype=np.int64) * 17011) % 200001 - 100000)
        .astype("<i4")
        .reshape(spatial, channels)
    )
    accumulator.flat[:6] = [-2147483648, 2147483647, -16777217, 16777217, 0, 1]
    expected = accumulator.astype(np.float32) * weight_scale[None, :]
    expected = expected * activation_scale[:, None]
    expected = (expected + bias[None, :]).T.copy()
    accumulator.tofile(args.output / "input.f32")
    expected.tofile(args.output / "w8a8-logits.f32")
    run(
        [
            sys.executable,
            HERE / "generate_runner.py",
            "--plan-dir",
            args.output,
            "--output",
            args.output / "runner",
        ],
        args.output / "generate.log",
    )
    baseline = args.output / "scalar-image"
    run(
        [
            sys.executable,
            HERE / "build_model_image.py",
            "--bounded-probe",
            "--strict-probe",
            "--plan-dir",
            args.output,
            "--runner",
            args.output / "runner",
            "--output",
            baseline,
            "--buddy-opt",
            BASE / "pinned-tools/buddy-opt",
        ],
        args.output / "build-scalar.log",
    )
    candidate = args.output / "rvv-image"
    candidate.mkdir(exist_ok=True)
    source = (BASE / "connectors-ddr-policy" / spec["name"] / "adapter.c").read_text()
    marker = "  for (int32_t x"
    if source.count(marker) != 1:
        raise ValueError("unexpected scalar adapter shape")
    wrapper = (
        "#include <stdint.h>\nextern void efficientnet_dequant_rvv_candidate(const int32_t *,const float *,"
        "const float *,const float *,float *,__SIZE_TYPE__,__SIZE_TYPE__);\n"
        + source.split(marker)[0]
        + f"  efficientnet_dequant_rvv_candidate((const int32_t *)p0,(const float *)p1,"
        f"(const float *)p2,(const float *)p3,(float *)p4,{channels}u,{spatial}u);\n}}\n"
    )
    (candidate / "adapter.c").write_text(wrapper)
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-ffp-contract=off",
    ]
    run(
        [
            LLVM / "clang",
            *flags,
            "-I",
            MODEL,
            "-c",
            candidate / "adapter.c",
            "-o",
            candidate / "adapter.o",
        ],
        candidate / "adapter.log",
    )
    command = ast.literal_eval((baseline / "link.log").read_text().splitlines()[0][2:])
    command = [
        str(candidate / "efficientnet_b3_model.elf")
        if value == str(baseline / "efficientnet_b3_model.elf")
        else str(args.candidate / "kernel.o")
        if value == str(BASE / "connectors-ddr-policy" / spec["name"] / "kernel.o")
        else str(candidate / "adapter.o")
        if value == str(baseline / "obj" / ("connector_" + spec["name"] + ".o"))
        else "-Map=" + str(candidate / "efficientnet_b3_model.map")
        if value.startswith("-Map=")
        else value
        for value in command
    ]
    run(command, candidate / "link.log")
    elf = candidate / "efficientnet_b3_model.elf"
    binary = candidate / "efficientnet_b3_model.bin"
    tools = BUDDY / "examples/FPGA-BOSCAME/tools"
    run(
        [
            sys.executable,
            tools / "check_nr_elf.py",
            elf,
            "--objdump",
            LLVM / "llvm-objdump",
            "--output",
            candidate / "elf-audit.json",
        ],
        candidate / "audit.log",
    )
    run([LLVM / "llvm-objcopy", "-O", "binary", elf, binary], candidate / "objcopy.log")
    manifest = json.loads((baseline / "image-manifest.json").read_text())
    manifest.update(
        scope="independent full-shape RVV dequantization probe; not whole-model evidence",
        image_sha256=sha(binary),
        image_bytes=binary.stat().st_size,
        elf_sha256=sha(elf),
        independent_candidate=True,
        candidate_source_sha256=sha(HERE / "dequant_rvv_candidate.c"),
    )
    manifest["connector_inputs"] = [
        {
            "name": spec["name"],
            "object": str(args.candidate / "kernel.o"),
            "object_sha256": sha(args.candidate / "kernel.o"),
            "adapter_source": str(candidate / "adapter.c"),
            "adapter_sha256": sha(candidate / "adapter.c"),
        }
    ]
    symbols = run([LLVM / "llvm-nm", "-n", elf], candidate / "symbols.log")
    wanted = set(manifest["elf_layout_symbols"])
    manifest["elf_layout_symbols"] = {
        fields[2]: "0x" + fields[0]
        for line in symbols.splitlines()
        if len(fields := line.split()) == 3 and fields[2] in wanted
    }
    (candidate / "image-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = {
        "status": "BUILT_BOARD_PENDING",
        "full_model": False,
        "channels": channels,
        "spatial": spatial,
        "elements": accumulator.size,
        "fixture": "deterministic nonzero integer accumulators, declared scales, genuine folded model weights/bias",
        "scalar_image_sha256": sha(baseline / "efficientnet_b3_model.bin"),
        "candidate_image_sha256": sha(binary),
        "baseline_policy_accesses": 5 * accumulator.size,
        "candidate_policy_transitions": 2 * channels * ((spatial + 15) // 16),
        "expected_sha256": sha(args.output / "w8a8-logits.f32"),
    }
    (args.output / "probe-build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def sha_bytes(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()

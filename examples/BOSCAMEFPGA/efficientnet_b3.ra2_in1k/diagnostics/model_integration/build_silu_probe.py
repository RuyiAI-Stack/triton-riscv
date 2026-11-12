#!/usr/bin/env python3
"""Build independent full-shape scalar/RVV SiLU probes; never submit jobs.

Input uses the actual folded/W8A8 stem output plus finite boundary values.
The 37 production pointwise and all production SiLU artifacts stay untouched.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BASE = ROOT / "build/model-integration"
BUDDY = REPO / "buddy-mlir"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(command, log, stdin=None):
    p = subprocess.run(
        list(map(str, command)), input=stdin, capture_output=True, text=True
    )
    log.write_text(repr(list(map(str, command))) + chr(10) + p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError("command failed: " + str(log))
    return p.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=BASE / "silu-rvv-probe")
    p.add_argument("--buddy-opt", type=Path, default=BASE / "pinned-tools/buddy-opt")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plan = copy.deepcopy(
        json.loads((BASE / "physical-plan/physical-plan.json").read_text())
    )
    call = copy.deepcopy(plan["calls"][1])
    call["index"] = 0
    old_input = call["arguments"][0]["value"]
    old_tensor = copy.deepcopy(plan["tensors"][old_input])
    old_tensor["storage"] = {"kind": "external_input"}
    plan["tensors"]["input"] = old_tensor
    call["arguments"][0]["value"] = "input"
    call["reads"] = ["input"]
    value = call["writes"][0]
    plan.update(
        calls=[call],
        physical_calls=1,
        logical_nodes=1,
        connectors=[],
        unique_connector_specializations=0,
        unique_existing_cases=1,
        output={
            "value": value,
            "shape": plan["tensors"][value]["shape"],
            "dtype": "f32",
        },
        target_counts={"existing_case": 1},
        mode="bounded_silu_probe",
        full_model=False,
    )
    (a.output / "physical-plan.json").write_text(json.dumps(plan, indent=2) + chr(10))
    for filename in ("parameters.bin", "parameters.json"):
        shutil.copyfile(BASE / "physical-plan" / filename, a.output / filename)
    x = np.fromfile(BASE / "prefix8/stem.f32", dtype="<f4").copy()
    boundaries = np.array(
        [
            -np.finfo(np.float32).max,
            -110,
            -104,
            -103.97208404541015625,
            -89,
            -88.72283935546875,
            -88,
            -20,
            -1,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1,
            20,
            88,
            88.72283935546875,
            89,
            104,
            110,
            np.finfo(np.float32).max,
            np.nextafter(np.float32(0), np.float32(1)),
            -np.nextafter(np.float32(0), np.float32(1)),
        ],
        np.float32,
    )
    x[: len(boundaries)] = boundaries
    x[-len(boundaries) :] = boundaries[::-1]
    # Match the declared F32 SiLU expression, including exp overflow to +inf.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = x / (np.float32(1) + np.exp(-x))
    x.tofile(a.output / "input.f32")
    expected.tofile(a.output / "w8a8-logits.f32")
    run(
        [
            sys.executable,
            HERE / "generate_runner.py",
            "--plan-dir",
            a.output,
            "--output",
            a.output / "runner",
        ],
        a.output / "generate.log",
    )
    baseline = a.output / "scalar-image"
    candidate = a.output / "rvv-image"
    candidate.mkdir(exist_ok=True)
    run(
        [
            sys.executable,
            HERE / "build_model_image.py",
            "--plan-dir",
            a.output,
            "--runner",
            a.output / "runner",
            "--output",
            baseline,
            "--bounded-probe",
            "--strict-probe",
            "--buddy-opt",
            a.buddy_opt,
        ],
        a.output / "build-scalar.log",
    )
    clang = BUDDY / "llvm/build/bin/clang"
    tools = BUDDY / "examples/FPGA-BOSCAME/tools"
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
    ]
    source = HERE / "silu_rvv_candidate.c"
    run(
        [clang, *flags, "-S", source, "-o", candidate / "kernel.s"],
        candidate / "compile.log",
    )
    restricted = run(
        [
            sys.executable,
            tools / "restrict_fpga_assembly.py",
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
            "--reject-vector-ordered-reductions",
            "--reject-vector-max-reductions",
        ],
        candidate / "restrict.log",
        (candidate / "kernel.s").read_text(),
    )
    (candidate / "kernel.nr.s").write_text(restricted)
    run(
        [clang, *flags, "-c", candidate / "kernel.nr.s", "-o", candidate / "kernel.o"],
        candidate / "object.log",
    )
    # Reuse every existing shape/offset/alignment/nonalias check, then replace
    # only the scalar grid loop with the independent full-array candidate.
    name = call["target"]
    adapter = (ROOT / "build/kernels" / name / "adapter.c").read_text()
    marker = "  for (int32_t x = 0;"
    if adapter.count(marker) != 1:
        raise ValueError("unexpected production adapter")
    wrapper = adapter.split(marker)[0]
    wrapper += (
        "  efficientnet_silu_rvv_candidate((const float *)p0,(float *)p1,%du);\n}\n"
        % x.size
    )
    wrapper = (
        "extern void efficientnet_silu_rvv_candidate(const float *,float *,__SIZE_TYPE__);\n"
        + wrapper
    )
    (candidate / "adapter.c").write_text(wrapper)
    run(
        [
            clang,
            *flags,
            "-I",
            ROOT,
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
        if v == str(baseline / "efficientnet_b3_model.elf")
        else str(candidate / "kernel.o")
        if v == str(baseline / "cpu-kernels" / name / "kernel.o")
        else str(candidate / "adapter.o")
        if v == str(baseline / "obj" / ("adapter_" + name + ".o"))
        else "-Map=" + str(candidate / "efficientnet_b3_model.map")
        if v.startswith("-Map=")
        else v
        for v in command
    ]
    run(command, candidate / "link.log")
    elf = candidate / "efficientnet_b3_model.elf"
    binary = candidate / "efficientnet_b3_model.bin"
    run(
        [
            sys.executable,
            tools / "check_nr_elf.py",
            elf,
            "--objdump",
            BUDDY / "llvm/build/bin/llvm-objdump",
            "--output",
            candidate / "elf-audit.json",
        ],
        candidate / "audit.log",
    )
    run(
        [BUDDY / "llvm/build/bin/llvm-objcopy", "-O", "binary", elf, binary],
        candidate / "objcopy.log",
    )
    manifest = json.loads((baseline / "image-manifest.json").read_text())
    manifest.update(
        scope="independent full-shape hybrid RVV SiLU probe; not whole-model evidence",
        candidate_source_sha256=sha(source),
        image_sha256=sha(binary),
        image_bytes=binary.stat().st_size,
        elf_sha256=sha(elf),
        baseline_image_sha256=sha(baseline / "efficientnet_b3_model.bin"),
        independent_candidate=True,
    )
    manifest["existing_case_inputs"] = [
        {
            "case": name,
            "family": "silu",
            "overlay": True,
            "object": str(candidate / "kernel.o"),
            "object_sha256": sha(candidate / "kernel.o"),
            "adapter_source": str(candidate / "adapter.c"),
            "adapter_source_sha256": sha(candidate / "adapter.c"),
        }
    ]
    manifest["model_local_cpu_policy_kernels"] = []
    manifest["model_local_hybrid_rvv_kernels"] = [
        {
            "name": name,
            "kernel_sha256": sha(candidate / "kernel.o"),
            "source_sha256": sha(source),
            "policy": "uncached input loads / write-through output stores per <=16-element block",
        }
    ]
    symbols = run(
        [BUDDY / "llvm/build/bin/llvm-nm", "-n", elf], candidate / "symbols.log"
    )
    wanted = set(manifest["elf_layout_symbols"])
    manifest["elf_layout_symbols"] = {}
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] in wanted:
            manifest["elf_layout_symbols"][fields[2]] = "0x" + fields[0]
    (candidate / "image-manifest.json").write_text(
        json.dumps(manifest, indent=2) + chr(10)
    )
    report = {
        "status": "PASS",
        "scope": "standalone SiLU build/ISA audit; no FPGA evidence",
        "elements": x.size,
        "input_source": "actual W8A8 stem CPU output with finite boundary insertions",
        "finite_boundary_values": boundaries.tolist(),
        "input_sha256": sha(a.output / "input.f32"),
        "expected_sha256": sha(a.output / "w8a8-logits.f32"),
        "scalar_image_sha256": sha(baseline / "efficientnet_b3_model.bin"),
        "rvv_image_sha256": sha(binary),
        "candidate_source_sha256": sha(source),
        "board_atol": 2e-6,
        "board_rtol": 2e-6,
        "full_model": False,
    }
    (a.output / "probe-build.json").write_text(json.dumps(report, indent=2) + chr(10))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

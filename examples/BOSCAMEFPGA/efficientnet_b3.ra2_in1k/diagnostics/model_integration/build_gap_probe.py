#!/usr/bin/env python3
"""Build independent full-shape per-access / phase-batched GAP probes; never submit jobs.

Input uses a signed finite cancellation fixture validated against generated LLVM IR.
The 37 production pointwise and all production GAP artifacts stay untouched.
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
    p.add_argument("--output", type=Path, default=BASE / "gap-phase-probe")
    p.add_argument("--buddy-opt", type=Path, default=BASE / "pinned-tools/buddy-opt")
    p.add_argument("--case", default="gap_144x144x40")
    p.add_argument("--overlay", type=Path, default=BASE / "gap-phase-overlay")
    p.add_argument("--source", type=Path, default=HERE / "gap_phase_candidate.c")
    p.add_argument(
        "--fixtures", type=Path, default=BASE / "gap-phase-candidate/fixtures"
    )
    p.add_argument(
        "--reference-semantics",
        default="signed cancellation fixture; host candidate exact vs emitted LLVM IR",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from cases import describe

    case = describe(a.case)
    c = case["constexprs"]
    name = a.case
    if case["family"] != "gap":
        raise ValueError("expected GAP case")
    plan = copy.deepcopy(
        json.loads((BASE / "physical-plan/physical-plan.json").read_text())
    )
    call = copy.deepcopy(
        next(c for c in plan["calls"] if c["target"] == "gap_144x144x40")
    )
    call["index"] = 0
    call.update(
        target=a.case,
        symbol="triton_" + a.case,
        logical_node="probe." + a.case,
        grid=case["grid"],
    )
    call["arguments"][0]["shape"] = case["shape"]
    call["arguments"][1]["shape"] = [1, c["C"]]
    old_input = call["arguments"][0]["value"]
    old_tensor = copy.deepcopy(plan["tensors"][old_input])
    old_tensor.update(
        shape=case["shape"],
        elements=c["C"] * c["SPATIAL"],
        bytes=c["C"] * c["SPATIAL"] * 4,
    )
    old_tensor["storage"] = {"kind": "external_input"}
    plan["tensors"]["input"] = old_tensor
    call["arguments"][0]["value"] = "input"
    call["reads"] = ["input"]
    value = call["writes"][0]
    plan["tensors"][value].update(
        shape=[1, c["C"], 1, 1], elements=c["C"], bytes=c["C"] * 4
    )
    plan["tensors"][value]["storage"]["reserved_bytes"] = ((c["C"] * 4 + 63) // 64) * 64
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
        mode="bounded_connector_probe",
        full_model=False,
    )
    (a.output / "physical-plan.json").write_text(json.dumps(plan, indent=2) + chr(10))
    for filename in ("parameters.bin", "parameters.json"):
        shutil.copyfile(BASE / "physical-plan" / filename, a.output / filename)
    x = np.fromfile(a.fixtures / a.case / "input.f32", dtype="<f4").copy()
    expected = np.fromfile(a.fixtures / a.case / "w8a8-logits.f32", dtype="<f4").copy()
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
    candidate = a.output / "phase-image"
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
            "--gap-reference",
        ],
        a.output / "build-scalar.log",
    )
    tools = BUDDY / "examples/FPGA-BOSCAME/tools"
    source = a.source
    for filename in (
        "kernel.s",
        "kernel.nr.s",
        "kernel.o",
        "adapter.c",
        "adapter.o",
        "build-result.json",
    ):
        shutil.copy2(a.overlay / a.case / filename, candidate / filename)
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
        scope="independent full-shape scalar phase-batched GAP probe; not whole-model evidence",
        candidate_source_sha256=sha(source),
        image_sha256=sha(binary),
        image_bytes=binary.stat().st_size,
        elf_sha256=sha(elf),
        baseline_image_sha256=sha(baseline / "efficientnet_b3_model.bin"),
        independent_candidate=True,
    )
    manifest["reference_semantics"] = a.reference_semantics
    manifest["existing_case_inputs"] = [
        {
            "case": name,
            "family": "gap",
            "overlay": True,
            "object": str(candidate / "kernel.o"),
            "object_sha256": sha(candidate / "kernel.o"),
            "adapter_source": str(candidate / "adapter.c"),
            "adapter_source_sha256": sha(candidate / "adapter.c"),
        }
    ]
    manifest["model_local_cpu_policy_kernels"] = []
    manifest["model_local_phase_batched_gap_kernels"] = [
        {
            "name": name,
            "kernel_sha256": sha(candidate / "kernel.o"),
            "source_sha256": sha(source),
            "policy": "uncached read phase and write-through output phase per channel",
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
        "scope": "standalone GAP build/ISA audit; no FPGA evidence",
        "elements": x.size,
        "input_source": a.reference_semantics,
        "full_shape": case["shape"],
        "case": a.case,
        "input_sha256": sha(a.output / "input.f32"),
        "expected_sha256": sha(a.output / "w8a8-logits.f32"),
        "scalar_image_sha256": sha(baseline / "efficientnet_b3_model.bin"),
        "phase_image_sha256": sha(binary),
        "candidate_source_sha256": sha(source),
        "board_atol": 2e-6,
        "board_rtol": 2e-6,
        "post_download_comparison": "bit-exact",
        "full_model": False,
    }
    (a.output / "probe-build.json").write_text(json.dumps(report, indent=2) + chr(10))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

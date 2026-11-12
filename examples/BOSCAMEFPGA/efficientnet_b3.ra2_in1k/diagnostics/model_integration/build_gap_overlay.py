#!/usr/bin/env python3
"""Build separate phase-batched scalar GAP objects; never publish or submit."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402
from export import adapter_source  # noqa: E402
from build_model_image import run, sha  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/model-integration/gap-phase-overlay",
    )
    p.add_argument("--source", type=Path, default=HERE / "gap_phase_candidate.c")
    p.add_argument("--source-symbol", default="efficientnet_gap_phase_candidate")
    p.add_argument(
        "--host-validation",
        type=Path,
        default=ROOT
        / "build/model-integration/gap-phase-candidate/host-validation.json",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = a.source
    host = json.loads(a.host_validation.read_text())
    if host["status"] != "PASS" or host["source_sha256"] != sha(source):
        raise ValueError("candidate lacks current exact sourceIR host validation")
    checked = {r["case"]: r for r in host["cases"]}
    clang = REPO / "buddy-mlir/llvm/build/bin/clang"
    restrict = REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools/restrict_fpga_assembly.py"
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
    results = []
    for case in inventory():
        if case["family"] != "gap":
            continue
        name = case["name"]
        c = case["constexprs"]
        check = checked[name]
        if check["source_ir_sha256"] != sha(a.kernels / name / "kernel.ll"):
            raise ValueError("reference IR changed " + name)
        original = a.kernels / name / "adapter.c"
        out = a.output / name
        out.mkdir(exist_ok=True)
        symbol = case["implementation_symbol"]
        # Generate from the declared native binding; no dependency on a legacy
        # grid-loop adapter surviving in the current production directory.
        (out / "adapter.c").write_text(adapter_source(case))
        run(
            [
                clang,
                *flags,
                "-D" + a.source_symbol + "=" + symbol,
                "-S",
                source,
                "-o",
                out / "kernel.s",
            ],
            out / "compile.log",
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(restrict),
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
            input=(out / "kernel.s").read_text(),
            capture_output=True,
            text=True,
        )
        (out / "restrict.log").write_text(proc.stderr)
        if proc.returncode:
            raise RuntimeError("ISA restriction failed " + name)
        (out / "kernel.nr.s").write_text(proc.stdout)
        run(
            [clang, *flags, "-c", out / "kernel.nr.s", "-o", out / "kernel.o"],
            out / "object.log",
        )
        run(
            [
                clang,
                *flags,
                "-I",
                ROOT,
                "-c",
                out / "adapter.c",
                "-o",
                out / "adapter.o",
            ],
            out / "adapter.log",
        )
        result = {
            "name": name,
            "status": "PASS",
            "implementation": "scalar_phase_batched_c",
            "spatial": c["SPATIAL"],
            "channels": c["C"],
            "BLOCK": c["BLOCK"],
            "source_sha256": sha(source),
            "kernel_sha256": sha(out / "kernel.o"),
            "adapter_sha256": sha(out / "adapter.c"),
            "original_adapter_sha256": sha(original),
            "source_ir_sha256": check["source_ir_sha256"],
            "host_exact_rounds": check["rounds"],
            "policy": "one uncached read phase + one write-through output phase per channel",
            "scope": "separate object build/ISA restriction + full-shape host exact comparison; board coverage recorded separately",
            "production_modified": False,
            "numerical_board_status": "NOT_RUN",
        }
        (out / "build-result.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print("PASS", name, flush=True)
    if len(results) != 13:
        raise ValueError("expected13GAP")
    (a.output / "overlay-manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "cases": results,
                "source_sha256": sha(source),
                "host_validation_sha256": sha(a.host_validation),
                "host_reference_semantics": host.get(
                    "reference_semantics", "original emitted LLVM reduction order"
                ),
                "scope": "isolated candidate; not default model integration or all13board validation",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

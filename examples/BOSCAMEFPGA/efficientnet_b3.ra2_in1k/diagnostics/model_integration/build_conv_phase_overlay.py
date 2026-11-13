#!/usr/bin/env python3
"""Build an isolated LLVM-phased im2col overlay; never publish or submit."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402
from add_conv_patch_ddr_phases import transform  # noqa: E402
from build_conv_im2col_phase_candidate import function_bytes  # noqa: E402
from build_model_image import run, sha  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/diagnostics/conv-im2col-phase-all18",
    )
    p.add_argument("--case", action="append")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = [c for c in inventory() if c["family"] in ("conv2d_f32", "dwconv2d_f32")]
    if a.case:
        cases = [c for c in cases if c["name"] in a.case]
        if {c["name"] for c in cases} != set(a.case):
            raise ValueError("unknown bounded convolution")
    llvm = REPO / "buddy-mlir/llvm/build/bin"
    tools = REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools"
    phase_tool = ROOT / "tools/add_conv_patch_ddr_phases.py"

    def worker(case):
        name = case["name"]
        original = a.kernels / name
        out = a.output / name
        out.mkdir(exist_ok=True)
        for filename in (
            "kernel.ll",
            "kernel.boscame.mlir",
            "kernel.frontend.mlir",
            "kernel.ttir",
            "frontend.json",
            "contract-manifest.json",
            "adapter.c",
        ):
            shutil.copy2(original / filename, out / filename)
        transformed, phase_report = transform(
            (original / "kernel.ll").read_text(), case
        )
        (out / "kernel.phase.ll").write_text(transformed)
        (out / "phase-transform.json").write_text(
            json.dumps(phase_report, indent=2) + "\n"
        )
        run(
            [
                llvm / "opt",
                "-O2",
                "-S",
                "-vectorize-loops=false",
                "-vectorize-slp=false",
                out / "kernel.phase.ll",
                "-o",
                out / "kernel.optimized.ll",
            ],
            out / "llvm-opt.log",
        )
        run(
            [
                llvm / "llc",
                out / "kernel.optimized.ll",
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                out / "kernel.s",
            ],
            out / "llc.log",
        )
        encoded = subprocess.run(
            [
                sys.executable,
                str(tools / "ame_to_word.py"),
                "--gpr-mode=fixed",
                "--preserve-matrix-registers",
            ],
            input=(out / "kernel.s").read_text(),
            capture_output=True,
            text=True,
        )
        (out / "encode.log").write_text(encoded.stderr)
        if encoded.returncode:
            raise ValueError("AME encoding failed " + name)
        (out / "kernel.encoded.s").write_text(encoded.stdout)
        restricted = subprocess.run(
            [
                sys.executable,
                str(tools / "restrict_fpga_assembly.py"),
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
            input=encoded.stdout,
            capture_output=True,
            text=True,
        )
        (out / "restrict.log").write_text(restricted.stderr)
        if restricted.returncode:
            raise ValueError("ISA restriction failed " + name)
        (out / "kernel.nr.s").write_text(restricted.stdout)
        run(
            [
                llvm / "clang",
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdcv_zicsr_zifencei",
                "-mabi=lp64d",
                "-mno-relax",
                "-c",
                out / "kernel.nr.s",
                "-o",
                out / "kernel.o",
            ],
            out / "object.log",
        )
        before = function_bytes(original / "kernel.o")
        after = function_bytes(out / "kernel.o")
        raw = "triton_" + name + "__im2col_quantize"
        if before.keys() != after.keys():
            raise ValueError("entry symbol set changed " + name)
        unchanged = {n: before[n] == after[n] for n in before if n != raw}
        if not all(unchanged.values()):
            raise ValueError(
                "non-im2col function bytes changed " + name + repr(unchanged)
            )
        result = {
            "name": name,
            "case": name,
            "status": "PASS_BUILD_AND_STATIC_AUDIT_ONLY",
            "board_status": "NOT_RUN",
            "kernel_sha256": sha(out / "kernel.o"),
            "baseline_kernel_sha256": sha(original / "kernel.o"),
            "adapter_sha256": sha(out / "adapter.c"),
            "source_ll_sha256": sha(original / "kernel.ll"),
            "phase_ll_sha256": sha(out / "kernel.phase.ll"),
            "phase_tool_sha256": sha(phase_tool),
            "builder_sha256": sha(Path(__file__)),
            "ISA_restriction": "PASS",
            "all_original_LLVM_math_unchanged": True,
            "non_im2col_function_bytes_unchanged": unchanged,
            "adapter_byte_identical": sha(out / "adapter.c")
            == sha(original / "adapter.c"),
            "MMA_modified": False,
            "production_modified": False,
            "phase_transform": phase_report,
            "provenance_note": "copied BOSCAME/frontend artifacts describe original math; actual object uses kernel.phase.ll with explicit CSR phases before scalar O2",
        }
        (out / "candidate-manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print("PASS", name, result["kernel_sha256"], flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for f in as_completed([pool.submit(worker, c) for c in cases]):
            results.append(f.result())
    summary = {
        "status": "PASS_BUILD_AND_STATIC_AUDIT_ONLY",
        "expected": len(cases),
        "completed": len(results),
        "board_status": "NOT_RUN",
        "production_modified": False,
        "cases": sorted(results, key=lambda x: x["case"]),
        "phase_tool_sha256": sha(phase_tool),
        "builder_sha256": sha(Path(__file__)),
    }
    (a.output / "overlay-manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

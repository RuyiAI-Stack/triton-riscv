#!/usr/bin/env python3
"""Compile and validate a separate full-array RVV overlay for all model SiLU.

Production artifacts and sources are retained. Each typed adapter keeps the
original descriptor checks and calls one independently named RVV C function.
"""

import argparse
import ctypes
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BASE = ROOT / "build/model-integration"
sys.path[:0] = [str(ROOT)]
from cases import inventory  # noqa: E402
from build_model_image import run, sha  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan", type=Path, default=BASE / "physical-plan/physical-plan.json"
    )
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument("--connectors", type=Path, default=BASE / "connectors-ddr-policy")
    p.add_argument("--output", type=Path, default=BASE / "silu-rvv-overlay")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = HERE / "silu_rvv_candidate.c"
    library = BASE / "silu-rvv-candidate/host.so"
    lib = ctypes.CDLL(str(library.resolve()))
    fn = lib.efficientnet_silu_rvv_candidate
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
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
    specs = []
    for case in inventory():
        if case["family"] == "silu":
            specs.append(
                (
                    case["name"],
                    math.prod(case["shape"]),
                    a.kernels / case["name"] / "adapter.c",
                    "existing_case",
                )
            )
    plan = json.loads(a.plan.read_text())
    for spec in plan["connectors"]:
        if spec["kernel"] == "silu":
            specs.append(
                (
                    spec["name"],
                    spec["constexprs"]["COUNT"],
                    a.connectors / spec["name"] / "adapter.c",
                    "connector",
                )
            )
    results = []
    rng = np.random.default_rng(912341)
    for name, count, original, kind in specs:
        out = a.output / name
        out.mkdir(exist_ok=True)
        symbol = "efficientnet_rvv_" + name
        text = original.read_text()
        marker = "  for (int32_t x"
        if text.count(marker) != 1:
            raise ValueError("unexpected adapter " + name)
        wrapper = (
            "extern void "
            + symbol
            + "(const float *,float *,__SIZE_TYPE__);\n"
            + text.split(marker)[0]
        )
        wrapper += "  " + symbol + "((const float *)p0,(float *)p1,%du);\n}\n" % count
        (out / "adapter.c").write_text(wrapper)
        run(
            [
                clang,
                *flags,
                "-Defficientnet_silu_rvv_candidate=" + symbol,
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
            raise RuntimeError("restriction " + name)
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
        # Full logical count plus two guards; tiny shapes exercise masked tails.
        x = rng.uniform(-100, 100, count).astype(np.float32)
        x[::251] = 0.0
        x[1::251] = -0.0
        output = np.full(count + 32, np.float32(-371.25))
        inside = output[16 : 16 + count]
        fn(x.ctypes.data, inside.ctypes.data, count)
        with np.errstate(over="ignore", under="ignore"):
            expected = (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
        np.testing.assert_allclose(inside, expected, atol=2e-6, rtol=2e-6)
        assert np.all(output[:16] == np.float32(-371.25)) and np.all(
            output[-16:] == np.float32(-371.25)
        )
        result = {
            "name": name,
            "kind": kind,
            "elements": count,
            "status": "PASS",
            "native_host_max_abs_error": float(np.max(np.abs(inside - expected))),
            "guards_intact": True,
            "source_sha256": sha(source),
            "kernel_sha256": sha(out / "kernel.o"),
            "adapter_sha256": sha(out / "adapter.c"),
            "original_adapter_sha256": sha(original),
            "policy": "one uncached read phase and one write-through store phase per <=16 values",
            "scope": "full-shape native scalar-equivalent validation + RVV compilation/restriction; representative board evidence recorded separately",
        }
        (out / "build-result.json").write_text(json.dumps(result, indent=2) + chr(10))
        results.append(result)
        print("PASS", name, count, flush=True)
    report = {
        "status": "PASS",
        "expected": 22,
        "completed": len(results),
        "existing_cases": sum(r["kind"] == "existing_case" for r in results),
        "connectors": sum(r["kind"] == "connector" for r in results),
        "native_host_elements": sum(r["elements"] for r in results),
        "atol": 2e-6,
        "rtol": 2e-6,
        "source_sha256": sha(source),
        "source": str(source),
        "host_library_sha256": sha(library),
        "results": results,
        "production_sources_modified": False,
    }
    if len(results) != 22:
        raise ValueError("expected20case+2connector SiLU")
    (a.output / "overlay-manifest.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build isolated direct-output RVV SiLU objects; never publish or submit.

The adapters are regenerated from case/exporter and connector specifications,
so no previous overlay or ignored native object is required. PASS in this
builder means compilation, ISA restriction and host validation only. Native
FPGA validation must be recorded separately before production publication.
"""

import argparse
import ctypes
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BASE = ROOT / "build/model-integration"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402
from export import adapter_source  # noqa: E402
from build_connectors import adapter as connector_adapter  # noqa: E402
from build_model_image import run, sha  # noqa: E402


def native_adapter(original, name, count):
    """Retain every public descriptor check and replace the grid launch."""
    marker = "  for (int32_t x"
    if original.count(marker) != 1:
        raise ValueError("unexpected generated scalar adapter " + name)
    symbol = "efficientnet_rvv_" + name
    text = "extern void " + symbol + "(const float *,float *,__SIZE_TYPE__);\n"
    text += original.split(marker)[0]
    text += "  " + symbol + "((const float *)p0,(float *)p1,%du);\n}\n" % count
    return symbol, text


def specifications(plan):
    specs = {}
    for case in inventory():
        if case["family"] != "silu":
            continue
        # Request the explicit scalar reference adapter from the exporter, then
        # retain its exact public checks while replacing only its grid call.
        reference = dict(case)
        reference.pop("implementation", None)
        reference.pop("implementation_symbol", None)
        count = math.prod(case["shape"])
        symbol, text = native_adapter(adapter_source(reference), case["name"], count)
        specs[case["name"]] = (count, "existing_case", symbol, text)
    for spec in plan["connectors"]:
        if spec["kernel"] != "silu":
            continue
        count = spec["constexprs"]["COUNT"]
        symbol, text = native_adapter(connector_adapter(spec), spec["name"], count)
        specs[spec["name"]] = (count, "connector", symbol, text)
    return specs


def host_shape_check(function, count, rng):
    """Exercise the full logical count and guard both sides of its output."""
    x = rng.uniform(-100, 100, count).astype(np.float32)
    x[::251] = 0.0
    x[1::251] = -0.0
    output = np.full(count + 32, np.float32(-371.25))
    inside = output[16 : 16 + count]
    function(x.ctypes.data, inside.ctypes.data, count)
    with np.errstate(over="ignore", under="ignore"):
        expected = (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    np.testing.assert_allclose(inside, expected, atol=2e-6, rtol=2e-6)
    if not (
        np.all(output[:16] == np.float32(-371.25))
        and np.all(output[-16:] == np.float32(-371.25))
    ):
        raise ValueError("host output guards changed")
    return float(np.max(np.abs(inside - expected)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "silu-rvv-direct-output-overlay"
    )
    p.add_argument(
        "--plan", type=Path, default=BASE / "physical-plan/physical-plan.json"
    )
    p.add_argument(
        "--host-validation",
        type=Path,
        default=BASE / "silu-rvv-direct-output-candidate/host-validation.json",
    )
    p.add_argument(
        "--source-overlay",
        type=Path,
        default=BASE / "silu-rvv-overlay",
        help="optional adapter hash comparison; never a build dependency",
    )
    p.add_argument("--case", action="append")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = HERE / "silu_rvv_direct_output_candidate.c"
    host = json.loads(a.host_validation.read_text())
    if host["status"] != "PASS" or host["candidate_source_sha256"] != sha(source):
        raise ValueError("current candidate host validation missing")
    equivalence_path = a.host_validation.parent / "host-equivalence.json"
    equivalence = json.loads(equivalence_path.read_text())
    if (
        not host.get("previous_host_bit_exact")
        or host.get("host_equivalence_sha256") != sha(equivalence_path)
        or equivalence.get("status") != "PASS"
        or equivalence.get("candidate_source_sha256") != sha(source)
        or equivalence.get("previous_source_sha256")
        != sha(HERE / "silu_rvv_candidate.c")
        or equivalence.get("broad_bit_mismatches") != 0
        or equivalence.get("stem_bit_mismatches") != 0
    ):
        raise ValueError("current old/new bit-exact host equivalence missing")
    library = a.host_validation.parent / "host.so"
    if sha(library) != host["host_library_sha256"]:
        raise ValueError("candidate host library changed; rerun host checker")
    lib = ctypes.CDLL(str(library.resolve()))
    function = lib.efficientnet_silu_rvv_direct_output_candidate
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    specs = specifications(json.loads(a.plan.read_text()))
    expected = len(specs)
    if a.case:
        if not set(a.case) <= specs.keys():
            raise ValueError("unknown SiLU candidate case")
        specs = {name: specs[name] for name in set(a.case)}
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
    rng = np.random.default_rng(912341)
    for name, (count, kind, symbol, text) in sorted(specs.items()):
        out = a.output / name
        out.mkdir(exist_ok=True)
        (out / "adapter.c").write_text(text)
        baseline = a.source_overlay / name / "adapter.c"
        identical = (
            sha(baseline) == sha(out / "adapter.c") if baseline.exists() else None
        )
        if identical is False:
            raise ValueError("regenerated native adapter differs from baseline " + name)
        run(
            [
                clang,
                *flags,
                "-Defficientnet_silu_rvv_direct_output_candidate=" + symbol,
                "-S",
                source,
                "-o",
                out / "kernel.s",
            ],
            out / "compile.log",
        )
        raw = (out / "kernel.s").read_text()
        vector_stores = [
            line.strip()
            for line in raw.splitlines()
            if re.match(
                r"\s+(?:vse[0-9]+\.v|vsse[0-9]+\.v|vsuxei[0-9]+\.v|vsoxei[0-9]+\.v|vs[1248]r\.v)\s",
                line,
            )
        ]
        if len(vector_stores) != 1 or not vector_stores[0].startswith("vse32.v"):
            raise ValueError(
                "candidate must have one output vector store and no vector stack spills "
                + name
                + str(vector_stores)
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
            input=raw,
            capture_output=True,
            text=True,
        )
        (out / "restrict.log").write_text(proc.stderr)
        if proc.returncode:
            raise RuntimeError("ISA restriction " + name)
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
        error = host_shape_check(function, count, rng)
        result = {
            "name": name,
            "case": name,
            "kind": kind,
            "elements": count,
            "status": "PASS",
            "status_scope": "build, ISA restriction and host scalar-equivalent validation only",
            "candidate_only": True,
            "source": str(source),
            "source_path": str(source),
            "source_sha256": sha(source),
            "kernel_sha256": sha(out / "kernel.o"),
            "kernel_o_sha256": sha(out / "kernel.o"),
            "adapter_sha256": sha(out / "adapter.c"),
            "baseline_adapter_byte_identical": identical,
            "symbol": symbol,
            "vector_store_instructions": vector_stores,
            "ISA_restriction": "PASS",
            "host_validation": str(a.host_validation),
            "host_validation_sha256": sha(a.host_validation),
            "host_equivalence_path": str(equivalence_path),
            "host_equivalence_sha256": sha(equivalence_path),
            "previous_host_bit_exact": True,
            "host_library_sha256": sha(library),
            "native_host_max_abs_error": error,
            "host_atol": 2e-6,
            "host_rtol": 2e-6,
            "guards_intact": True,
            "board_status": "NOT_RUN",
            "numerical_board_status": "NOT_RUN",
            "policy": "CPU input/scratch writes; RVV scratch reads; direct output vector store; "
            "CPU special output writes after fence; no CPU load of RVV-stored scratch",
            "production_modified": False,
        }
        for filename in ("candidate-manifest.json", "build-result.json"):
            (out / filename).write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print("PASS", name, sha(out / "kernel.o"), flush=True)
    summary = {
        "status": "PASS",
        "status_scope": "build, ISA restriction and host validation only",
        "source_path": str(source),
        "source_sha256": sha(source),
        "expected": len(specs),
        "completed": len(results),
        "all_model_specializations": expected,
        "existing_cases": sum(r["kind"] == "existing_case" for r in results),
        "connectors": sum(r["kind"] == "connector" for r in results),
        "native_host_elements": sum(r["elements"] for r in results),
        "cases": results,
        "production_modified": False,
        "board_status": "NOT_RUN",
        "builder_source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(a.plan),
    }
    for filename in ("candidate-summary.json", "overlay-manifest.json"):
        (a.output / filename).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check an isolated explicit Torch AVX2 reduction order against pinned CPU mean.

This intentionally changes the original emitted Triton GAP arithmetic order.
It leaves primary references and production kernels unchanged, and proves only
host arithmetic. Board coverage is a separate full-output suite.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=HERE / "gap_torch_order_candidate.c")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/model-integration/gap-torch-order-candidate",
    )
    p.add_argument(
        "--clang", type=Path, default=REPO / "buddy-mlir/llvm/build/bin/clang"
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0":
        raise ValueError(
            "candidate validated only for pinned Torch2.10 CPU SumKernel source"
        )
    torch.set_num_threads(4)
    shutil.copy2(a.source, a.output / "source-snapshot.c")
    lib = a.output / "host.so"
    command = [
        str(a.clang),
        "-shared",
        "-fPIC",
        "-O2",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
        str(a.source),
        "-o",
        str(lib),
    ]
    proc = subprocess.run(command, text=True, capture_output=True)
    (a.output / "host-compile.log").write_text(
        json.dumps(command) + "\n" + proc.stdout + proc.stderr
    )
    if proc.returncode:
        raise RuntimeError("host compile failed")
    fn = ctypes.CDLL(str(lib.resolve())).efficientnet_gap_torch_order_candidate
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    rng = np.random.default_rng(157401)
    results = []
    for case in inventory():
        if case["family"] != "gap":
            continue
        name = case["name"]
        c = case["constexprs"]
        sp, ch, block = c["SPATIAL"], c["C"], c["BLOCK"]
        rounds = []
        for fixture in range(3):
            x = (
                rng.uniform(-32, 32, (ch, sp)).astype(np.float32)
                if fixture == 0
                else np.full(
                    (ch, sp), np.float32(-0.0) if fixture == 1 else np.float32(1e-30)
                )
            )
            if fixture == 0:
                x.flat[::37] = np.float32(1e4)
                x.flat[::41] = np.float32(-1e4)
            expected = torch.from_numpy(x).mean(1).numpy()
            guarded = np.full(ch + 32, np.float32(-371.25))
            actual = guarded[16:-16]
            fn(x.ctypes.data, actual.ctypes.data, sp, ch, block)
            bad = int(
                np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))
            )
            guards = bool(
                np.all(guarded[:16] == -371.25) and np.all(guarded[-16:] == -371.25)
            )
            if bad or not guards:
                raise ValueError((name, fixture, bad, guards))
            rounds.append(
                {
                    "fixture": fixture,
                    "outputs": ch,
                    "bit_mismatches": bad,
                    "host_guards": guards,
                }
            )
            if fixture == 0:
                out = a.output / "fixtures" / name
                out.mkdir(parents=True, exist_ok=True)
                x.tofile(out / "input.f32")
                expected.tofile(out / "w8a8-logits.f32")
        ir = ROOT / "build/kernels" / name / "kernel.ll"
        results.append(
            {
                "case": name,
                "spatial": sp,
                "channels": ch,
                "block": block,
                "status": "PASS",
                "source_ir": str(ir),
                "source_ir_sha256": sha(ir),
                "source_ir_role": "provenance only; arithmetic order intentionally differs",
                "rounds": rounds,
            }
        )
        print("PASS", name, flush=True)
    report = {
        "status": "PASS",
        "source_sha256": sha(a.source),
        "cases": results,
        "reference_semantics": "pinned Torch2.10 CPU AVX2 eight-lane, four-ILP, four-level cascade then true f32 division",
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "host_library_sha256": sha(lib),
        "scope": "all13 full shapes x3 finite fixtures; exact pinned Torch CPU mean comparison; no FPGA evidence; host guards only",
    }
    (a.output / "host-validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare isolated phase-batched GAP to exact host-compiled generated LLVM IR."""

import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402

out = ROOT / "build/model-integration/gap-phase-candidate"
out.mkdir(parents=True, exist_ok=True)
src = Path(__file__).with_name("gap_phase_candidate.c")
clang = ROOT.parents[2] / "buddy-mlir/llvm/build/bin/clang"
subprocess.run(
    [
        str(clang),
        "-shared",
        "-fPIC",
        "-O2",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
        str(src),
        "-o",
        str(out / "host.so"),
    ],
    check=True,
)
fn = ctypes.CDLL(str(out / "host.so")).efficientnet_gap_phase_candidate
fn.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]


class Memref(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


class Unranked(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int64), ("descriptor", ctypes.c_void_p)]


rng = np.random.default_rng(157401)
results = []
for c in inventory():
    if c["family"] != "gap":
        continue
    name = c["name"]
    sp = c["constexprs"]["SPATIAL"]
    ch = c["constexprs"]["C"]
    block = c["constexprs"]["BLOCK"]
    ir = ROOT / "build/kernels" / name / "kernel.ll"
    lib = out / (name + ".so")
    proc = subprocess.run(
        [
            str(clang),
            "-shared",
            "-fPIC",
            "-O2",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-ffp-contract=off",
            str(ir),
            "-o",
            str(lib),
        ],
        capture_output=True,
        text=True,
    )
    (out / (name + "-source-ir-compile.log")).write_text(proc.stderr)
    assert proc.returncode == 0
    raw = getattr(ctypes.CDLL(str(lib)), "_mlir_ciface_triton_" + name)
    raw.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int32] * 6
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
        expected = np.zeros(ch, np.float32)
        guarded = np.full(ch + 32, np.float32(-371.25))
        got = guarded[16:-16]
        xm = Memref(x.ctypes.data, x.ctypes.data, 0, x.size, 1)
        ym = Memref(expected.ctypes.data, expected.ctypes.data, 0, ch, 1)
        xu = Unranked(1, ctypes.addressof(xm))
        yu = Unranked(1, ctypes.addressof(ym))
        for channel in range(ch):
            raw(ctypes.byref(xu), ctypes.byref(yu), ch, 1, 1, channel, 0, 0)
        fn(x.ctypes.data, got.ctypes.data, sp, ch, block)
        mismatches = int(
            np.count_nonzero(got.view(np.uint32) != expected.view(np.uint32))
        )
        assert mismatches == 0, (name, fixture, mismatches)
        assert np.all(guarded[:16] == -371.25) and np.all(guarded[-16:] == -371.25)
        rounds.append(
            {
                "fixture": fixture,
                "outputs": ch,
                "bit_mismatches": mismatches,
                "guards": True,
            }
        )
        if fixture == 0:
            fixture_dir = out / "fixtures" / name
            fixture_dir.mkdir(parents=True, exist_ok=True)
            x.tofile(fixture_dir / "input.f32")
            expected.tofile(fixture_dir / "w8a8-logits.f32")
            if name == "gap_144x144x40":
                x.tofile(out / "input.f32")
                expected.tofile(out / "w8a8-logits.f32")
    results.append(
        {
            "case": name,
            "spatial": sp,
            "channels": ch,
            "block": block,
            "status": "PASS",
            "source_ir": str(ir),
            "source_ir_sha256": hashlib.sha256(ir.read_bytes()).hexdigest(),
            "rounds": rounds,
        }
    )
    print("PASS", name, flush=True)
(out / "host-validation.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "cases": results,
            "scope": "all13 full shapes x3 finite fixtures; exact host-compiled emitted LLVM IR comparison; no board evidence yet",
        },
        indent=2,
    )
    + "\n"
)

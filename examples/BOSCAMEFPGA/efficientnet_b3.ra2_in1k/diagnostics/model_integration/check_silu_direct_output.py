#!/usr/bin/env python3
"""Check candidate binary32 semantics against f64 and actual scalar BOSCAME IR.

This validates the scalar-equivalent host branch, not the RVV machine code;
the separate full-output FPGA probe is required for the latter.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BASE = ROOT / "build/model-integration"
sys.path.insert(0, str(REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "silu-rvv-direct-output-candidate"
    )
    p.add_argument("--buddy-opt", type=Path, default=BASE / "pinned-tools/buddy-opt")
    p.add_argument(
        "--stem",
        type=Path,
        default=BASE / "prefix8/stem.f32",
        help="actual W8A8 stem output, without inserted boundary values",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = HERE / "silu_rvv_direct_output_candidate.c"
    library = a.output / "host.so"
    command = [
        "cc",
        "-shared",
        "-fPIC",
        "-O2",
        "-fno-tree-vectorize",
        "-fno-tree-slp-vectorize",
        "-ffp-contract=off",
        str(source),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True, capture_output=True)
    lib = ctypes.CDLL(str(library.resolve()))
    f = lib.efficientnet_silu_rvv_direct_output_candidate
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    rng = np.random.default_rng(93421)
    specials = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            np.nextafter(np.float32(0), np.float32(1)),
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ],
        np.float32,
    )
    x = np.concatenate(
        [
            np.linspace(-110, 110, 1000001, dtype=np.float32),
            rng.integers(0, 2**32, size=1000000, dtype=np.uint32).view(np.float32),
            specials,
        ]
    )
    out = np.empty_like(x)
    f(x.ctypes.data, out.ctypes.data, x.size)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        expected = (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    finite = np.isfinite(expected)
    error = np.abs(out[finite].astype(np.float64) - expected[finite].astype(np.float64))
    assert np.all(error <= 2e-6 + 2e-6 * np.abs(expected[finite].astype(np.float64)))
    assert np.array_equal(np.isnan(out), np.isnan(expected))
    assert np.array_equal(np.isinf(out), np.isinf(expected))
    zeros = expected == 0
    assert np.array_equal(np.signbit(out[zeros]), np.signbit(expected[zeros]))
    # Prove that rearranging scratch ownership leaves the CPU mathematical
    # recipe unchanged, including NaN payloads and signed zeros. Compile the
    # old source into this isolated output directory; never touch its library.
    previous_source = HERE / "silu_rvv_candidate.c"
    previous_library = a.output / "previous-host.so"
    previous_command = list(command)
    previous_command[-3:] = [str(previous_source), "-o", str(previous_library)]
    subprocess.run(previous_command, check=True, capture_output=True)
    previous_lib = ctypes.CDLL(str(previous_library.resolve()))
    previous = previous_lib.efficientnet_silu_rvv_candidate
    previous.argtypes = f.argtypes
    old = np.empty_like(x)
    previous(x.ctypes.data, old.ctypes.data, x.size)
    mismatches = int(np.count_nonzero(old.view(np.uint32) != out.view(np.uint32)))
    assert mismatches == 0, ("old/new broad host bit mismatch", mismatches)
    stem = np.fromfile(a.stem, dtype="<f4")
    assert stem.size == 829440, ("unexpected real stem size", stem.size)
    stem_old = np.empty_like(stem)
    stem_new = np.empty_like(stem)
    previous(stem.ctypes.data, stem_old.ctypes.data, stem.size)
    f(stem.ctypes.data, stem_new.ctypes.data, stem.size)
    stem_mismatches = int(
        np.count_nonzero(stem_old.view(np.uint32) != stem_new.view(np.uint32))
    )
    assert stem_mismatches == 0, ("old/new real stem bit mismatch", stem_mismatches)
    equivalence = {
        "status": "PASS",
        "scope": "host binary32 mathematical recipe only; not native RVV/FPGA proof",
        "previous_source_path": str(previous_source),
        "previous_source_sha256": sha(previous_source),
        "candidate_source_path": str(source),
        "candidate_source_sha256": sha(source),
        "previous_host_library_sha256": sha(previous_library),
        "candidate_host_library_sha256": sha(library),
        "broad_input_elements": x.size,
        "broad_input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "broad_bit_mismatches": mismatches,
        "broad_output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
        "stem_input_path": str(a.stem),
        "stem_input_sha256": sha(a.stem),
        "stem_elements": stem.size,
        "stem_bit_mismatches": stem_mismatches,
        "stem_output_sha256": hashlib.sha256(stem_new.tobytes()).hexdigest(),
        "nan_payloads_and_signed_zeros_compared_bitwise": True,
    }
    equivalence_path = a.output / "host-equivalence.json"
    equivalence_path.write_text(json.dumps(equivalence, indent=2) + chr(10))
    # Validate the host branch against the real generated scalar exp recipe,
    # including NaN/Inf and thresholds rather than only reusing its polynomial.
    boscame = ROOT / "build/kernels/silu_1x40x144x144/kernel.boscame.mlir"
    loops = a.output / "scalar-reference.loops.mlir"
    subprocess.run(
        [
            str(a.buddy_opt),
            str(boscame),
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            str(loops),
        ],
        check=True,
        capture_output=True,
    )
    small = np.resize(
        np.concatenate([specials, np.linspace(-110, 110, 113, dtype=np.float32)]), 128
    ).copy()
    actual = np.empty_like(small)
    f(small.ctypes.data, actual.ctypes.data, small.size)
    machine = Machine(loops, "triton_silu_1x40x144x144")
    raw_input = bytearray(small.tobytes())
    raw_output = bytearray(128 * 4)
    machine.launch(
        {"%arg0": memoryview(raw_input), "%arg1": memoryview(raw_output)},
        {"%arg2": 6480, "%arg3": 1, "%arg4": 1, "%arg5": 0, "%arg6": 0, "%arg7": 0},
    )
    reference = np.frombuffer(raw_output, dtype="<f4")
    np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6, equal_nan=True)
    comparable = np.isfinite(reference)
    report = {
        "status": "PASS",
        "scope": "host scalar-equivalent candidate and actual scalar BOSCAME IR; not native RVV/FPGA evidence",
        "elements": x.size,
        "f64_reference_finite_elements": int(finite.sum()),
        "max_abs_error": float(error.max()),
        "atol": 2e-6,
        "rtol": 2e-6,
        "nan_classification_matches": True,
        "inf_classification_matches": True,
        "signed_zero_matches": True,
        "actual_scalar_ir_elements": 128,
        "actual_scalar_ir_max_abs_error": float(
            np.max(np.abs(actual[comparable] - reference[comparable]))
        ),
        "candidate_source_sha256": sha(source),
        "host_library_sha256": sha(library),
        "previous_source_sha256": sha(previous_source),
        "previous_host_bit_exact": True,
        "previous_host_broad_elements": x.size,
        "previous_host_stem_elements": stem.size,
        "host_equivalence_path": str(equivalence_path),
        "host_equivalence_sha256": sha(equivalence_path),
        "actual_scalar_boscame_sha256": sha(boscame),
        "buddy_opt_sha256": sha(a.buddy_opt),
    }
    (a.output / "host-validation.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

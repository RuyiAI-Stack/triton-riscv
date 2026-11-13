#!/usr/bin/env python3
"""Build a finite-stress full-output SiLU probe using the current native overlay.

This does not publish production artifacts or submit a scheduler job. Expected
bytes come from the exact host binary32 recipe, with independently checked old/
new equivalence and f64 tolerance. It is one shape, not complete model evidence.
"""

import argparse
import copy
import ctypes
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "build/model-integration"
from build_model_image import run, sha  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "silu-direct-output-finite-stress"
    )
    p.add_argument(
        "--overlay", type=Path, default=BASE / "silu-rvv-direct-output-overlay"
    )
    p.add_argument("--stem", type=Path, default=BASE / "prefix8/stem.f32")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.overlay = a.overlay.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    source = HERE / "silu_rvv_direct_output_candidate.c"
    evidence_dir = BASE / "silu-rvv-direct-output-candidate"
    evidence = json.loads((evidence_dir / "host-validation.json").read_text())
    if (
        evidence.get("status") != "PASS"
        or not evidence.get("previous_host_bit_exact")
        or evidence.get("candidate_source_sha256") != sha(source)
    ):
        raise ValueError("current independent host validation/equivalence required")
    library = evidence_dir / "host.so"
    if evidence["host_library_sha256"] != sha(library):
        raise ValueError("host candidate library changed")
    lib = ctypes.CDLL(str(library.resolve()))
    function = lib.efficientnet_silu_rvv_direct_output_candidate
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    x = np.fromfile(a.stem, dtype="<f4").copy()
    if x.size != 829440:
        raise ValueError("expected829440 real stem values")
    rng = np.random.default_rng(510294)
    x[x.size // 2 :] = rng.uniform(-110, 110, x.size - x.size // 2).astype(np.float32)
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
    for start in range(0, x.size - boundaries.size, 8191):
        x[start : start + boundaries.size] = boundaries
    x[-boundaries.size :] = boundaries[::-1]
    expected = np.empty_like(x)
    function(x.ctypes.data, expected.ctypes.data, x.size)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        f64 = (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    np.testing.assert_allclose(expected, f64, atol=2e-6, rtol=2e-6)
    x.tofile(a.output / "input.f32")
    expected.tofile(a.output / "w8a8-logits.f32")
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
    (a.output / "physical-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for filename in ("parameters.bin", "parameters.json"):
        shutil.copy2(BASE / "physical-plan" / filename, a.output / filename)
    provenance = {
        "status": "HOST_BINARY32_SILU_RECIPE_REFERENCE",
        "scope": "one SiLU finite-stress probe; not model logits, NaN/Inf board proof or full-model evidence",
        "source_sha256": sha(source),
        "host_library_sha256": sha(library),
        "host_validation_sha256": sha(evidence_dir / "host-validation.json"),
        "elements": x.size,
        "finite_input": True,
        "input_source": "first half real W8A8 stem, second half seeded uniform[-110,110], finite boundary blocks",
        "stem_input_path": str(a.stem),
        "stem_input_sha256": sha(a.stem),
        "input_sha256": sha(a.output / "input.f32"),
        "expected_sha256": sha(a.output / "w8a8-logits.f32"),
        "f64_atol": 2e-6,
        "f64_rtol": 2e-6,
        "finite_boundary_values": boundaries.tolist(),
    }
    (a.output / "reference-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
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
    run(
        [
            sys.executable,
            HERE / "build_model_image.py",
            "--plan-dir",
            a.output,
            "--runner",
            a.output / "runner",
            "--output",
            a.output / "image",
            "--bounded-probe",
            "--strict-probe",
            "--silu-overlay",
            a.overlay,
            "--silu-source",
            source,
        ],
        a.output / "build-image.log",
    )
    report = dict(
        provenance,
        status="PASS_BUILD_ONLY",
        board_status="NOT_RUN",
        image_manifest_sha256=sha(a.output / "image/image-manifest.json"),
    )
    (a.output / "probe-build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict local comparison of downloaded full-model outputs to emitted-SSA execution."""

import argparse
import json
from pathlib import Path

import numpy as np
from build_model_host import ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=ROOT / "build/model-host-emitted")
    p.add_argument(
        "--output",
        type=Path,
        help="new report path; default: RUN/local-emitted-validation.json (never overwritten)",
    )
    a = p.parse_args()
    output = a.output or a.run / "local-emitted-validation.json"
    if output.exists():
        p.error(
            "report already exists; preserve it and select a new --output: "
            + str(output)
        )
    manifest = json.loads((a.run / "artifacts/manifest.json").read_text())
    task = json.loads((a.run / "task.json").read_text())
    build = json.loads((a.run / "build-manifest.json").read_text())
    reference = json.loads((a.reference / "manifest.json").read_text())
    params = json.loads((a.run / "params-manifest.json").read_text())
    if (
        manifest["id"] != task["id"]
        or reference["params_sha256"] != params["params_sha256"]
    ):
        raise ValueError("Task or checkpoint mismatch")
    task_result = task.get("result") or {}
    details = task_result.get("details") or {}
    if (
        task.get("state") != "completed"
        or task_result.get("execution_status") != "exited"
        or task_result.get("exit_code") != 0
        or task_result.get("app_exit_code") != 0
        or details.get("image_sha256") != build["binary_sha256"]
        or sha(a.run / "electra-model.bin") != build["binary_sha256"]
        or details.get("application_started") is not True
        or details.get("reset_confirmed") is not True
        or details.get("ddr_readback_matches") is not True
        or details.get("artifact_error")
        or sha(a.run / "params-manifest.json") != build["params_manifest"]
    ):
        raise ValueError("Execution, uploaded image or parameter provenance mismatch")
    built_inputs = {x["path"]: x["sha256"] for x in build["inputs"]}
    records = []
    for path, digest in reference["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Host source changed after reference execution: " + path)
        if path in built_inputs and digest != built_inputs[path]:
            raise ValueError(
                "Uploaded graph/adapter and local graph/adapter source differs: " + path
            )
        if path.endswith("/kernel.ll"):
            parent = Path(path).parent
            artifacts = json.loads((parent / "build-result.json").read_text())[
                "artifacts"
            ]
            if (
                artifacts["kernel.ll"] != digest
                or artifacts["kernel.o"] != built_inputs[str(parent / "kernel.o")]
            ):
                raise ValueError(
                    "Local scalar LLVM and uploaded raw kernel provenance differs"
                )
    expected = {x["name"]: x for x in reference["files"]}
    shapes = {
        "embedding": [1, 128, 768],
        **{f"layer_{i:02d}": [1, 128, 768] for i in range(12)},
        "logits": [128],
    }
    if (
        set(expected) != set(shapes)
        or len(reference["files"]) != len(shapes)
        or set(expected) != {x["name"] for x in manifest["artifacts"]}
        or len(manifest["artifacts"]) != len(shapes)
    ):
        raise ValueError("Missing or extra full-model output")
    if sorted(task_result.get("artifacts") or [], key=lambda x: x["name"]) != sorted(
        manifest["artifacts"], key=lambda x: x["name"]
    ):
        raise ValueError(
            "Downloaded manifest differs from scheduler task artifact metadata"
        )
    for item in manifest["artifacts"]:
        name = item["name"]
        if item["dtype"] != "<f4" or item["shape"] != shapes[name]:
            raise ValueError("Output dtype/shape mismatch: " + name)
        actual_file, reference_file = (
            a.run / "artifacts" / (name + ".bin"),
            a.reference / (name + ".bin"),
        )
        if (
            sha(actual_file) != item["sha256"]
            or sha(reference_file) != expected[name]["sha256"]
            or actual_file.stat().st_size != item["nbytes"]
            or item["nbytes"] != expected[name]["bytes"]
            or item["nbytes"] != int(np.prod(shapes[name])) * 4
        ):
            raise ValueError("Output/reference file hash or length mismatch")
        got, want = (
            np.fromfile(actual_file, dtype="<f4"),
            np.fromfile(reference_file, dtype="<f4"),
        )
        actual_bits, expected_bits = got.view("<u4"), want.view("<u4")
        different = np.flatnonzero(actual_bits != expected_bits)
        diff = np.abs(got.astype("f8") - want.astype("f8"))
        finite = bool(np.isfinite(got).all() and np.isfinite(want).all())
        record = {
            "name": name,
            "elements": len(got),
            "finite": finite,
            "max_abs": float(diff.max()),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "errors": len(different),
            "bit_mismatches": len(different),
            "bit_exact": bool(len(different) == 0),
            "actual_sha256": item["sha256"],
            "expected_sha256": expected[name]["sha256"],
        }
        if len(different):
            index = int(different[0])
            record["first_mismatch"] = {
                "index": index,
                "actual_hex": f"0x{int(actual_bits[index]):08x}",
                "expected_hex": f"0x{int(expected_bits[index]):08x}",
            }
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "status": "PASS" if all(x["bit_exact"] for x in records) else "FAIL",
        "acceptance": "all FP32 uint32 bit patterns and tensor bytes identical",
        "task_id": task["id"],
        "image_sha256": build["binary_sha256"],
        "tensors": records,
        "elements": sum(x["elements"] for x in records),
        "verifier_sha256": sha(Path(__file__)),
        "reference_manifest_sha256": sha(a.reference / "manifest.json"),
        "scope": reference["scope"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

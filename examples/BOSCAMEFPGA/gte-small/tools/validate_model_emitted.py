#!/usr/bin/env python3
"""Compare downloaded GTE BINs with the identical local emitted-code execution.

Offline only. Every returned byte must match; report metrics do not relax this
criterion. Historical reports and downloaded artifacts are never overwritten.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def check_reference(host, kernels, build, contract, launcher):
    """Fail before submission if the local result belongs to another build."""
    oracle = json.loads((host / "manifest.json").read_text())
    assert (
        oracle["status"] == "LOCAL_GRAPH_EXECUTED" and oracle["fpga_executed"] is False
    )
    assert oracle["encoder_layers"] == build.get("encoder_layers", 12)
    assert oracle["params_sha256"] == contract["params_sha256"]
    assert oracle["input_sha256"] == contract["input_sha256"]
    for name, digest in oracle["source_hashes"].items():
        source = host / oracle["frozen_source_files"][name]
        assert sha(source) == digest, "Changed local emitted source: " + name
    assert (
        sha(launcher)
        == oracle["source_hashes"][
            next(
                name
                for name in oracle["source_hashes"]
                if name.endswith("/model_launcher.c")
            )
        ]
    )
    graph = "examples/BOSCAMEFPGA/gte-small/model_runtime/model.c"
    assert build["sources"][graph] == oracle["model_graph_sha256"]
    assert set(oracle["kernel_hashes"]) == set(build["compiled_kernels"])
    for name, hashes in oracle["kernel_hashes"].items():
        original = json.loads((kernels / name / "build-result.json").read_text())
        for filename in ("kernel.ll", "kernel.o"):
            assert (
                hashes[filename]
                == original["artifacts"][filename]
                == sha(kernels / name / filename)
            )
        assert hashes["kernel.o"] == build["compiled_kernels"][name]["object_sha256"]
        assert hashes["adapter.c"] == build["compiled_kernels"][name]["adapter_sha256"]
        assert sha(kernels / name / "adapter.c") == hashes["adapter.c"]
    host_files = {item["name"]: item for item in oracle["files"]}
    assert set(host_files) == set(build["outputs"]), (
        "Emitted output set differs from target build"
    )
    for name, item in host_files.items():
        path = host / (name + ".bin")
        assert path.stat().st_size == item["bytes"] == build["outputs"][name]["nbytes"]
        assert sha(path) == item["sha256"], "Changed emitted BIN: " + name
    return oracle


def validate(board, host, kernels, output=None):
    destination = output or board / "emitted-validation.json"
    if destination.exists():
        raise FileExistsError(
            "Preserve existing evidence; choose a new --output: " + str(destination)
        )
    record = json.loads((board / "result.json").read_text())
    contract = json.loads((board / "model-input.json").read_text())
    build = json.loads((board / "build-report.json").read_text())
    oracle = check_reference(host, kernels, build, contract, board / "model_launcher.c")
    assert record["state"] == "completed" and record["execution_status"] == "exited"
    assert record["app_exit_code"] == record["exit_code"] == 0
    assert record["operator_calls"] == build["expected_calls"] == 1649
    assert (
        record["image_sha256"]
        == build["image_sha256"]
        == sha(board / "gte-small-full.bin")
    )
    for name, digest in record["frozen_files"].items():
        assert sha(board / name) == digest, "Changed frozen evidence: " + name
    uart = (board / "uart.log").read_text()
    assert "@@FPGA_START " in uart
    declarations = {
        item["name"]: item
        for item in (
            json.loads(match)
            for match in re.findall(r"@@FPGA_OUTPUT (\{[^\r\n]+\})", uart)
        )
    }
    listing = json.loads((board / "artifacts-list.stdout").read_text())
    assert listing["id"] == record["task_id"]
    artifacts = listing["artifacts"]
    host_files = {item["name"]: item for item in oracle["files"]}
    assert len(artifacts) == len(declarations) == len(host_files)
    assert {item["name"] for item in artifacts} == set(declarations) == set(host_files)
    comparisons, arrays = [], {}
    for artifact in artifacts:
        name = artifact["name"]
        declaration = declarations[name]
        for field in ("dtype", "shape", "nbytes"):
            assert declaration[field] == artifact[field]
        assert int(declaration["address"], 0) == int(str(artifact["address"]), 0)
        assert artifact["dtype"] == "<f4"
        assert artifact["shape"] == build["outputs"][name]["shape"]
        actual_path = board / "artifacts" / (name + ".bin")
        expected_path = host / (name + ".bin")
        assert sha(actual_path) == artifact["sha256"]
        assert actual_path.stat().st_size == artifact["nbytes"]
        assert (
            expected_path.stat().st_size
            == host_files[name]["bytes"]
            == artifact["nbytes"]
        )
        actual = np.fromfile(actual_path, dtype="<f4").reshape(artifact["shape"])
        expected = np.fromfile(expected_path, dtype="<f4").reshape(artifact["shape"])
        differences = np.argwhere(actual.view("<u4") != expected.view("<u4"))
        comparisons.append(
            {
                "name": name,
                "shape": artifact["shape"],
                "bytes": artifact["nbytes"],
                "board_sha256": artifact["sha256"],
                "host_sha256": host_files[name]["sha256"],
                "bitwise_equal": len(differences) == 0,
                "different_elements": len(differences),
                "first_differences": [
                    {
                        "index": i.tolist(),
                        "actual_bits": f"0x{int(actual.view('<u4')[tuple(i)]):08x}",
                        "expected_bits": f"0x{int(expected.view('<u4')[tuple(i)]):08x}",
                    }
                    for i in differences[:8]
                ],
                **metrics(actual, expected),
            }
        )
        arrays[name] = actual
    stages = []
    if "layer_checkpoints" in arrays:
        expected = np.fromfile(host / "layer_checkpoints.bin", dtype="<f4").reshape(
            13, 128, 384
        )
        for stage, (actual, target) in enumerate(
            zip(arrays["layer_checkpoints"], expected)
        ):
            stages.append(
                {
                    "stage": stage,
                    "kind": "embedding" if stage == 0 else "encoder_layer",
                    "bitwise_equal": actual.tobytes() == target.tobytes(),
                    **metrics(actual, target),
                }
            )
    passed = all(item["bitwise_equal"] for item in comparisons)
    report = {
        "status": "LOCAL_EMITTED_MODEL_PASS" if passed else "LOCAL_EMITTED_MODEL_FAIL",
        "task_id": record["task_id"],
        "server": record["server"],
        "fpga": record["fpga"],
        "evidence": "actual scheduler-downloaded FPGA BINs; all comparison performed locally",
        "primary_reference": oracle["scope"],
        "criterion": "every returned tensor byte-for-byte identical; no tolerance",
        "operator_calls": record["operator_calls"],
        "encoder_layers": oracle["encoder_layers"],
        "kernel_artifact_pairs_verified": len(oracle["kernel_hashes"]),
        "params_sha256": contract["params_sha256"],
        "input_sha256": contract["input_sha256"],
        "comparisons": comparisons,
        "stages": stages,
        "oracle_manifest_sha256": sha(host / "manifest.json"),
        "validator_sha256": sha(Path(__file__).resolve()),
        "board_oracle_executed": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument(
        "--host", type=Path, default=ROOT / "build/full-model-host-emitted"
    )
    parser.add_argument(
        "--kernels", type=Path, default=ROOT / "build/scheduler-output/kernels"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new report path; never overwrites an existing report",
    )
    args = parser.parse_args()
    return validate(
        args.board.resolve(),
        args.host.resolve(),
        args.kernels.resolve(),
        args.output.resolve() if args.output else None,
    )


if __name__ == "__main__":
    sys.exit(main())

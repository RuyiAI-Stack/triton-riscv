#!/usr/bin/env python3
"""Compare complete scheduler-returned E5 outputs with frozen emitted-code output."""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OUTPUTS = {"sentence_embedding", "last_hidden", "layer_hidden"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def freeze_reference(reference, run, destination):
    """Bind an existing local execution to this image before copying any oracle."""
    manifest = reference / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            "missing emitted reference; run build_model_host.py first: "
            + str(reference)
        )
    metadata = json.loads(manifest.read_text())
    build = json.loads((run / "build.json").read_text())
    params = json.loads((run / "params.json").read_text())
    if (
        metadata.get("status") != "LOCAL_GRAPH_EXECUTED"
        or metadata.get("fpga_executed") is not False
    ):
        raise ValueError("reference is not an executed local emitted-code graph")
    if metadata.get("params_sha256") != params["sha256"] or metadata.get(
        "inputs_sha256"
    ) != sha(run / "inputs.bin"):
        raise ValueError("emitted reference checkpoint/input mismatch")
    files = {item["name"]: item for item in metadata["files"]}
    if set(files) != OUTPUTS or len(files) != len(metadata["files"]):
        raise ValueError("emitted reference must contain all three complete outputs")
    objects = {
        str(Path(item["path"]).resolve()): item["sha256"] for item in build["objects"]
    }
    source_hashes = metadata.get("source_hashes", {})
    if not source_hashes:
        raise ValueError("emitted reference lacks source provenance")
    pairs = {}
    proof_files = {}
    for name, expected in source_hashes.items():
        source = Path(name)
        if not source.is_file() or sha(source) != expected:
            raise ValueError("emitted source changed after execution: " + name)
        if source.is_relative_to(REPO):
            recorded = build["sources"].get(str(source.relative_to(REPO)))
            if recorded is not None and recorded != expected:
                raise ValueError("emitted and target source differ: " + name)
        if source.name == "kernel.ll":
            artifacts = json.loads((source.parent / "build-result.json").read_text())[
                "artifacts"
            ]
            target = source.with_name("kernel.o")
            if (
                artifacts["kernel.ll"] != expected
                or artifacts["kernel.o"] != objects.get(str(target.resolve()))
                or sha(target) != artifacts["kernel.o"]
            ):
                raise ValueError(
                    "emitted scalar LLVM and target object differ: " + name
                )
            pairs[source.parent.name] = {
                "llvm_sha256": expected,
                "target_object_sha256": artifacts["kernel.o"],
                "build_result_sha256": sha(source.parent / "build-result.json"),
            }
            for proof in (target, source.parent / "build-result.json"):
                proof_files[proof] = sha(proof)
    expected_cases = {
        name
        for name in build["real_kernel_objects"]
        if not name.startswith("matmul_i8_")
    }
    if set(pairs) != expected_cases:
        raise ValueError(
            "emitted reference must verify every non-matrix scalar LLVM/target object pair"
        )
    destination.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for name, item in files.items():
        source = reference / (name + ".bin")
        if source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]:
            raise ValueError("emitted output length/SHA mismatch: " + name)
        target = destination / source.name
        shutil.copyfile(source, target)
        frozen[target.name] = sha(target)
    shutil.copyfile(manifest, destination / "manifest.json")
    frozen["manifest.json"] = sha(destination / "manifest.json")
    for source, expected in {
        **{Path(name): value for name, value in source_hashes.items()},
        **proof_files,
    }.items():
        relative = (
            source.relative_to(REPO)
            if source.is_relative_to(REPO)
            else Path("external") / (expected + "-" + source.name)
        )
        target = destination / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        frozen[str(target.relative_to(destination))] = expected
    receipt = {
        "status": "EMITTED_REFERENCE_FROZEN",
        "origin": str(reference),
        "params_sha256": metadata["params_sha256"],
        "inputs_sha256": metadata["inputs_sha256"],
        "scalar_llvm_target_pairs": len(pairs),
        "pair_evidence": pairs,
        "files": frozen,
    }
    save(destination / "frozen.json", receipt)
    return receipt


def validate(run, reference, output):
    """Validate transport/provenance, then use bytes as the only numeric criterion."""
    record = json.loads((run / "result.json").read_text())
    build = json.loads((run / "build.json").read_text())
    required = (
        "e5_model.bin",
        "e5_model.elf",
        "elf-audit.json",
        "build.json",
        "input_reference.json",
        "params.json",
        "inputs.bin",
        "operator_trace.json",
    )
    for name in required:
        if sha(run / name) != record["frozen_files"].get(name):
            raise ValueError("frozen target evidence SHA mismatch: " + name)
    audit = json.loads((run / "elf-audit.json").read_text())
    if (
        sha(run / "e5_model.bin") != record["image_sha256"]
        or build["image_sha256"] != record["image_sha256"]
        or audit["status"] != "PASS"
        or audit["elf_sha256"] != sha(run / "e5_model.elf")
    ):
        raise ValueError("target image or ELF audit provenance mismatch")
    task = json.loads((run / "task.json").read_text())
    result = task.get("result") or {}
    if (
        task["id"] != record["task_id"]
        or task["state"] != "completed"
        or result.get("execution_status") != "exited"
        or result.get("app_exit_code") != 0
        or result.get("exit_code") != 0
        or result.get("details", {}).get("image_sha256") != record["image_sha256"]
        or result.get("details", {}).get("ddr_readback_matches") is not True
    ):
        raise ValueError("scheduler task did not complete successfully with this image")
    receipt = json.loads((reference / "frozen.json").read_text())
    if (
        receipt.get("status") != "EMITTED_REFERENCE_FROZEN"
        or receipt["params_sha256"]
        != json.loads((run / "params.json").read_text())["sha256"]
        or receipt["inputs_sha256"] != sha(run / "inputs.bin")
    ):
        raise ValueError("frozen emitted reference input/checkpoint mismatch")
    for name, expected in receipt["files"].items():
        if sha(reference / name) != expected:
            raise ValueError("frozen emitted reference changed: " + name)
    metadata = json.loads((reference / "manifest.json").read_text())
    expected = {item["name"]: item for item in build["outputs"]}
    manifest = json.loads((run / "artifacts/manifest.json").read_text())
    actual = {item["name"]: item for item in manifest["artifacts"]}
    scheduler = {item["name"]: item for item in result["artifacts"]}
    local = {item["name"]: item for item in metadata["files"]}
    if (
        manifest["id"] != task["id"]
        or set(actual) != OUTPUTS
        or set(expected) != OUTPUTS
        or set(local) != OUTPUTS
    ):
        raise ValueError("incomplete complete-model output set")
    if actual != scheduler or len(actual) != len(manifest["artifacts"]):
        raise ValueError("scheduler task/listed artifact metadata mismatch")
    comparisons = []
    for name in sorted(OUTPUTS):
        item, contract = actual[name], expected[name]
        if any(item[key] != contract[key] for key in ("shape", "dtype", "nbytes")):
            raise ValueError("downloaded output contract mismatch: " + name)
        path = run / "artifacts" / (name + ".bin")
        target = reference / (name + ".bin")
        if path.stat().st_size != item["nbytes"] or sha(path) != item["sha256"]:
            raise ValueError("downloaded BIN length/SHA mismatch: " + name)
        if (
            target.stat().st_size != item["nbytes"]
            or sha(target) != local[name]["sha256"]
        ):
            raise ValueError("emitted BIN length/SHA mismatch: " + name)
        got, want = path.read_bytes(), target.read_bytes()
        different = []
        count = 0
        for i, (a, b) in enumerate(
            zip(struct.iter_unpack("<I", got), struct.iter_unpack("<I", want))
        ):
            if a != b:
                count += 1
                if len(different) < 8:
                    different.append(
                        {
                            "flat_index": i,
                            "actual_bits": f"0x{a[0]:08x}",
                            "expected_bits": f"0x{b[0]:08x}",
                        }
                    )
        comparisons.append(
            {
                "output": name,
                "shape": contract["shape"],
                "elements": len(got) // 4,
                "bitwise_equal": got == want,
                "different_elements": count,
                "download_sha256": item["sha256"],
                "reference_sha256": local[name]["sha256"],
                "samples": different,
            }
        )
    passed = all(item["bitwise_equal"] for item in comparisons)
    report = {
        "status": "MODEL_NUMERICAL_PASS" if passed else "MODEL_NUMERICAL_FAIL",
        "task_id": task["id"],
        "fpga": task["fpga"],
        "image_sha256": record["image_sha256"],
        "validation": "actual scheduler BINs versus frozen complete local emitted-code outputs",
        "acceptance": "all three complete outputs byte for byte identical; no tolerance",
        "board_math_mocked": False,
        "emitted_implementation_agreement": passed,
        "comparisons": comparisons,
        "reference_receipt_sha256": sha(reference / "frozen.json"),
        "verifier_sha256": sha(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    save(output / "local-validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "build/model-host")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new review directory; original evidence stays read-only",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new review directory")
    args.output.mkdir(parents=True)
    freeze_reference(
        args.reference.resolve(), args.run.resolve(), args.output / "emitted-reference"
    )
    report = validate(
        args.run.resolve(), args.output / "emitted-reference", args.output
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "task_id": report["task_id"],
                "emitted_implementation_agreement": report[
                    "emitted_implementation_agreement"
                ],
            }
        )
    )
    return report["status"] != "MODEL_NUMERICAL_PASS"


if __name__ == "__main__":
    raise SystemExit(main())

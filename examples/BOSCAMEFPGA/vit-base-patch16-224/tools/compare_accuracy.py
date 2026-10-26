#!/usr/bin/env python3
"""Prepare identical ViT inputs and compare FP32/emitted-W8A8 logits.

No candidate arithmetic is implemented here. Candidate logits must come from
the complete emitted model and carry its execution evidence. Missing model,
dataset or execution evidence produces SKIP, never an accuracy PASS.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from fetch_assets import EXPECTED_SHA256, MODEL_ID, REVISION, digest_file

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = (
    "model_id",
    "revision",
    "checkpoint_sha256",
    "inputs_sha256",
    "preprocessor_sha256",
    "sample_ids",
)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def checked_asset(assets, name):
    path = assets / name
    if digest_file(path) != EXPECTED_SHA256[name]:
        raise ValueError("pinned asset hash mismatch: " + name)
    return path


def prepare(args):
    from PIL import Image
    from transformers import ViTImageProcessor

    processor_path = checked_asset(args.assets, "preprocessor_config.json")
    processor = ViTImageProcessor.from_pretrained(
        str(args.assets), local_files_only=True
    )
    data = json.loads(args.samples.read_text())
    records = data["samples"]
    if not records or len({x["id"] for x in records}) != len(records):
        raise ValueError("sample manifest needs unique, nonempty sample IDs")
    images, labels, source_hashes = [], [], []
    for record in records:
        path = args.samples.parent / record["path"]
        with Image.open(path) as image:
            pixels = processor(
                images=image.convert("RGB"), return_tensors="np"
            )["pixel_values"]
        if pixels.shape != (1, 3, 224, 224):
            raise ValueError("processor did not produce the fixed model shape")
        images.append(pixels[0].astype(np.float32))
        label = record.get("label", -1)
        if not isinstance(label, int) or not -1 <= label < 1000:
            raise ValueError(
                "labels must be zero-based class indices or absent"
            )
        labels.append(label)
        source_hashes.append(digest_file(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pixel_values=np.stack(images),
        labels=np.array(labels, dtype=np.int64),
    )
    result = {
        "schema": 1,
        "status": "PREPARED",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_sha256": EXPECTED_SHA256["model.safetensors"],
        "inputs_sha256": digest_file(args.output),
        "preprocessor_sha256": digest_file(processor_path),
        "sample_ids": [x["id"] for x in records],
        "source_image_sha256": source_hashes,
        "preprocessing": processor.to_dict(),
        "sample_manifest_sha256": digest_file(args.samples),
        "candidate_executed": False,
    }
    save_json(args.output.with_suffix(".json"), result)
    return result


def load_inputs(path):
    identity = json.loads(path.with_suffix(".json").read_text())
    if identity["inputs_sha256"] != digest_file(path):
        raise ValueError("prepared input hash mismatch")
    if (
        identity["model_id"] != MODEL_ID
        or identity["revision"] != REVISION
        or identity["checkpoint_sha256"] != EXPECTED_SHA256["model.safetensors"]
        or identity["preprocessor_sha256"]
        != EXPECTED_SHA256["preprocessor_config.json"]
    ):
        raise ValueError(
            "prepared inputs do not describe the pinned model/preprocessor"
        )
    with np.load(path, allow_pickle=False) as data:
        pixels, labels = data["pixel_values"].copy(), data["labels"].copy()
    if (
        pixels.shape != (len(identity["sample_ids"]), 3, 224, 224)
        or pixels.dtype != np.float32
        or labels.shape != (len(pixels),)
        or not np.all(np.isfinite(pixels))
    ):
        raise ValueError("invalid prepared input arrays")
    return identity, pixels, labels


def reference(args):
    import torch
    from import_model import load_model

    identity, pixels, _ = load_inputs(args.inputs)
    checkpoint = checked_asset(args.assets, "model.safetensors")
    config = checked_asset(args.assets, "config.json")
    torch.set_num_threads(1)
    model, _, _ = load_model(config, checkpoint, torch.float32)
    with torch.no_grad():
        logits = np.concatenate(
            [model(torch.from_numpy(x[None])).cpu().numpy() for x in pixels]
        )
    np.savez(args.output, logits=logits.astype(np.float32))
    result = {
        **{key: identity[key] for key in IDENTITY},
        "schema": 1,
        "status": "REFERENCE_ONLY",
        "execution": "huggingface-fp32-eager",
        "patch_projection": "float32",
        "dense_linear": "float32",
        "logits_sha256": digest_file(args.output),
        "torch_version": torch.__version__,
        "candidate_executed": False,
    }
    save_json(args.output.with_suffix(".json"), result)
    return result


def metrics(reference_logits, candidate_logits, labels):
    if (
        reference_logits.shape != candidate_logits.shape
        or reference_logits.ndim != 2
        or reference_logits.shape[1] != 1000
        or len(labels) != len(reference_logits)
        or not np.all(np.isfinite(reference_logits))
        or not np.all(np.isfinite(candidate_logits))
    ):
        raise ValueError(
            "logits must be finite arrays of the same [samples,1000] shape"
        )
    difference = candidate_logits.astype(np.float64) - reference_logits.astype(
        np.float64
    )
    ref_rank = np.argsort(-reference_logits, axis=1, kind="stable")
    got_rank = np.argsort(-candidate_logits, axis=1, kind="stable")
    labeled = labels >= 0
    result = {
        "samples": len(labels),
        "labeled_samples": int(np.count_nonzero(labeled)),
        "max_abs_logit_error": float(np.max(np.abs(difference), initial=0)),
        "mean_abs_logit_error": float(np.mean(np.abs(difference))),
        "logit_rmse": float(np.sqrt(np.mean(difference**2))),
        "prediction_agreement": float(
            np.mean(ref_rank[:, 0] == got_rank[:, 0])
        ),
        "top1": None,
        "top5": None,
    }
    if np.any(labeled):
        for topk in (1, 5):
            ref = np.mean(
                np.any(
                    ref_rank[labeled, :topk] == labels[labeled, None], axis=1
                )
            )
            got = np.mean(
                np.any(
                    got_rank[labeled, :topk] == labels[labeled, None], axis=1
                )
            )
            result[f"top{topk}"] = {
                "reference": float(ref),
                "w8a8": float(got),
                "change_percentage_points": float((got - ref) * 100),
            }
    return result


def compare(args):
    identity, _, labels = load_inputs(args.inputs)
    records, arrays = [], []
    for path in (args.reference, args.candidate):
        record = json.loads(path.with_suffix(".json").read_text())
        if any(record.get(key) != identity[key] for key in IDENTITY):
            raise ValueError(
                "logits use a different checkpoint, input set, or preprocessing"
            )
        if record.get("logits_sha256") != digest_file(path):
            raise ValueError("logits hash differs from execution record")
        with np.load(path, allow_pickle=False) as data:
            arrays.append(data["logits"].copy())
        records.append(record)
    if records[0].get("execution") != "huggingface-fp32-eager":
        raise ValueError(
            "reference must be the explicit FP32 eager model, including patch projection"
        )
    candidate = records[1]
    if (
        candidate.get("execution")
        not in ("boscame-emitted-simulator", "boscame-hardware")
        or candidate.get("full_model_executed") is not True
        or candidate.get("all_numerical_kernels_emitted") is not True
        or not candidate.get("module_sha256")
        or not candidate.get("execution_report_sha256")
    ):
        raise ValueError(
            "candidate lacks complete emitted-model execution evidence; host checks cannot substitute"
        )
    return {
        "schema": 1,
        "status": "MEASURED",
        "accuracy_accepted": False,
        "scope": "same pinned checkpoint, preprocessed samples and class labels",
        **{key: identity[key] for key in IDENTITY},
        "candidate_execution": candidate["execution"],
        "reference_logits_sha256": records[0]["logits_sha256"],
        "candidate_logits_sha256": candidate["logits_sha256"],
        "metrics": metrics(arrays[0], arrays[1], labels),
    }


def self_test(_):
    reference_logits = np.zeros((2, 1000), np.float32)
    reference_logits[0, 5] = reference_logits[1, 9] = 2
    candidate = reference_logits.copy()
    candidate[0, 3] = 3
    result = metrics(reference_logits, candidate, np.array([5, 9]))
    assert result["prediction_agreement"] == 0.5
    assert result["top1"]["change_percentage_points"] == -50
    assert result["top5"]["w8a8"] == 1
    assert (
        metrics(reference_logits, candidate, np.array([-1, -1]))["top1"] is None
    )
    with tempfile.TemporaryDirectory(
        prefix="vit-accuracy-protocol-"
    ) as directory:
        work = Path(directory)
        inputs = work / "inputs.npz"
        np.savez(
            inputs,
            pixel_values=np.zeros((2, 3, 224, 224), np.float32),
            labels=np.array([5, 9], dtype=np.int64),
        )
        identity = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "checkpoint_sha256": EXPECTED_SHA256["model.safetensors"],
            "preprocessor_sha256": EXPECTED_SHA256["preprocessor_config.json"],
            "inputs_sha256": digest_file(inputs),
            "sample_ids": ["fixture-a", "fixture-b"],
        }
        save_json(inputs.with_suffix(".json"), identity)
        reference_path, candidate_path = (
            work / "reference.npz",
            work / "candidate.npz",
        )
        np.savez(reference_path, logits=reference_logits)
        np.savez(candidate_path, logits=candidate)
        save_json(
            reference_path.with_suffix(".json"),
            {
                **identity,
                "execution": "huggingface-fp32-eager",
                "logits_sha256": digest_file(reference_path),
            },
        )
        invocation = argparse.Namespace(
            inputs=inputs, reference=reference_path, candidate=candidate_path
        )
        for invalid in (
            {"execution": "host-execution-engine"},
            {
                "execution": "boscame-emitted-simulator",
                "full_model_executed": False,
            },
            {
                "execution": "boscame-emitted-simulator",
                "sample_ids": ["wrong", "order"],
            },
        ):
            save_json(
                candidate_path.with_suffix(".json"),
                {
                    **identity,
                    "logits_sha256": digest_file(candidate_path),
                    **invalid,
                },
            )
            try:
                compare(invocation)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "invalid candidate execution evidence was accepted"
                )
    return {
        "status": "SELF_TEST_PASS",
        "full_model_executed": False,
        "accuracy_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "reference", "compare", "self-test"):
        command = commands.add_parser(name)
        command.set_defaults(
            run={
                "prepare": prepare,
                "reference": reference,
                "compare": compare,
                "self-test": self_test,
            }[name]
        )
        command.add_argument("--report", type=Path)
        if name in ("prepare", "reference"):
            command.add_argument("--assets", type=Path, default=ROOT / "assets")
            command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--samples", type=Path, required=True)
        if name in ("reference", "compare"):
            command.add_argument("--inputs", type=Path, required=True)
        if name == "compare":
            command.add_argument("--reference", type=Path, required=True)
            command.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
        result = args.run(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        result = {
            "schema": 1,
            "status": "SKIP",
            "accuracy_accepted": False,
            "reason": str(error),
            "full_model_executed": False,
        }
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "SKIP" else 0


if __name__ == "__main__":
    raise SystemExit(main())

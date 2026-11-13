#!/usr/bin/env python3
"""Build one existing W8A8 convolution with a frozen real graph-node fixture."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
BASE = MODEL / "build/model-integration"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, path):
    with path.open("w") as stream:
        subprocess.run(
            [str(v) for v in command],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--case", default="conv2d_144x144_40x24_k1_s1")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    original = BASE / "physical-plan"
    plan = json.loads((original / "physical-plan.json").read_text())
    call = copy.deepcopy(next(v for v in plan["calls"] if v["target"] == a.case))
    fixture = np.load(a.fixture)
    input_tensor = copy.deepcopy(plan["tensors"][call["reads"][0]])
    input_tensor["storage"] = {"kind": "external_input"}
    input_tensor["external"] = True
    plan["tensors"]["input"] = input_tensor
    call["index"] = 0
    call["arguments"][0]["value"] = "input"
    call["reads"] = ["input"]
    output_value = call["writes"][0]
    output_tensor = plan["tensors"][output_value]
    plan.update(
        calls=[call],
        physical_calls=1,
        logical_nodes=1,
        connectors=[],
        unique_connector_specializations=0,
        unique_existing_cases=1,
        target_counts={"existing_case": 1},
        input={"value": "input", "shape": input_tensor["shape"], "dtype": "f32"},
        output={"value": output_value, "shape": output_tensor["shape"], "dtype": "f32"},
        mode="bounded_connector_probe",
        full_model=False,
    )
    # Preserve VIT_WORKSPACE_BYTES from the full model so recompiling the exact
    # adapter source does not change its compiled scratch-offset constant.
    (a.output / "physical-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    original_meta = json.loads((original / "parameters.json").read_text())
    original_blob = (original / "parameters.bin").read_bytes()
    entries = {v["name"]: v for v in original_meta["entries"]}
    metadata = {k: v for k, v in original_meta.items() if k != "entries"}
    metadata["entries"] = []
    blob = bytearray()
    for argument, fixture_key in zip(call["arguments"][1:3], ("weight", "bias")):
        source = entries[argument["value"]]
        data = np.asarray(fixture[fixture_key], dtype="<f4").tobytes()
        assert list(fixture[fixture_key].shape) == argument["shape"]
        assert (
            data == original_blob[source["offset"] : source["offset"] + source["bytes"]]
        )
        offset = (len(blob) + 63) & ~63
        blob.extend(b"\0" * (offset - len(blob)))
        blob.extend(data)
        entry = copy.deepcopy(source)
        entry["offset"] = offset
        metadata["entries"].append(entry)
    (a.output / "parameters.bin").write_bytes(blob)
    metadata.update(bytes=len(blob), sha256=sha(a.output / "parameters.bin"))
    (a.output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    x = np.asarray(fixture["input_0"], dtype="<f4")
    expected = np.asarray(fixture["output"], dtype="<f4")
    assert list(x.shape) == input_tensor["shape"]
    assert list(expected.shape) == output_tensor["shape"]
    x.tofile(a.output / "input.f32")
    expected.tofile(a.output / "w8a8-logits.f32")
    expected.tofile(a.output / "original-logits.f32")
    kernel_dir = MODEL / "build/kernels" / a.case
    provenance = {
        "status": "FROZEN_REAL_NODE_FIXTURE",
        "scope": "one standalone full-shape existing composite convolution; not whole-model proof",
        "case": a.case,
        "logical_node": call["logical_node"],
        "fixture": str(a.fixture.resolve()),
        "fixture_sha256": sha(a.fixture),
        "input_sha256": sha(a.output / "input.f32"),
        "expected_sha256": sha(a.output / "w8a8-logits.f32"),
        "elements": int(expected.size),
        "kernel_o_sha256": sha(kernel_dir / "kernel.o"),
        "adapter_source_sha256": sha(kernel_dir / "adapter.c"),
        "full_model_workspace_bytes_preserved": plan["workspace"][
            "provisional_total_bytes"
        ],
        "builder_sha256": sha(__file__),
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
            "--host-numerical-validation",
            "--kernels",
            MODEL / "build/kernels",
            "--conv-overlay",
            MODEL / "build/kernels",
        ],
        a.output / "build-image.log",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

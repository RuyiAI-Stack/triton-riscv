#!/usr/bin/env python3
"""Prepare a nonzero first-block prefix probe without claiming whole-model execution."""

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "build/model-integration"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calls", type=int, default=8)
    p.add_argument("--source-plan", type=Path, default=BASE / "physical-plan")
    p.add_argument("--output", type=Path, default=BASE / "prefix8")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = json.loads((a.source_plan / "physical-plan.json").read_text())
    graph = json.loads((BASE / "shape-graph-manifest.json").read_text())
    if not 1 <= a.calls <= len(d["calls"]):
        raise ValueError("invalid prefix length")
    calls = copy.deepcopy(d["calls"][: a.calls])
    out = calls[-1]["writes"]
    if len(out) != 1 or d["tensors"][out[0]]["dtype"] != "f32":
        raise ValueError("prefix must end in one f32 tensor")
    value = out[0]
    count = math.prod(d["tensors"][value]["shape"])
    if count > 4000000:
        raise ValueError("prefix output too large")
    # The CPU golden executes only semantic graph nodes completed by the prefix.
    final_node = calls[-1]["logical_node"]
    stop = next(i for i, n in enumerate(graph["nodes"]) if n["name"] == final_node)
    if graph["nodes"][stop]["output"] != value:
        raise ValueError("prefix ends within a composite logical node")
    partial = copy.deepcopy(graph)
    partial["nodes"] = partial["nodes"][: stop + 1]
    partial["output"] = {
        "value": value,
        "shape": d["tensors"][value]["shape"],
        "dtype": "f32",
    }
    spec = importlib.util.spec_from_file_location(
        "integration_reference", HERE / "reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.torch.set_num_threads(4)
    model, _ = mod.load_model(BASE / "assets")
    image = mod.torch.from_numpy(
        np.fromfile(BASE / "reference/input.f32", dtype="<f4")
        .reshape(1, 3, 288, 288)
        .copy()
    )
    parameters = mod.folded_parameters(model, graph)
    with mod.torch.inference_mode():
        expected, trace = mod.interpret(model, partial, image, parameters, True)
    expected.numpy().astype("<f4").tofile(a.output / "expected.f32")
    checkpoint_files = {}
    if a.calls >= 8:

        def capture(name):
            sub = copy.deepcopy(graph)
            i = next(i for i, n in enumerate(sub["nodes"]) if n["name"] == name)
            sub["nodes"] = sub["nodes"][: i + 1]
            sub["output"] = {"value": sub["nodes"][-1]["output"]}
            with mod.torch.inference_mode():
                return mod.interpret(model, sub, image, parameters, True)[0]

        for filename, node in (
            ("stem.f32", "conv_stem"),
            ("depthwise.f32", "blocks.0.0.conv_dw"),
        ):
            capture(node).numpy().astype("<f4").tofile(a.output / filename)
            checkpoint_files[filename] = {
                "node": node,
                "sha256": sha(a.output / filename),
            }
        pooled = capture("blocks.0.0.se.pool").reshape(1, 40)
        qx, _ = mod.quantize_rows(pooled)
        acc = mod.integer_mm(qx, parameters["blocks.0.0.se.conv_reduce"]["qweight"].T)
        acc.numpy().astype("<i4").tofile(a.output / "pointwise-acc.i32")
        checkpoint_files["pointwise-acc.i32"] = {
            "node": "blocks.0.0.se.conv_reduce integer GEMM",
            "sha256": sha(a.output / "pointwise-acc.i32"),
        }
    d["calls"] = calls
    d["physical_calls"] = len(calls)
    d["logical_nodes"] = len(partial["nodes"])
    d["output"] = partial["output"]
    used = {c["target"] for c in calls if c["target_kind"] == "connector"}
    d["connectors"] = [s for s in d["connectors"] if s["name"] in used]
    d["unique_connector_specializations"] = len(d["connectors"])
    d["unique_existing_cases"] = len(
        {c["target"] for c in calls if c["target_kind"] == "existing_case"}
    )
    d["target_counts"] = dict(Counter(c["target_kind"] for c in calls))
    d["mode"] = "real_weight_nonzero_prefix_probe"
    d["full_model"] = False
    # Full model allocation bounds are retained deliberately; the cut does not
    # replan storage, avoiding any new aliasing assumptions for probe calls.
    (a.output / "physical-plan.json").write_text(json.dumps(d, indent=2) + chr(10))
    for name in ("parameters.bin", "parameters.json", "input.f32"):
        shutil.copyfile(a.source_plan / name, a.output / name)
    # Image builder embeds this intentionally prefix-only golden name.
    shutil.copyfile(a.output / "expected.f32", a.output / "w8a8-logits.f32")
    manifest = {
        "mode": d["mode"],
        "full_model": False,
        "calls": len(calls),
        "logical_nodes": len(partial["nodes"]),
        "output": partial["output"],
        "output_elements": count,
        "expected_sha256": sha(a.output / "expected.f32"),
        "input_sha256": sha(a.output / "input.f32"),
        "parameter_sha256": sha(a.output / "parameters.bin"),
        "checkpoint_sha256": mod.EXPECTED_CHECKPOINT_SHA256,
        "trace": trace,
    }
    manifest["checkpoint_files"] = checkpoint_files
    (a.output / "prefix-reference.json").write_text(
        json.dumps(manifest, indent=2) + chr(10)
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "generate_runner.py"),
            "--plan-dir",
            str(a.output),
            "--output",
            str(a.output / "runner"),
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("mode", "calls", "logical_nodes", "output", "expected_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

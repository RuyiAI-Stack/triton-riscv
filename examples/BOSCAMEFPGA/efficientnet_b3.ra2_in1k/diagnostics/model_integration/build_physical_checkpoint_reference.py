#!/usr/bin/env python3
"""Build CPU references for selected quantize/INT8 MMA physical output spans.

Consumes actual logical inputs dumped by check_gap_torch_model_reference.py.
This diagnostic writes an independent oracle; it never changes model goldens.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference  # noqa: E402
import torch  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv(raw):
    value = 0xCBF29CE484222325
    for byte in raw:
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument(
        "--physical-plan",
        type=Path,
        default=ROOT / "build/model-integration/physical-plan/physical-plan.json",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "build/model-integration/shape-graph-manifest.json",
    )
    parser.add_argument("--call", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    replay_trace = args.replay_dir / "whole-graph-node-trace.json"
    if sha(replay_trace) != sha(args.reference_trace):
        raise ValueError(
            "dumped-input replay differs from the current logical reference trace"
        )
    replay_report = json.loads(
        (args.replay_dir / "whole-graph-report.json").read_text()
    )
    if (
        replay_report["output_bit_mismatches"] != 0
        or not replay_report["all_27_actual_gaps_bitexact_torch"]
    ):
        raise ValueError(
            "replay must reproduce the original primary logits and Torch GAP exactly"
        )
    plan = json.loads(args.physical_plan.read_text())
    graph = json.loads(args.graph.read_text())
    nodes = {node["name"]: node for node in graph["nodes"]}
    trace = {item["node"]: item for item in json.loads(replay_trace.read_text())}
    writers = {node["output"]: node["name"] for node in graph["nodes"]}
    parameter_manifest = args.physical_plan.parent / "parameters.json"
    parameter_blob = args.physical_plan.parent / "parameters.bin"
    parameters = json.loads(parameter_manifest.read_text())
    if sha(parameter_blob) != parameters["sha256"]:
        raise ValueError("primary plan parameter blob hash mismatch")
    parameter_entries = {item["name"]: item for item in parameters["entries"]}
    parameter_bytes = parameter_blob.read_bytes()
    dumps = {
        json.loads(path.read_text())["node"]: path.with_suffix(".npz")
        for path in (args.replay_dir / "node-values").glob("*.json")
    }
    calls = args.call if args.call is not None else [95, 191, 287, 319, 351, 415]
    outputs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_dir = args.output.parent / "expected-physical-spans"
    expected_dir.mkdir(exist_ok=True)
    dtype_map = {"i8": np.dtype("i1"), "i32": np.dtype("<i4"), "f32": np.dtype("<f4")}
    for index in calls:
        call = plan["calls"][index]
        if call["index"] != index:
            raise ValueError("plan calls are not indexed densely")
        name = call["logical_node"]
        node = nodes[name]
        if node["family"] != "conv2d_i8" or node["weight_shape"][2:] != [1, 1]:
            raise ValueError(
                "only preserved INT8 pointwise physical calls are supported"
            )
        with np.load(dumps[name], allow_pickle=False) as dump:
            source = np.ascontiguousarray(dump["input_0"], dtype=np.float32)
            qweight = np.ascontiguousarray(dump["qweight"], dtype=np.int8)
        source_hash = hashlib.sha256(source.tobytes()).hexdigest()
        input_node = writers[node["inputs"][0]]
        if source_hash != trace[input_node]["output_sha256"]:
            raise ValueError(
                name + ": dumped input does not match the actual CPU trace value"
            )
        if source.shape[0] != 1 or not np.isfinite(source).all():
            raise ValueError("expected one finite NCHW input")
        channels = source.shape[1]
        rows = torch.from_numpy(source.reshape(channels, -1).T.copy())
        qx, scales = reference.quantize_rows(rows)
        values = {name + ".qx": qx.numpy(), name + ".activation_scale": scales.numpy()}
        if call["target_kind"] == "existing_case":
            entry = parameter_entries[name + ".qweight"]
            raw = parameter_bytes[entry["offset"] : entry["offset"] + entry["bytes"]]
            if (
                raw != qweight.tobytes()
                or hashlib.sha256(raw).hexdigest() != entry["sha256"]
            ):
                raise ValueError(
                    name + ": replay QWeight differs from the linked primary parameters"
                )
            entry = parameter_entries[name + ".zero_bias_i32"]
            if any(parameter_bytes[entry["offset"] : entry["offset"] + entry["bytes"]]):
                raise ValueError("MMA ABI placeholder bias is not zero")
            acc = reference.integer_mm(qx, torch.from_numpy(qweight).T)
            values = {name + ".acc": acc.numpy()}
        elif not call["target"].startswith("quantize_pack_nchw__"):
            raise ValueError("unsupported physical primitive " + call["target"])
        written = 0
        for span, argument in enumerate(call["arguments"]):
            if argument["value"] not in call["writes"]:
                continue
            written += 1
            array = np.ascontiguousarray(
                values[argument["value"]], dtype=dtype_map[argument["dtype"]]
            )
            spec = plan["tensors"][argument["value"]]
            if list(array.shape) != argument["shape"] or array.nbytes != spec["bytes"]:
                raise ValueError(
                    "reference shape/size does not match the exact runtime output span"
                )
            raw = array.tobytes()
            file = (
                expected_dir / f"call-{index:03d}-span-{span}-{argument['dtype']}.bin"
            )
            file.write_bytes(raw)
            outputs.append(
                {
                    "call": index,
                    "span": span,
                    "logical_node": name,
                    "target": call["target"],
                    "target_kind": call["target_kind"],
                    "value": argument["value"],
                    "dtype": argument["dtype"],
                    "shape": list(array.shape),
                    "bytes": len(raw),
                    "expected_fnv64": fnv(raw),
                    "expected_sha256": hashlib.sha256(raw).hexdigest(),
                    "expected_file": str(file.resolve()),
                    "input_node": input_node,
                    "input_sha256": source_hash,
                    "dump_sha256": sha(dumps[name]),
                }
            )
        if written != len(call["writes"]):
            raise ValueError("plan output span mapping is ambiguous")
    report = {
        "schema": "efficientnet-independent-physical-checkpoint-reference-v1",
        "scope": "CPU references for selected physical output bytes; independent from logical checkpoints; sparse hashes are not whole-model/full-output acceptance",
        "rounding": "symmetric W8A8, [-127,127], half away from zero, per spatial row",
        "mma": "signed i8 x i8 -> i32 exact dot; original zero ABI bias placeholder",
        "layout": "Q[HW,C], Scale[HW], Acc[HW,OC], contiguous little endian",
        "physical_calls": calls,
        "output_spans": outputs,
        "provenance": {
            "replay_dir": str(args.replay_dir.resolve()),
            "replay_trace_sha256": sha(replay_trace),
            "logical_reference_trace_sha256": sha(args.reference_trace),
            "physical_plan_sha256": sha(args.physical_plan),
            "parameter_manifest_sha256": sha(parameter_manifest),
            "parameter_blob_sha256": sha(parameter_blob),
            "original_primary_logits_sha256": replay_report["primary_golden_sha256"],
            "replay_logits_bit_mismatches": replay_report["output_bit_mismatches"],
            "generator_sha256": sha(Path(__file__)),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "calls": len(calls), "spans": len(outputs)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

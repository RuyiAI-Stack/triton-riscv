#!/usr/bin/env python3
"""Prepare real EfficientNet parameters and a physical, nonaliasing call plan.

This is executable integration preparation, not a linked or validated model.
Existing packed INT8 pointwise kernels are never rewritten. Missing graph
connectors reference independent prototypes in this directory.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402
from export import argument_shapes  # noqa: E402
from quantized_conv_pipeline import pipeline_layout  # noqa: E402

ITEM_BYTES = {"f32": 4, "i32": 4, "i8": 1}
EXP_BUDGET = {
    "exp": {
        "domain": "all-f32",
        "special_values": "ieee",
        "max_abs_error": 1.1210387714598537e-44,
        "max_rel_error": 2e-6,
    }
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(value):
    return (value + 63) & ~63


class Plan:
    def __init__(self, graph, cases, arrays):
        self.graph, self.cases, self.arrays = graph, cases, arrays
        self.by_case = {c["name"]: c for c in cases}
        self.calls, self.connectors, self.parameters, self.tensors = [], {}, {}, {}
        self.views, self.scratch_by_case, self.blockers = {}, {}, []
        self.tensor("input", graph["input"]["shape"], "f32", external=True)

    def tensor(self, name, shape, dtype="f32", external=False):
        spec = {
            "shape": list(shape),
            "dtype": dtype,
            "elements": math.prod(shape),
            "bytes": math.prod(shape) * ITEM_BYTES[dtype],
            "external": external,
        }
        if name in self.tensors and self.tensors[name] != spec:
            raise ValueError("tensor definition mismatch " + name)
        self.tensors[name] = spec
        return name

    def parameter(self, name, shape=None, dtype=None, zero=False):
        if name not in self.parameters:
            if zero:
                arr = np.zeros(shape, {"i32": "<i4"}[dtype])
            else:
                arr = self.arrays[name]
                dtype = {
                    np.dtype("float32"): "f32",
                    np.dtype("int8"): "i8",
                    np.dtype("int32"): "i32",
                }[arr.dtype]
            arr = np.ascontiguousarray(arr)
            self.parameters[name] = {
                "shape": list(arr.shape),
                "dtype": dtype,
                "elements": arr.size,
                "bytes": arr.nbytes,
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "source": "declared ABI-required zero i32 placeholder"
                if zero
                else "real checkpoint, inference BN folded and/or symmetric row quantized",
                "array": arr,
            }
        return name

    def arg(self, value, shape=None):
        source = "parameter" if value in self.parameters else "tensor"
        spec = self.parameters[value] if source == "parameter" else self.tensors[value]
        shape = list(shape or spec["shape"])
        if math.prod(shape) != spec["elements"]:
            raise ValueError("invalid descriptor view " + value)
        return {
            "source": source,
            "value": value,
            "shape": shape,
            "dtype": spec["dtype"],
        }

    def connector(self, name, constants, grid, arguments, math_accuracy=None):
        key = name + "__" + "_".join(str(constants[k]) for k in sorted(constants))
        spec = {
            "name": key,
            "kernel": name,
            "kernel_source": str(HERE / "connector_kernels.py"),
            "symbol": "triton_model_" + key,
            "constexprs": constants,
            "grid": grid,
            "arguments": [
                {"name": n, "rank": len(s), "dtype": d} for n, s, d in arguments
            ],
            "argument_shapes": [s for n, s, d in arguments],
            "signature": {
                n: {"f32": "*fp32", "i32": "*i32", "i8": "*i8"}[d]
                for n, s, d in arguments
            },
            "math_accuracy": math_accuracy,
            "status": "prototype_actual_IR_small_shape_checked_not_board_validated",
        }
        if name in ("quantize_pack_nchw", "dequantize_unpack_bias"):
            spec["performance_limitation"] = (
                "scalar correctness baseline; no efficient transpose lowering yet"
            )
        previous = self.connectors.get(key)
        if previous is not None and previous != spec:
            raise ValueError("connector signature collision " + key)
        self.connectors[key] = spec
        return key

    def call(self, node, target, args, outputs, connector=False, note=None):
        source = self.connectors[target] if connector else self.by_case[target]
        shapes = source["argument_shapes"] if connector else argument_shapes(source)
        signature = source["arguments"]
        if len(signature) != len(args):
            raise ValueError("arity " + target)
        for actual, declared, shape in zip(args, signature, shapes):
            if actual["dtype"] != declared["dtype"] or actual["shape"] != list(shape):
                raise ValueError(
                    "ABI mismatch "
                    + target
                    + ": "
                    + repr(actual)
                    + " != "
                    + repr((declared, shape))
                )
        reads = sorted(
            {
                a["value"]
                for a in args
                if a["source"] == "tensor" and a["value"] not in outputs
            }
        )
        call = {
            "index": len(self.calls),
            "logical_node": node["name"],
            "target": target,
            "target_kind": "connector" if connector else "existing_case",
            "arguments": args,
            "reads": reads,
            "writes": outputs,
            "symbol": source["symbol"],
            "grid": source["grid"],
        }
        if note:
            call["note"] = note
        self.calls.append(call)

    def make_connector(
        self, node, name, constants, grid, names, outputs, math_accuracy=None
    ):
        argdefs = [
            (n, self.tensors[v]["shape"], self.tensors[v]["dtype"]) for n, v in names
        ]
        target = self.connector(name, constants, grid, argdefs, math_accuracy)
        self.call(node, target, [self.arg(v) for n, v in names], outputs, True)

    def compile_graph(self):
        for node in self.graph["nodes"]:
            kind, name, output = node["kind"], node["name"], node["output"]
            inputs = node["inputs"]
            self.tensor(output, node["shape"])
            if kind == "reshape_view":
                self.views[output] = inputs[0]
                continue
            if kind in ("conv", "depthwise"):
                case = self.by_case[node["case"]]
                if case["family"] == "conv2d_i8":
                    _, ci, h, w = node["input_shapes"][0]
                    oc = node["shape"][1]
                    qx = self.tensor(name + ".qx", [h * w, ci], "i8")
                    acts = self.tensor(name + ".activation_scale", [h * w])
                    acc = self.tensor(name + ".acc", [h * w, oc], "i32")
                    qw = self.parameter(name + ".qweight")
                    ws = self.parameter(name + ".weight_scale")
                    bias = self.parameter(name + ".bias")
                    zero = self.parameter(
                        name + ".zero_bias_i32", [oc], "i32", zero=True
                    )
                    self.make_connector(
                        node,
                        "quantize_pack_nchw",
                        {"C": ci, "H": h, "W": w},
                        [h * w, 1, 1],
                        [("X", inputs[0]), ("QPacked", qx), ("ActivationScale", acts)],
                        [qx, acts],
                    )
                    self.call(
                        node,
                        case["name"],
                        [self.arg(v) for v in (qx, qw, zero, acc)],
                        [acc],
                        note="Preserved existing INT8 MMA kernel. Raw entry overwrites Acc; its Bias is ignored. F32 bias is added after dequantization.",
                    )
                    target = self.connector(
                        "dequantize_unpack_bias",
                        {"OC": oc, "H": h, "W": w},
                        [h * w, oc, 1],
                        [
                            ("Acc", [h * w, oc], "i32"),
                            ("ActivationScale", [h * w], "f32"),
                            ("WeightScale", [oc], "f32"),
                            ("Bias", [oc], "f32"),
                            ("Out", node["shape"], "f32"),
                        ],
                    )
                    self.call(
                        node,
                        target,
                        [self.arg(v) for v in (acc, acts, ws, bias, output)],
                        [output],
                        True,
                    )
                else:
                    weight, bias = (
                        self.parameter(name + ".weight"),
                        self.parameter(name + ".bias"),
                    )
                    self.call(
                        node,
                        case["name"],
                        [self.arg(v) for v in (inputs[0], weight, bias, output)],
                        [output],
                    )
                    self.scratch_by_case[case["name"]] = pipeline_layout(case)[
                        "total_bytes"
                    ]
            elif kind in ("silu", "sigmoid"):
                if node["case"]:
                    self.call(
                        node,
                        node["case"],
                        [self.arg(v) for v in (inputs[0], output)],
                        [output],
                    )
                else:
                    count = math.prod(node["shape"])
                    block = 128
                    self.make_connector(
                        node,
                        kind,
                        {"COUNT": count, "BLOCK": block},
                        [(count + block - 1) // block, 1, 1],
                        [("X", inputs[0]), ("Out", output)],
                        [output],
                        EXP_BUDGET,
                    )
            elif kind == "se_broadcast_multiply":
                _, c, h, w = node["shape"]
                block = 128
                self.make_connector(
                    node,
                    kind,
                    {"C": c, "SPATIAL": h * w, "BLOCK": block},
                    [(h * w + block - 1) // block, c, 1],
                    [("X", inputs[0]), ("Gate", inputs[1]), ("Out", output)],
                    [output],
                )
            elif kind == "gap":
                c = node["shape"][1]
                self.call(
                    node,
                    node["case"],
                    [self.arg(inputs[0]), self.arg(output, [1, c])],
                    [output],
                )
            elif kind == "add":
                self.call(
                    node,
                    node["case"],
                    [self.arg(v) for v in (*inputs, output)],
                    [output],
                )
            elif kind == "linear_w8a8_chain":
                # Root owns this new public F32 ABI, internally W8A8 composite.
                case = self.by_case["linear_1x1000x1536"]
                weight, bias = (
                    self.parameter(name + ".weight"),
                    self.parameter(name + ".bias"),
                )
                self.call(
                    node,
                    case["name"],
                    [self.arg(v) for v in (inputs[0], weight, bias, output)],
                    [output],
                    note="Public F32 ABI must bind to quantize -> i8xi8/i32 MMA -> dequant+bias composite; legacy F32 dot forbidden.",
                )
                try:
                    from quantized_linear_pipeline import (
                        pipeline_layout as linear_layout,
                    )

                    self.scratch_by_case[case["name"]] = linear_layout(case)[
                        "total_bytes"
                    ]
                except (ImportError, AttributeError):
                    self.blockers.append(
                        "Classifier W8A8 public composite scratch layout is not yet available; workspace total is provisional."
                    )
                if case.get("implementation") not in (
                    "w8a8_gemm",
                    "w8a8_linear",
                    "linear_w8a8",
                ):
                    self.blockers.append(
                        "Classifier public case binding requires validation after root finishes W8A8 composite; plan rejects legacy F32 dot execution."
                    )
            else:
                raise ValueError("unsupported graph node " + kind)
        self.plan_liveness()
        return self

    def root_tensor(self, value):
        while value in self.views:
            value = self.views[value]
        return value

    def plan_liveness(self):
        born, last = {}, {}
        for name, spec in self.tensors.items():
            if spec["external"]:
                born[self.root_tensor(name)] = -1
        for c in self.calls:
            for v in c["writes"]:
                if v in self.views:
                    raise ValueError("write to alias view")
                if v in born:
                    raise ValueError("multiple writers " + v)
                born[v] = c["index"]
            for v in c["reads"]:
                root = self.root_tensor(v)
                if root not in born:
                    raise ValueError("use before definition " + v)
                last[root] = c["index"]
        output = self.root_tensor(self.graph["output"]["value"])
        last[output] = len(self.calls)
        active, free, allocated = {}, [], {}
        end = 0
        peak_live = 0
        peak_at = None

        def combine():
            free.sort()
            merged = []
            for off, size in free:
                if merged and merged[-1][0] + merged[-1][1] == off:
                    merged[-1] = (merged[-1][0], merged[-1][1] + size)
                else:
                    merged.append((off, size))
            free[:] = merged

        for c in self.calls:
            # Outputs are reserved before releasing inputs, so public adapters
            # can enforce nonaliasing for every pointer on the current call.
            for v in c["writes"]:
                spec = self.tensors[v]
                size = align(spec["bytes"])
                off = None
                for j, (candidate, available) in enumerate(free):
                    if available >= size:
                        off = candidate
                        if available == size:
                            free.pop(j)
                        else:
                            free[j] = (candidate + size, available - size)
                        break
                if off is None:
                    off = end
                    end += size
                active[v] = (off, size)
                allocated[v] = {
                    "offset": off,
                    "reserved_bytes": size,
                    "born": c["index"],
                    "last_use": last.get(v, c["index"]),
                }
            live = sum(n for _, n in active.values())
            if live > peak_live:
                peak_live = live
                peak_at = c["index"]
            for v in list(active):
                if last.get(v, born[v]) <= c["index"]:
                    free.append(active.pop(v))
                    combine()
        for name, spec in self.tensors.items():
            root = self.root_tensor(name)
            if spec["external"] or self.tensors[root]["external"]:
                spec["storage"] = {"kind": "external_input", "root": root}
            else:
                spec["storage"] = {"kind": "workspace", "root": root, **allocated[root]}
        self.workspace = {
            "alignment": 64,
            "policy": "first-fit; allocate call outputs before freeing last-use inputs; no pointer aliases",
            "activation_high_water_bytes": end,
            "peak_live_activation_bytes": peak_live,
            "peak_live_call": peak_at,
            "composite_scratch_bytes": max(self.scratch_by_case.values(), default=0),
            "scratch_by_case": self.scratch_by_case,
            "scratch_offset": align(end),
            "provisional_total_bytes": align(end)
            + max(self.scratch_by_case.values(), default=0),
            "provisional": bool(self.blockers),
            "input_storage": "separate embedded/input tensor; not charged to workspace",
            "parameter_storage": "separate read-only parameters.bin; not charged to workspace",
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "build/model-integration/shape-graph-manifest.json",
    )
    p.add_argument(
        "--reference", type=Path, default=ROOT / "build/model-integration/reference"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "build/model-integration/physical-plan"
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    graph = json.loads(a.graph.read_text())
    reference = json.loads((a.reference / "reference-report.json").read_text())
    if sha(a.graph) != reference["graph_sha256"]:
        raise ValueError("graph differs from validated real-weight CPU reference")
    with np.load(
        a.reference / "folded-and-quantized-parameters.npz", allow_pickle=False
    ) as arrays:
        plan = Plan(graph, inventory(), arrays).compile_graph()
        params = bytearray()
        layout = []
        for name, spec in plan.parameters.items():
            offset = align(len(params))
            params.extend(bytes(offset - len(params)))
            raw = spec.pop("array").tobytes()
            params.extend(raw)
            spec["offset"] = offset
            layout.append({"name": name, **spec})
        (a.output / "parameters.bin").write_bytes(params)
    for name in ("input.f32", "original-logits.f32", "w8a8-logits.f32"):
        if sha(a.reference / name) != reference["output_hashes"][name]:
            raise ValueError("reference asset hash mismatch " + name)
        shutil.copyfile(a.reference / name, a.output / name)
    params_manifest = {
        "alignment": 64,
        "bytes": len(params),
        "sha256": sha(a.output / "parameters.bin"),
        "hf_revision": reference["hf_revision"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "entries": layout,
    }
    (a.output / "parameters.json").write_text(
        json.dumps(params_manifest, indent=2) + chr(10)
    )
    report = {
        "schema": "efficientnet-b3-physical-plan-v1",
        "production_ready": False,
        "board_submitted": False,
        "graph_sha256": sha(a.graph),
        "reference_report_sha256": sha(a.reference / "reference-report.json"),
        "planner_sha256": sha(__file__),
        "connector_source_sha256": sha(HERE / "connector_kernels.py"),
        "logical_nodes": len(graph["nodes"]),
        "physical_calls": len(plan.calls),
        "target_counts": dict(Counter(c["target_kind"] for c in plan.calls)),
        "unique_existing_cases": len(
            {c["target"] for c in plan.calls if c["target_kind"] == "existing_case"}
        ),
        "unique_connector_specializations": len(plan.connectors),
        "workspace": plan.workspace,
        "input": graph["input"],
        "output": graph["output"],
        "views": plan.views,
        "tensors": plan.tensors,
        "connectors": list(plan.connectors.values()),
        "calls": plan.calls,
        "parameter_manifest": "parameters.json",
        "parameter_blob": "parameters.bin",
        "blockers": plan.blockers
        + [
            "Connector specializations need full-shape compilation and numeric checks.",
            "NCHW pack/unpack connectors currently scalar baselines; performance is unvalidated.",
            "No linked whole-model C launcher/ELF or whole-model FPGA execution exists yet.",
            "Optimized operator artifacts must pass the parent phase gates before whole-model board submission.",
        ],
    }
    (a.output / "physical-plan.json").write_text(json.dumps(report, indent=2) + chr(10))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "production_ready",
                    "logical_nodes",
                    "physical_calls",
                    "target_counts",
                    "unique_existing_cases",
                    "unique_connector_specializations",
                    "workspace",
                )
            },
            indent=2,
        )
    )
    print("parameter_bytes", len(params), "parameter_sha256", params_manifest["sha256"])


if __name__ == "__main__":
    main()

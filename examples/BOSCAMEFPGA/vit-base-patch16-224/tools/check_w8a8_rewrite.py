#!/usr/bin/env python3
"""Check 73 dense rewrites and overwrite ABI using a graph structure fixture.

This fixture uses real Buddy graph operations but no model values. It verifies
graph surgery and descriptors, and makes no numerical or model accuracy claim.
"""

import argparse
import json
import tempfile
from pathlib import Path

import replace_triton as rewrite

from build import ROOT, inventory


def fixture(shapes):
    from buddy.compiler.graph import Graph, TensorDType
    from buddy.compiler.graph.operation import (
        AddOp,
        MatmulOp,
        PermuteOp,
        PlaceholderOp,
        SelectOp,
        ViewOp,
    )

    graph = Graph({}, "fixture", enable_external_calls=True)
    parameters, inputs, biases = [], [], []

    def node(cls, name, shape, args=(), parents=()):
        result = rewrite._new_node(
            graph, cls, name, args, shape, TensorDType.Float32
        )
        for parent in parents:
            result.add_parent(parent.name)
            parent.add_children(name)
        return result

    for number, (m, n, k) in enumerate(shapes):
        prefix = f"dense_{number}"
        weight = node(PlaceholderOp, prefix + "_weight", [n, k])
        bias = node(PlaceholderOp, prefix + "_bias", [n])
        source = node(
            PlaceholderOp, prefix + "_source", [1, 197 if m == 1 else m, k]
        )
        parameters.extend([weight, bias])
        inputs.append(source)
        if m == 1:
            activation = node(
                SelectOp,
                prefix + "_view",
                [m, k],
                [source.name, 1, 0],
                [source],
            )
        else:
            activation = node(
                ViewOp,
                prefix + "_view",
                [m, k],
                [source.name, [m, k]],
                [source],
            )
        transpose = node(
            PermuteOp,
            prefix + "_transpose",
            [k, n],
            [weight.name, [1, 0]],
            [weight],
        )
        product = node(
            MatmulOp,
            prefix,
            [m, n],
            [activation.name, transpose.name],
            [activation, transpose],
        )
        biased = node(
            AddOp,
            prefix + "_biased",
            [m, n],
            [product.name, bias.name],
            [product, bias],
        )
        biases.append(biased)
    rewrite._reindex(graph, parameters, inputs)
    return graph, biases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "build/w8a8-rewrite-check.json"
    )
    args = parser.parse_args()
    shapes = (
        [(197, 768, 768)] * 4 + [(197, 3072, 768), (197, 768, 3072)]
    ) * 12 + [(1, 1000, 768)]
    index = {
        case["name"]: {
            **case,
            "adapter_entry": "_mlir_ciface_kernel_" + case["name"],
        }
        for case in inventory()
    }
    graph, biases = fixture(shapes)
    calls, report = [], {"uncovered": []}
    assert rewrite.rewrite_linears_to_w8a8(graph, index, calls, report) == 73
    assert len(calls) == 219 and not report["uncovered"]
    assert all(
        call["case"].get("accumulation") == "overwrite" for call in calls[1::3]
    )
    assert len(graph.inputs) == 73 * 5
    for bias in biases:
        assert bias.args[0].endswith("_out") and bias._parents[0].endswith(
            "_out"
        )
    with tempfile.TemporaryDirectory(prefix="vit-w8a8-rewrite-") as directory:
        adapter = Path(directory) / "adapter.c"
        rewrite.generate_adapters(calls, adapter)
        text = adapter.read_text()
        assert "acc[i] = 0" not in text and "matmul_i8_overwrite_" in text
        assert "MemRef2 case_arg_2" in text
    missing = {
        name: case for name, case in index.items() if "overwrite" not in name
    }
    graph, _ = fixture([(1, 1000, 768)])
    rejected = {"uncovered": []}
    assert rewrite.rewrite_linears_to_w8a8(graph, missing, [], rejected) == 0
    assert (
        "matmul_i8_overwrite_1x1000x768" in rejected["uncovered"][0]["reason"]
    )
    result = {
        "schema": 1,
        "status": "PASS",
        "scope": "graph structure and generated ABI",
        "dense_rewrites": 73,
        "external_calls": 219,
        "fallback_to_accumulating_kernel_rejected": True,
        "numerical_validation": False,
        "full_model_executed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

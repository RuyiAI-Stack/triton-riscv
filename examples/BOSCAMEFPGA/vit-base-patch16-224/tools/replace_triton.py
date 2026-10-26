#!/usr/bin/env python3
"""Capture ViT again and replace shape-checked nodes with Triton calls.

The graph is captured with the same trace-friendly wrapper as import_model.py,
then edited as a Buddy graph.  Matching uses tensor rank/shape and dataflow,
never module names.  Every call is recorded with the kernel manifest and the
adapter generator emits only ABI glue; numerical work remains in Triton.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import (  # noqa: E402
    adapter_source,
    argument_shapes,
    describe,
    digest,
)


def shape_of(op):
    meta = getattr(op, "tensor_meta", None) or getattr(op, "_tensor_meta", None)
    if isinstance(meta, dict) and meta.get("shape") is not None:
        try:
            return [int(v) for v in meta["shape"]]
        except (TypeError, ValueError):
            return None
    return None


def dtype_of(op):
    meta = getattr(op, "tensor_meta", None) or getattr(op, "_tensor_meta", None)
    return str(meta.get("dtype")) if isinstance(meta, dict) else ""


# The W8A8 rewrite uses the same graph surgery conventions as the Qwen3
# importer.  Keep these tiny helpers local so this model can be captured with
# the ViT-specific matcher above (its matmul nodes are named ``mm`` rather than
# the generic ``MatmulOp`` in a few Buddy revisions).
def parents_of(graph, op):
    return [
        graph.node_table[name]
        for name in (getattr(op, "_parents", []) or [])
        if name in graph.node_table
    ]


def children_of(graph, op):
    return [
        graph.node_table[name]
        for name in (getattr(op, "_children", []) or [])
        if name in graph.node_table
    ]


def workspace_record(node):
    meta = getattr(node, "tensor_meta", {}) or {}
    return {
        "name": node.name,
        "shape": list(shape_of(node) or []),
        "dtype": str(meta.get("dtype")),
        "role": getattr(node, "workspace_role", "kernel_output"),
    }


def _reindex(graph, params, inputs):
    """Rebuild graph parameter/input body indices after W8A8 insertions."""
    index_of = {id(node): i for i, node in enumerate(graph.body)}
    graph._fake_params = sorted(
        index_of[id(node)] for node in params if id(node) in index_of
    )
    graph._inputs = sorted(
        index_of[id(node)] for node in inputs if id(node) in index_of
    )


def _placeholder_anchor(graph, parameters):
    known = {id(node) for node in parameters}
    return next((node for node in graph.body if id(node) not in known), None)


def _input_anchor(graph):
    return next(
        (node for node in graph.body if type(node).__name__ != "PlaceholderOp"),
        None,
    )


def _new_node(graph, cls, name, arguments, shape, dtype, before=None):
    import torch

    node = cls()
    node.name = name
    node._arguments = list(arguments)
    node.tensor_meta = {"shape": torch.Size(shape), "dtype": dtype}
    graph.node_table[name] = node
    if before is None:
        graph.body.append(node)
    else:
        graph.body.insert(graph.body.index(before), node)
    return node


def parents(graph, op):
    return [
        graph.node_table[n]
        for n in (getattr(op, "_parents", []) or [])
        if n in graph.node_table
    ]


def delete_detached(graph, node):
    """Remove a node after a replacement has already rewritten its users."""
    for producer in graph.body:
        producer._children[:] = [
            child for child in producer._children if child != node.name
        ]
    node._parents = []
    graph.delete_node(node, [])


def flat_shape(shape):
    if shape and len(shape) == 3 and shape[0] == 1:
        return shape[1:]
    return shape


def load_index(directory, rejected=None):
    """Index only complete current contract-pipeline kernel builds.

    Export creates frontend.json before lowering starts. Its existence cannot
    establish that a numerical external call has an executable implementation.
    """
    index = {}
    rejected = rejected if rejected is not None else []
    for path in sorted(Path(directory).glob("*/frontend.json")):
        data = json.loads(path.read_text())
        if data["name"] in index:
            raise ValueError("duplicate kernel manifest: " + data["name"])
        try:
            result = json.loads((path.parent / "build-result.json").read_text())
            if result.get("status") != "PASS" or result.get("stage") not in (
                "boscame",
                "llvm",
                "asm",
            ):
                raise ValueError(
                    result.get("reason", "kernel lowering is incomplete")
                )
            current = describe(data["name"])
            for key in (
                "name",
                "family",
                "symbol",
                "arguments",
                "signature",
                "constexprs",
                "grid",
                "shape",
            ):
                if data.get(key) != current[key]:
                    raise ValueError("current case metadata differs at " + key)
            if data.get("accumulation") != current.get("accumulation"):
                raise ValueError(
                    "compiled destination semantics differ from current case"
                )
            expected_adapter = hashlib.sha256(
                adapter_source(current).encode()
            ).hexdigest()
            if data.get("adapter_sha256") != expected_adapter:
                raise ValueError(
                    "compiled adapter differs from current graph ABI"
                )
            source_digest = digest(current["kernel_source"])
            if (
                data.get("kernel_module_sha256") != source_digest
                or result.get("kernel_module_sha256") != source_digest
            ):
                raise ValueError(
                    "current Triton source differs from successful build"
                )
            module = path.parent / "kernel.boscame.mlir"
            if result.get("artifacts", {}).get(module.name) != digest(module):
                raise ValueError(
                    "current emitted module differs from successful build"
                )
            for name, field in (
                ("kernel.ttir", "ttir_sha256"),
                ("adapter.c", "adapter_sha256"),
            ):
                if data.get(field) != digest(path.parent / name):
                    raise ValueError(
                        "current " + name + " differs from manifest"
                    )
        except (OSError, ValueError, KeyError) as error:
            rejected.append({"name": data["name"], "reason": str(error)})
            continue
        data["adapter_entry"] = "_mlir_ciface_kernel_" + data["name"]
        index[data["name"]] = data
    # A raw im2col entry is insufficient to implement the complete patch
    # projection. Its adapter's dependencies must also have current builds.
    while True:
        missing = {
            name: [
                dep for dep in case.get("dependencies", []) if dep not in index
            ]
            for name, case in index.items()
        }
        missing = {name: deps for name, deps in missing.items() if deps}
        if not missing:
            break
        for name, deps in missing.items():
            rejected.append(
                {
                    "name": name,
                    "reason": "missing compiled dependencies: "
                    + ", ".join(deps),
                }
            )
            del index[name]
    return index


def find_case(index, name, report, node, shape):
    case = index.get(name)
    if case is None:
        report["uncovered"].append(
            {
                "node": node.name,
                "shape": shape,
                "reason": "case not present in Triton build: " + name,
            }
        )
    return case


def match_layernorm_add(graph, op):
    """Recognize eager LayerNorm's decomposed VarMean arithmetic.

    Buddy's decomposition emits ``VarMean -> rsqrt -> sub/mul -> affine
    add`` rather than LayerNormOp.  Anchor on the final affine add and return
    the source/weight/bias tensors consumed by the Triton layernorm kernel.
    """
    if type(op).__name__ != "AddOp" or flat_shape(shape_of(op) or []) != [
        197,
        768,
    ]:
        return None
    ps = parents(graph, op)
    bias = next(
        (p for p in ps if len(shape_of(p) or []) == 1 and shape_of(p) == [768]),
        None,
    )
    affine = next((p for p in ps if p is not bias), None)
    if bias is None or affine is None or type(affine).__name__ != "MulOp":
        return None
    aps = parents(graph, affine)
    weight = next(
        (
            p
            for p in aps
            if len(shape_of(p) or []) == 1 and shape_of(p) == [768]
        ),
        None,
    )
    scaled = next((p for p in aps if p is not weight), None)
    if weight is None or scaled is None or type(scaled).__name__ != "MulOp":
        return None
    sps = parents(graph, scaled)
    normalized = next((p for p in sps if type(p).__name__ == "SubOp"), None)
    if normalized is None:
        return None
    nps = parents(graph, normalized)
    source = next((p for p in nps if shape_of(p) == shape_of(op)), None)
    if source is None:
        return None
    # Require the other normalized operand to be rsqrt(var + eps), avoiding a
    # false positive on ordinary elementwise products.
    mean_value = next((p for p in nps if p is not source), None)
    if mean_value is None or type(mean_value).__name__ != "GetItemOp":
        return None
    var = parents(graph, mean_value)
    if len(var) != 1 or type(var[0]).__name__ != "VarMeanOp":
        return None
    rsqrt = next((p for p in sps if p is not normalized), None)
    if rsqrt is None or type(rsqrt).__name__ != "RsqrtOp":
        return None
    eps_add = parents(graph, rsqrt)
    if len(eps_add) != 1 or type(eps_add[0]).__name__ != "AddOp":
        return None
    eps_args = eps_add[0].args
    if (
        len(eps_args) != 2
        or not isinstance(eps_args[1], (int, float))
        or not math.isclose(float(eps_args[1]), 1e-12)
    ):
        return None
    return {
        "operands": [source, weight, bias],
        "interior": [
            affine,
            scaled,
            normalized,
            mean_value,
            rsqrt,
            eps_add[0],
            var[0],
        ],
    }


def match_gelu_mul(graph, op):
    """Recognize exact GELU's decomposed erf/multiply chain."""
    if type(op).__name__ != "MulOp" or flat_shape(shape_of(op) or []) != [
        197,
        3072,
    ]:
        return None
    ps = parents(graph, op)
    if len(ps) != 2:
        return None
    erf_add = next(
        (
            p
            for p in ps
            if type(p).__name__ == "AddOp"
            and len(p.args) == 2
            and p.args[1] == 1
        ),
        None,
    )
    half = next((p for p in ps if p is not erf_add), None)
    if erf_add is None or half is None or type(half).__name__ != "MulOp":
        return None
    hargs = half.args
    if len(hargs) != 2 or hargs[1] != 0.5:
        return None
    erf_nodes = parents(graph, erf_add)
    if len(erf_nodes) != 1 or type(erf_nodes[0]).__name__ != "ErfOp":
        return None
    scaled = parents(graph, erf_nodes[0])
    if len(scaled) != 1 or type(scaled[0]).__name__ != "MulOp":
        return None
    sargs = scaled[0].args
    if len(sargs) != 2 or not math.isclose(
        float(sargs[1]), 0.7071067811865476, rel_tol=1e-12
    ):
        return None
    source = parents(graph, half)
    source_scaled = parents(graph, scaled[0])
    if (
        len(source) != 1
        or len(source_scaled) != 1
        or source[0] is not source_scaled[0]
    ):
        return None
    return {
        "operands": source,
        "interior": [half, erf_add, erf_nodes[0], scaled[0]],
    }


def match_softmax_div(graph, op):
    """Recognize the eager softmax ``exp(x-max(x)) / sum(exp(...))`` chain."""
    if type(op).__name__ != "DivOp" or shape_of(op) not in (
        [1, 12, 197, 197],
        [12, 197, 197],
    ):
        return None
    ps = parents(graph, op)
    if len(ps) != 2 or type(ps[0]).__name__ != "ExpOp":
        return None
    exp_node, denom = ps
    sub = parents(graph, exp_node)
    if len(sub) != 1 or type(sub[0]).__name__ != "SubOp":
        return None
    sub = sub[0]
    logits = parents(graph, sub)
    if not logits:
        return None
    scaled = logits[0]
    # The other operand is the row maximum.  Require the scaled QK value to
    # be the rank-4 tensor consumed by the eager attention chain.
    if shape_of(scaled) not in ([1, 12, 197, 197], [12, 197, 197]):
        scaled = logits[1] if len(logits) > 1 else None
    if scaled is None or shape_of(scaled) not in (
        [1, 12, 197, 197],
        [12, 197, 197],
    ):
        return None
    return {
        "operands": [scaled],
        "interior": [exp_node, sub, denom] + logits[1:],
    }


def match_linear(graph, op, ps):
    """Match ``A @ weight.T`` against the kernel's physical ``[N,K]`` B.

    The Triton linear kernel reads B as contiguous ``[N,K]`` and performs
    the transpose itself.  Passing the graph's PermuteOp would apply the
    transpose twice, even for square weights.  The captured ViT graph always
    presents a rank-2 weight placeholder through a [1,0] PermuteOp.
    """
    out = flat_shape(shape_of(op) or [])
    if len(out) != 2 or len(ps) != 2:
        return None
    left, transposed_weight = ps
    left_shape = shape_of(left) or []
    if len(left_shape) != 2 or type(transposed_weight).__name__ != "PermuteOp":
        return None
    permutation = (
        transposed_weight.args[1] if len(transposed_weight.args) == 2 else None
    )
    if list(permutation or []) != [1, 0]:
        return None
    weight_parents = parents(graph, transposed_weight)
    if (
        len(weight_parents) != 1
        or type(weight_parents[0]).__name__ != "PlaceholderOp"
    ):
        return None
    weight = weight_parents[0]
    m, n = out
    k = left_shape[-1]
    if (
        left_shape != [m, k]
        or shape_of(weight) != [n, k]
        or shape_of(transposed_weight) != [k, n]
    ):
        return None
    # ViT's rank-2 A is either a reshape of a dense [1,M,K] activation or
    # the classifier's CLS slice.  Both lower to a dense row-major tensor.
    activation_parents = parents(graph, left)
    if len(activation_parents) != 1:
        return None
    if type(left).__name__ == "ViewOp":
        if shape_of(activation_parents[0]) != [1, m, k]:
            return None
    elif type(left).__name__ == "SelectOp":
        if (
            left.args[1:] != [1, 0]
            or shape_of(activation_parents[0]) != [1, 197, k]
            or m != 1
        ):
            return None
    else:
        return None
    cases = {
        (197, 768, 768): "linear_197x768x768",
        (197, 3072, 768): "linear_197x3072x768",
        (197, 768, 3072): "linear_197x768x3072",
        (1, 1000, 768): "linear_1x1000x768",
    }
    case = cases.get((m, n, k))
    return (case, [left, weight]) if case else None


def _w8a8_case(index, prefix, rows, cols):
    """Require a compiled overwrite kernel for the model pure product."""
    candidates = []
    if prefix == "quantize":
        candidates = [f"quantize_{rows}x{cols}"]
    elif prefix == "dequantize":
        candidates = [f"dequantize_{rows}x{cols}"]
    elif prefix == "matmul":
        candidates = [
            f"matmul_i8_overwrite_{rows[0]}x{rows[1]}x{rows[2]}",
        ]
    for name in candidates:
        case = index.get(name)
        if case is not None:
            if prefix == "matmul" and case.get("accumulation") != "overwrite":
                continue
            return case
    return None


def rewrite_linears_to_w8a8(graph, index, calls, report, only=None):
    """Replace every ViT FP32 linear with explicit quantize/matmul/dequantize.

    The graph receives int8 weight/scale parameters and caller-owned workspace
    descriptors.  External calls are void: each destination is an explicit
    argument, which prevents one-shot-bufferize from inserting a copy around
    the AME accumulator.  A dedicated compiled overwrite kernel carries its
    own zero initial value and never reads the old accumulator workspace.
    """
    from buddy.compiler.graph import TensorDType
    from buddy.compiler.graph.operation import CallExternalOp, PlaceholderOp

    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    report.setdefault("original_parameters", [p.name for p in params_before])
    added_params, added_inputs = [], []
    matched = 0
    for op in list(graph.body):
        if type(op).__name__ not in ("MatmulOp", "MmOp", "LinearOp"):
            continue
        if only is not None and op.name not in only:
            continue
        ps = parents(graph, op)
        found = match_linear(graph, op, ps)
        if found is None:
            continue
        fp32_case, operands = found
        activation, weight = operands
        a_shape, w_shape, out_shape = (
            shape_of(activation),
            shape_of(weight),
            shape_of(op),
        )
        if not a_shape or not w_shape or not out_shape or len(a_shape) != 2:
            continue
        rows, k = a_shape
        n = w_shape[0]
        qcase = _w8a8_case(index, "quantize", rows, k)
        mcase = _w8a8_case(index, "matmul", (rows, n, k), None)
        dcase = _w8a8_case(index, "dequantize", rows, n)
        missing = [
            name
            for name, case in (
                (f"quantize_{rows}x{k}", qcase),
                (f"matmul_i8_overwrite_{rows}x{n}x{k}", mcase),
                (f"dequantize_{rows}x{n}", dcase),
            )
            if case is None
        ]
        if missing:
            report.setdefault("uncovered", []).append(
                {
                    "kind": "linear-w8a8",
                    "node": op.name,
                    "shape": [rows, n, k],
                    "reason": "no built case(s): " + ", ".join(missing),
                }
            )
            continue

        # Add quantised weights/scales as parameters immediately before the
        # first input placeholder.  Their names are stable for the image
        # packer, and the report records which FP32 checkpoint tensor they
        # replace.
        prefix = f"w8a8_{op.name}"
        anchor = _placeholder_anchor(graph, params_before)
        w8 = _new_node(
            graph,
            PlaceholderOp,
            prefix + "_weight_i8",
            [],
            w_shape,
            TensorDType.Int8,
            before=anchor,
        )
        ws = _new_node(
            graph,
            PlaceholderOp,
            prefix + "_weight_scale",
            [],
            [n],
            TensorDType.Float32,
            before=anchor,
        )
        added_params.extend([w8, ws])

        # Destinations are workspace inputs; the compiled overwrite matmul
        # establishes its zero initial value. Bias/residual users consume the
        # dequantized floating-point result later in graph order.
        in_anchor = _input_anchor(graph)
        workspace = []
        for suffix, shape, dtype, role in (
            ("_q", [rows, k], TensorDType.Int8, "quantized_activation"),
            ("_a_scale", [rows], TensorDType.Float32, "activation_scale"),
            ("_acc", [rows, n], TensorDType.Int32, "accumulator"),
            ("_out", [rows, n], TensorDType.Float32, "kernel_output"),
        ):
            node = _new_node(
                graph,
                PlaceholderOp,
                prefix + suffix,
                [],
                shape,
                dtype,
                before=in_anchor,
            )
            node.workspace_role = role
            workspace.append(node)
            added_inputs.append(node)
        quantized, a_scale, accumulator, result = workspace

        def emit(
            case, operands, suffix, destination_indices=(), prefix=prefix, op=op
        ):
            name = f"{prefix}_{suffix}"
            node = CallExternalOp(
                call_func_name=name,
                args=[x.name for x in operands],
                args_index=[0] * len(operands),
                tensor_meta={},
                name=name,
            )
            graph.node_table[name] = node
            graph.body.insert(graph.body.index(op), node)
            # Workspace buffers are explicit output arguments.  They must stay
            # in the call's ``_arguments`` list for ABI lowering, but marking a
            # destination as a graph parent makes GraphDriver count it once per
            # writer and once per reader (its input inference is occurrence
            # based in older Buddy builds).  Record only read operands in the
            # dependency edges; consumers still carry the output buffer as a
            # parent, so the placeholder appears exactly once in the subgraph
            # signature while call order remains the body order.
            for position, operand in enumerate(operands):
                if position in destination_indices:
                    continue
                node.add_parent(operand.name)
                operand.add_children(name)
            calls.append(
                {
                    "symbol": name,
                    "node": name,
                    "case": case,
                    "family": "w8a8",
                    "void": True,
                    "graph_ranks": [len(shape_of(x) or []) for x in operands],
                    "graph_shapes": [shape_of(x) for x in operands],
                    "result_rank": 0,
                    "result_shape": None,
                    "operands": list(operands),
                    "storage": None,
                }
            )
            report.setdefault("w8a8_calls", []).append(
                {
                    "node": name,
                    "case": case["name"],
                    "operands": [x.name for x in operands],
                    "operand_ranks": [len(shape_of(x) or []) for x in operands],
                }
            )
            return node

        emit(qcase, [activation, quantized, a_scale], qcase["name"], (1, 2))
        emit(mcase, [quantized, w8, accumulator], mcase["name"], (2,))
        emit(dcase, [accumulator, a_scale, ws, result], dcase["name"], (3,))

        # Replace the original matmul SSA value by the explicit output buffer.
        for child in children_of(graph, op):
            child._arguments = [
                result.name if arg == op.name else arg for arg in child.args
            ]
            child._parents[:] = [
                result.name if parent == op.name else parent
                for parent in child._parents
            ]
            result.add_children(child.name)
        # Remove the original op and dead transposed weight view.  The original
        # FP32 weight placeholder is pruned later once every linear is covered.
        graph.delete_node(op, parents(graph, op))
        transposed = graph.node_table.get(ps[1].name)
        if transposed is not None and not getattr(
            transposed, "_children", None
        ):
            delete_detached(graph, transposed)
        report.setdefault("w8a8_linears", []).append(
            {
                "node": op.name,
                "rows": rows,
                "n": n,
                "k": k,
                "fp32_case": fp32_case,
                "cases": [qcase["name"], mcase["name"], dcase["name"]],
                "weight_param": weight.name,
                "weight_param_name": w8.name,
                "scale_param_name": ws.name,
                "workspace": [node.name for node in workspace],
            }
        )
        matched += 1
    _reindex(graph, params_before + added_params, inputs_before + added_inputs)
    report.setdefault("w8a8_workspace_inputs", []).extend(
        workspace_record(node) for node in added_inputs
    )
    report["w8a8_linear_count"] = matched
    return matched


def result_descriptor(lines, rank, shape, name):
    count = math.prod(shape)
    lines.append(f"static float {name}[{count}] __attribute__((aligned(64)));")
    lines.append(f"static void init_{name}(MemRef{rank} *out) {{")
    lines.append("  out->allocated = (void *)0;")
    lines.append(f"  out->aligned = {name}; out->offset = 0;")
    stride = 1
    strides = []
    for value in reversed(shape):
        strides.append(stride)
        stride *= value
    for i, (size, step) in enumerate(zip(shape, reversed(strides))):
        lines.append(f"  out->sizes[{i}] = {size}; out->strides[{i}] = {step};")
    lines.append("}")


def normalized_arguments(lines, case, source_names, source_shapes):
    """Adapt only contiguous leading-unit batch dimensions, never C layouts.

    MemRefN stores N sizes before N strides, so casting between descriptor
    ranks corrupts both fields even when the tensor has the same elements.
    Validate the graph ABI and construct a fresh descriptor of the case rank.
    """
    shapes = argument_shapes(case)
    if len(source_names) != len(shapes) or len(source_shapes) != len(shapes):
        raise ValueError("incomplete graph argument shapes for " + case["name"])
    references = []
    pointers = []
    for index, (source, original, target, argument) in enumerate(
        zip(source_names, source_shapes, shapes, case["arguments"])
    ):
        if original is None:
            raise ValueError("missing graph argument shape for " + case["name"])
        original, target = tuple(original), tuple(target)
        low, high = sorted((original, target), key=len)
        if high != (1,) * (len(high) - len(low)) + low:
            raise ValueError(
                "only leading-unit batch rank changes are legal: "
                + str(original)
                + " -> "
                + str(target)
            )
        lines.append(
            f"  if (!{source} || {source}->offset < 0 || !{source}->aligned) __builtin_trap();"
        )
        stride = 1
        for axis in reversed(range(len(original))):
            stride_check = (
                f" || {source}->strides[{axis}] != {stride}"
                if original[axis] != 1
                else ""
            )
            lines.append(
                f"  if ({source}->sizes[{axis}] != {original[axis]}{stride_check}) __builtin_trap();"
            )
            stride *= original[axis]
        size = {"f32": 4, "i32": 4, "i8": 1}[argument["dtype"]]
        length = math.prod(target) * size
        pointer = f"case_pointer_{index}"
        lines.append(
            f"  if ((uint64_t){source}->offset > (UINTPTR_MAX - (uintptr_t){source}->aligned) / {size}) __builtin_trap();"
        )
        lines.append(
            f"  uintptr_t {pointer} = (uintptr_t){source}->aligned + (uint64_t){source}->offset * {size};"
        )
        lines.append(
            f"  if (!{pointer} || ({pointer} & 63u) || {pointer} > UINTPTR_MAX - {length}) __builtin_trap();"
        )
        for previous, previous_length in pointers:
            lines.append(
                f"  if (!({pointer} + {length} <= {previous} || {previous} + {previous_length} <= {pointer})) __builtin_trap();"
            )
        pointers.append((pointer, length))
        strides, stride = [], 1
        for extent in reversed(target):
            strides.insert(0, stride)
            stride *= extent
        rank = argument["rank"]
        if len(target) != rank:
            raise ValueError("kernel argument rank/shape disagreement")
        descriptor = f"case_arg_{index}"
        lines.append(
            f"  MemRef{rank} {descriptor} = {{{source}->allocated, {source}->aligned, {source}->offset, "
            + "{"
            + ", ".join(map(str, target))
            + "}, {"
            + ", ".join(map(str, strides))
            + "}};"
        )
        references.append("&" + descriptor)
    return references


def generate_adapters(calls, output):
    lines = [
        "/* Generated by vit-base-patch16-224/tools/replace_triton.py. */",
        '#include "support.h"',
        f"#define VIT_TRACE_CALL_COUNT {len(calls)}",
        "#if defined(VIT_TRACE_OPS) && VIT_TRACE_OPS",
        "#define VIT_TRACE_MARK(message) nr_puts(message)",
        "#else",
        "#define VIT_TRACE_MARK(message) ((void)0)",
        "#endif",
        "#ifdef NR_HANG_DIAGNOSTICS",
        "#define VIT_DIAG_MARK_ENTER(index) nr_diag_mark(NR_DIAG_KERNEL_ENTER, (index))",
        "#define VIT_DIAG_MARK_RETURN(index) nr_diag_mark(NR_DIAG_KERNEL_RETURN, (index))",
        "#else",
        "#define VIT_DIAG_MARK_ENTER(index) ((void)0)",
        "#define VIT_DIAG_MARK_RETURN(index) ((void)0)",
        "#endif",
        "",
    ]
    for call in calls:
        if not call.get("void"):
            result_descriptor(
                lines,
                call["result_rank"],
                call["result_shape"],
                call["storage"],
            )
    declared = set()
    for call in calls:
        entry = call["case"]["adapter_entry"]
        if entry in declared:
            continue
        declared.add(entry)
        ranks = [a["rank"] for a in call["case"]["arguments"]]
        lines.append(
            f"extern void {entry}({', '.join('MemRef%d *' % r for r in ranks)});"
        )
    lines.append("")
    for ordinal, call in enumerate(calls, start=1):
        symbol, case = call["symbol"], call["case"]
        graph_ranks = call["graph_ranks"]
        if call.get("void"):
            params = ", ".join(
                f"MemRef{r} *a{i}" for i, r in enumerate(graph_ranks)
            )
            lines.append(f"void _mlir_ciface_{symbol}({params}) {{")
            marker = f"[vit-op] {ordinal:03d}/{len(calls):03d} symbol={symbol} case={case['name']}"
            lines.append("#ifdef VIT_TRACE_TIMING")
            lines.append("  uint64_t vit_cycle_begin = nr_cycles();")
            lines.append("#endif")
            lines.append(
                "  VIT_TRACE_MARK(" + json.dumps(marker + " BEGIN\r\n") + ");"
            )
            lines.append("#ifdef VIT_TRACE_TIMING")
            lines.append(
                '  nr_puts("[vit-op-cycle] ' + marker + ' BEGIN cycles=0x");'
            )
            lines.append('  nr_hex64(vit_cycle_begin); nr_puts("\\r\\n");')
            lines.append("#endif")
            args = normalized_arguments(
                lines,
                case,
                [f"a{i}" for i in range(len(graph_ranks))],
                call["graph_shapes"],
            )
            # Pure-product initialization belongs to the compiled overwrite
            # kernel; this ABI bridge never clears or reads the C workspace.
            lines.append("#ifdef NR_HANG_DIAGNOSTICS")
            for operand, rank in enumerate(graph_ranks):
                cols = f"a{operand}->sizes[1]" if rank >= 2 else "1"
                stride1 = f"a{operand}->strides[1]" if rank >= 2 else "1"
                lines.extend(
                    [
                        f"  if (a{operand}) nr_diag_memref({operand}, (uintptr_t)a{operand}, "
                        f"(uintptr_t)a{operand}->aligned, a{operand}->offset, "
                        f"a{operand}->sizes[0], {cols}, a{operand}->strides[0], {stride1});",
                        f"  else nr_diag_memref({operand}, 0, 0, 0, 0, 0, 0, 0);",
                    ]
                )
            lines.append("#endif")
            lines.append(f"  VIT_DIAG_MARK_ENTER({ordinal}ULL);")
            lines.append(f"  {case['adapter_entry']}({', '.join(args)});")
            lines.append(f"  VIT_DIAG_MARK_RETURN({ordinal}ULL);")
            lines.append("#ifdef VIT_TRACE_TIMING")
            lines.append("  uint64_t vit_cycle_end = nr_cycles();")
            lines.append("#endif")
            lines.append(
                "  VIT_TRACE_MARK(" + json.dumps(marker + " END\r\n") + ");"
            )
            lines.append("#ifdef VIT_TRACE_TIMING")
            lines.append(
                '  nr_puts("[vit-op-cycle] ' + marker + ' END cycles=0x");'
            )
            lines.append('  nr_hex64(vit_cycle_end); nr_puts("\\r\\n");')
            lines.append("#endif")
            lines.append("}")
            lines.append("")
            continue
        params = ", ".join(
            f"MemRef{r} *a{i}"
            for i, r in enumerate([call["result_rank"]] + graph_ranks)
        )
        lines.append(f"void _mlir_ciface_{symbol}({params}) {{")
        marker = f"[vit-op] {ordinal:03d}/{len(calls):03d} symbol={symbol} case={case['name']}"
        lines.append("#ifdef VIT_TRACE_TIMING")
        lines.append("  uint64_t vit_cycle_begin = nr_cycles();")
        lines.append("#endif")
        lines.append(
            "  VIT_TRACE_MARK(" + json.dumps(marker + " BEGIN\r\n") + ");"
        )
        lines.append("#ifdef VIT_TRACE_TIMING")
        lines.append(
            '  nr_puts("[vit-op-cycle] ' + marker + ' BEGIN cycles=0x");'
        )
        lines.append('  nr_hex64(vit_cycle_begin); nr_puts("\\r\\n");')
        lines.append("#endif")
        lines.append(f"  init_{call['storage']}(a0);")
        lines.append("#ifdef NR_HANG_DIAGNOSTICS")
        ranks = [call["result_rank"], *graph_ranks]
        for operand, rank in enumerate(ranks[:3]):
            cols = f"a{operand}->sizes[1]" if rank >= 2 else "1"
            stride1 = f"a{operand}->strides[1]" if rank >= 2 else "1"
            lines.extend(
                [
                    f"  if (a{operand}) nr_diag_memref({operand}, (uintptr_t)a{operand}, "
                    f"(uintptr_t)a{operand}->aligned, a{operand}->offset, "
                    f"a{operand}->sizes[0], {cols}, a{operand}->strides[0], {stride1});",
                    f"  else nr_diag_memref({operand}, 0, 0, 0, 0, 0, 0, 0);",
                ]
            )
        lines.append("#endif")
        lines.append(f"  VIT_DIAG_MARK_ENTER({ordinal}ULL);")
        if call["family"] == "linear":
            lines.append(
                f"  for (size_t i = 0; i < sizeof({call['storage']}) / sizeof(float); ++i) {call['storage']}[i] = 0.0f;"
            )
        # Kernels use A,B,C or X,Bias,Out. The output is always last.
        args = normalized_arguments(
            lines,
            case,
            [f"a{i + 1}" for i in range(len(graph_ranks))] + ["a0"],
            call["graph_shapes"] + [call["result_shape"]],
        )
        lines.append(f"  {case['adapter_entry']}({', '.join(args)});")
        lines.append(f"  VIT_DIAG_MARK_RETURN({ordinal}ULL);")
        lines.append("#ifdef VIT_TRACE_TIMING")
        lines.append("  uint64_t vit_cycle_end = nr_cycles();")
        lines.append("#endif")
        lines.append(
            "  VIT_TRACE_MARK(" + json.dumps(marker + " END\r\n") + ");"
        )
        lines.append("#ifdef VIT_TRACE_TIMING")
        lines.append(
            '  nr_puts("[vit-op-cycle] ' + marker + ' END cycles=0x");'
        )
        lines.append('  nr_hex64(vit_cycle_end); nr_puts("\\r\\n");')
        lines.append("#endif")
        lines.append("}")
        lines.append("")
    output.write_text("\n".join(lines))


def prune_dead_w8a8_nodes(graph, report):
    """Drop dead FP32 weight placeholders/views after a complete rewrite."""
    removed = []
    view_types = {"PermuteOp", "ViewOp", "AliasOp", "ReshapeOp", "ExpandOp"}
    changed = True
    while changed:
        changed = False
        for node in list(graph.body):
            if getattr(node, "_children", None):
                continue
            if type(node).__name__ not in view_types:
                continue
            for parent in parents_of(graph, node):
                if node.name in parent._children:
                    parent._children.remove(node.name)
            graph.body.remove(node)
            graph.node_table.pop(node.name, None)
            removed.append(node.name)
            changed = True
    for node in list(graph.body):
        if type(node).__name__ == "PlaceholderOp" and not getattr(
            node, "_children", None
        ):
            graph.body.remove(node)
            graph.node_table.pop(node.name, None)
            removed.append(node.name)
    if removed:
        report.setdefault("pruned_nodes", []).extend(removed)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-dir", type=Path, required=True)
    parser.add_argument("--triton-build", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--w8a8",
        action="store_true",
        default=True,
        help="default: replace every FP32 linear with quantize -> int8 AME matmul -> dequantize",
    )
    mode.add_argument(
        "--fp32-reference",
        dest="w8a8",
        action="store_false",
        help="explicit floating-point dense reference; patch projection stays W8A8",
    )
    parser.add_argument(
        "--w8a8-only",
        action="append",
        default=None,
        help="restrict W8A8 rewrite to selected Matmul node names",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets = args.assets or (ROOT / "assets")
    args.checkpoint = args.checkpoint or (args.assets / "model.safetensors")
    rejected_kernels = []
    index = load_index(args.triton_build, rejected_kernels)
    if not index:
        raise SystemExit(
            "no frontend.json files found; build Triton cases first"
        )
    # The raw graph is useful even when an environment lacks Buddy bindings.
    raw_path = args.import_dir / "raw-ops.json"
    raw = json.loads(raw_path.read_text()) if raw_path.is_file() else []
    report = {
        "stage": "Buddy graph -> external Triton calls",
        "model": "google/vit-base-patch16-224",
        "triton_build": str(args.triton_build),
        "kernel_index_size": len(index),
        "rejected_kernels": rejected_kernels,
        "raw_op_count": len(raw),
        "replaced": [],
        "uncovered": [],
        "graphs": {},
        "status": "NO_MATCHES",
        "w8a8_enabled": bool(args.w8a8),
    }
    calls = []
    # Import the real graph, rather than fabricating MLIR from raw JSON.
    try:
        import torch
        from buddy.compiler.frontend import DynamoCompiler
        from buddy.compiler.graph import GraphDriver
        from buddy.compiler.graph.operation import (
            CallExternalOp,
            OpType,
            PlaceholderOp,
        )
        from buddy.compiler.graph.type import DeviceType
        from buddy.compiler.ops import tosa

        sys.path.insert(0, str(Path(__file__).parent))
        import import_model as importer

        model, config, _ = importer.load_model(
            args.assets / "config.json", args.checkpoint, torch.float32
        )
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            # Keep this identical to import_model.py so
            # the replacement pass sees the same
            # trace-friendly kernel boundaries.
            aot_autograd_decomposition=importer.capture_decompositions(),
            func_name="forward",
            enable_external_calls=True,
        )
        with torch.no_grad():
            graphs = compiler.importer(
                model,
                pixel_values=torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            )
        graph = graphs[0]
        graph._enable_external_calls = True
        # W8A8 surgery must precede the generic replacement loop: each original
        # Matmul is consumed by this pass and replaced by three void calls.
        if args.w8a8:
            rewrite_linears_to_w8a8(
                graph, index, calls, report, only=args.w8a8_only
            )
        for op in list(graph.body):
            # A previous pattern (currently scaled QK) may prune producers
            # while this stable snapshot is being visited.
            if op.name not in graph.node_table:
                continue
            # The W8A8 pass already replaced all FP32 MatmulOps.  The explicit
            # guard also avoids treating a workspace call as a numerical op in
            # Buddy revisions that expose CallExternalOp as Matmul-like.
            if type(op).__name__ == "CallExternalOp":
                continue
            ps = parents(graph, op)
            shape = shape_of(op) or []
            typ = type(op).__name__
            case_name = None
            operands = None
            family = None
            interior = []
            # In full W8A8 mode, leave an uncovered linear in the graph so the
            # coverage report and --require-complete fail loudly when a
            # quantize/matmul_i8/dequantize specialization is missing.  The
            # diagnostic --w8a8-only mode intentionally permits the remaining
            # linears to use their FP32 external kernels.
            if (
                args.w8a8
                and args.w8a8_only is None
                and typ in ("MatmulOp", "MmOp", "LinearOp")
            ):
                continue
            # Patch projection: Conv2d(3,768,16,16,stride=16).  The
            # Triton case writes NCHW directly, so subsequent flatten/transpose
            # nodes keep their original graph semantics.
            if ("Conv" in typ or "Convolution" in typ) and shape == [
                1,
                768,
                14,
                14,
            ]:
                patch_inputs = [
                    p for p in ps if len(shape_of(p) or []) in (1, 4)
                ]
                if len(patch_inputs) >= 3:
                    case_name = "patch_embed_196x768"
                    family = "patch_embed"
                    operands = patch_inputs[:3]
            # 2-D (or flattened batch) linear projections.
            if "Matmul" in typ or typ in ("MmOp", "LinearOp"):
                linear = match_linear(graph, op, ps)
                if linear is not None:
                    case_name, operands = linear
                    family = "linear"
            # Dynamo's decomposition emits BatchMatmulOp with rank-3 head
            # tensors (the leading batch dimension is folded into HEADS).
            # Match the PV product directly; QK is matched below at its
            # explicit 1/sqrt(head_dim) scale so the Triton kernel owns that
            # numerical factor exactly once.
            if case_name is None and typ == "BatchMatmulOp" and len(shape) == 3:
                if shape == [12, 197, 64] and len(ps) >= 2:
                    case_name = "attention_pv_12x197x64x197"
                    family = "attention_pv"
                    operands = ps[:2]
                elif shape == [12, 197, 197]:
                    # The unscaled QK producer is consumed by a scalar 0.125
                    # MulOp. Leave it for that consumer to replace below.
                    pass
            if (
                case_name is None
                and typ == "MulOp"
                and shape in ([12, 197, 197], [1, 12, 197, 197])
            ):
                scalar = op.args[1] if len(op.args) == 2 else None
                if isinstance(scalar, (int, float)) and math.isclose(
                    float(scalar), 0.125
                ):
                    scaled_view = ps[0] if ps else None
                    qk = (
                        parents(graph, scaled_view)[0]
                        if scaled_view is not None
                        and parents(graph, scaled_view)
                        else None
                    )
                    qk_parents = parents(graph, qk) if qk is not None else []
                    if (
                        qk is not None
                        and type(qk).__name__ == "BatchMatmulOp"
                        and shape_of(qk) == [12, 197, 197]
                        and len(qk_parents) >= 2
                    ):
                        case_name = "attention_qk_12x197x197x64"
                        family = "attention_qk"
                        operands = qk_parents[:2]
            # Explicit attention from TraceableViT: rank-4 batch head tensors.
            if case_name is None and "Matmul" in typ and len(shape) == 4:
                if shape[-2:] == [197, 197]:
                    case_name = "attention_qk_12x197x197x64"
                    family = "attention_qk"
                elif shape[-2:] == [197, 64]:
                    case_name = "attention_pv_12x197x64x197"
                    family = "attention_pv"
                if case_name and len(ps) >= 2:
                    operands = ps[:2]
            if (
                case_name is None
                and "Softmax" in typ
                and shape[-2:] == [197, 197]
            ):
                case_name = "softmax_12x197x197"
                family = "softmax"
                operands = ps[:1]
            if (
                case_name is None
                and ("Gelu" in typ or typ in ("GELUOp",))
                and flat_shape(shape) == [197, 3072]
            ):
                case_name = "gelu_197x3072"
                family = "gelu"
                operands = ps[:1]
            if case_name is None and typ == "MulOp":
                gelu = match_gelu_mul(graph, op)
                if gelu is not None:
                    case_name = "gelu_197x3072"
                    family = "gelu"
                    operands = gelu["operands"]
                    interior = gelu["interior"]
            if case_name is None and typ == "DivOp":
                softmax = match_softmax_div(graph, op)
                if softmax is not None:
                    case_name = "softmax_12x197x197"
                    family = "softmax"
                    operands = softmax["operands"]
                    interior = softmax["interior"]
            if (
                case_name is None
                and "LayerNorm" in typ
                and flat_shape(shape) == [197, 768]
                and len(ps) >= 3
            ):
                case_name = "layernorm_197x768"
                family = "layernorm"
                operands = ps[:3]
            if case_name is None and typ == "AddOp":
                ln = match_layernorm_add(graph, op)
                if ln is not None:
                    case_name = "layernorm_197x768"
                    family = "layernorm"
                    operands = ln["operands"]
                    interior = ln["interior"]
            if (
                case_name is None
                and typ == "AddOp"
                and flat_shape(shape)
                in ([196, 768], [197, 768], [197, 3072], [1, 768], [1, 1000])
            ):
                bias = next(
                    (p for p in ps if len(shape_of(p) or []) == 1), None
                )
                matrix = next((p for p in ps if p is not bias), None)
                if (
                    bias is not None
                    and matrix is not None
                    and shape_of(bias) == [flat_shape(shape)[1]]
                    and shape_of(matrix)
                    in (flat_shape(shape), [1] + flat_shape(shape))
                ):
                    case_name = "bias_add_%dx%d" % tuple(flat_shape(shape))
                    family = "bias_add"
                    operands = [matrix, bias]
                elif bias is None and flat_shape(shape) in (
                    [197, 768],
                    [1, 768],
                    [1, 1000],
                ):
                    case_name = "add_%dx%d" % tuple(flat_shape(shape))
                    family = "add"
                    operands = ps[:2]
            if not case_name:
                # Keep the report honest for numerical nodes that the matcher
                # was expected to cover.  Reshapes/transposes and indexing are
                # ordinary graph plumbing and deliberately remain Buddy ops;
                # an unrecognised Conv/Matmul/normalisation/etc. is a real
                # coverage gap and must make --require-complete fail.
                numerical = (
                    "Conv" in typ
                    or "Convolution" in typ
                    or "Matmul" in typ
                    or "Bmm" in typ
                    or typ
                    in (
                        "MmOp",
                        "LinearOp",
                        "AddMMOp",
                        "BatchMatmulOp",
                        "SoftmaxOp",
                        "GeluOp",
                        "GELUOp",
                        "LayerNormOp",
                    )
                ) and len(shape) >= 2
                if numerical:
                    report["uncovered"].append(
                        {
                            "node": op.name,
                            "shape": shape,
                            "reason": "unsupported numerical graph node: "
                            + typ,
                            "parents": list(getattr(op, "_parents", []) or []),
                            "children": list(
                                getattr(op, "_children", []) or []
                            ),
                        }
                    )
                continue
            case = find_case(index, case_name, report, op, shape)
            if (
                case is None
                or not operands
                or len(operands) != len(case["arguments"]) - 1
            ):
                continue
            symbol = "vit_graph_" + re.sub(r"[^A-Za-z0-9_]", "_", op.name)
            call = CallExternalOp(
                call_func_name=symbol,
                args=[p.name for p in operands],
                args_index=[0] * len(operands),
                tensor_meta=(
                    dict(op.tensor_meta)
                    if isinstance(getattr(op, "tensor_meta", None), dict)
                    else {"shape": shape, "dtype": dtype_of(op)}
                ),
                name=op.name,
            )
            graph.displace_node(op, call)
            call._op_type = OpType.Unfusable
            # displace_node preserves the old producer list; retarget it to the
            # physical tensors selected above (especially transposed weights).
            old_parents = list(getattr(call, "_parents", []) or [])
            call._parents = []
            for operand in operands:
                call.add_parent(operand.name)
                if call.name not in operand._children:
                    operand.add_children(call.name)
            for old_name in old_parents:
                if old_name not in [p.name for p in operands]:
                    producer = graph.node_table.get(old_name)
                    if producer is not None and call.name in producer._children:
                        producer._children.remove(call.name)
            call._arguments = [p.name for p in operands]
            call._args_index = [0] * len(operands)
            # The original Matmul consumed weight.T.  After the external
            # call consumes the physical weight, that PermuteOp is dead.  If
            # it stays in graph.body, GraphDriver treats its old dependency
            # as another subgraph input and duplicates every linear weight
            # in @vit's ABI.
            if family == "linear":
                transposed_weight = ps[1]
                if transposed_weight.name in graph.node_table and not getattr(
                    transposed_weight, "_children", None
                ):
                    graph.delete_node(
                        transposed_weight, parents(graph, transposed_weight)
                    )
            # QK replacement is anchored on the scaled rank-4 consumer, so
            # remove the now-dead view and unscaled BatchMatmul producer after
            # retargeting the call to their physical Q/K operands.
            if family == "attention_qk":
                dead = [p for p in ps if p.name in graph.node_table]
                for candidate in dead:
                    if (
                        candidate is call
                        or candidate.name not in graph.node_table
                    ):
                        continue
                    if (
                        getattr(candidate, "_children", None)
                        and type(candidate).__name__ != "ViewOp"
                    ):
                        continue
                    delete_detached(graph, candidate)
                for producer in list(graph.body):
                    if type(producer).__name__ != "BatchMatmulOp":
                        continue
                    if shape_of(producer) != [12, 197, 197]:
                        continue
                    if not getattr(producer, "_children", None):
                        delete_detached(graph, producer)
            # Remove the decomposed LayerNorm arithmetic once its source,
            # affine weight and bias have been retargeted to the Triton call.
            if family == "layernorm":
                for candidate in reversed(interior):
                    if candidate.name not in graph.node_table or getattr(
                        candidate, "_children", None
                    ):
                        continue
                    graph.delete_node(candidate, parents(graph, candidate))
            if family == "gelu":
                for candidate in reversed(interior):
                    if candidate.name not in graph.node_table or getattr(
                        candidate, "_children", None
                    ):
                        continue
                    graph.delete_node(candidate, parents(graph, candidate))
            if family == "softmax":
                for candidate in reversed(interior):
                    if candidate.name not in graph.node_table or getattr(
                        candidate, "_children", None
                    ):
                        continue
                    graph.delete_node(candidate, parents(graph, candidate))
            calls.append(
                {
                    "symbol": symbol,
                    "node": op.name,
                    "case": case,
                    "family": family,
                    "graph_ranks": [len(shape_of(p) or []) for p in operands],
                    "graph_shapes": [shape_of(p) for p in operands],
                    "result_rank": len(shape),
                    "result_shape": shape,
                    "storage": "vit_out_"
                    + re.sub(r"[^A-Za-z0-9_]", "_", op.name),
                }
            )
            report["replaced"].append(
                {
                    "ordinal": len(calls),
                    "node": op.name,
                    "case": case_name,
                    "family": family,
                    "symbol": symbol,
                    "shape": shape,
                    "operands": [p.name for p in operands],
                }
            )
        # QK replacement is anchored on the scaled MulOp.  Some Buddy graph
        # versions retain the old BatchMatmul -> View reverse chain even after
        # that consumer is displaced; remove that provably dead chain here.
        for stale in list(graph.body):
            if type(stale).__name__ != "BatchMatmulOp" or shape_of(stale) != [
                12,
                197,
                197,
            ]:
                continue
            children = [
                graph.node_table[n]
                for n in getattr(stale, "_children", [])
                if n in graph.node_table
            ]
            if children and all(
                type(child).__name__ == "ViewOp" for child in children
            ):
                for child in children:
                    delete_detached(graph, child)
            if stale.name in graph.node_table and not getattr(
                stale, "_children", None
            ):
                delete_detached(graph, stale)
        if args.w8a8:
            before_params = list(graph.params)
            before_inputs = list(graph.inputs)
            pruned = prune_dead_w8a8_nodes(graph, report)
            _reindex(graph, before_params, before_inputs)
            report["w8a8_pruned_count"] = len(pruned)
            report["parameter_layout"] = [
                {
                    "index": i,
                    "name": p.name,
                    "shape": list(shape_of(p) or []),
                    "dtype": str(
                        (getattr(p, "tensor_meta", {}) or {}).get("dtype")
                    ),
                }
                for i, p in enumerate(graph.params)
            ]

            # Persist the exact positional ABI after surgery.  The image packer
            # and host checker cannot infer whether two equal-shaped tensors
            # are a weight, a scale, or a caller-owned workspace from shape
            # alone; node names and dtypes are the stable anchor shared by the
            # graph and the checkpoint mapper.
            def _layout(nodes):
                return [
                    {
                        "name": node.name,
                        "shape": list(shape_of(node) or []),
                        "dtype": str(
                            (getattr(node, "tensor_meta", {}) or {}).get(
                                "dtype"
                            )
                        ),
                        "role": getattr(node, "workspace_role", "parameter"),
                    }
                    for node in nodes
                ]

            report["parameter_layout"] = _layout(list(graph.params))
            report["input_layout"] = _layout(list(graph.inputs))
            report["placeholder_sequence"] = _layout(
                [node for node in graph.body if isinstance(node, PlaceholderOp)]
            )
        report["uncovered"] = [
            entry
            for entry in report["uncovered"]
            if entry["node"] in graph.node_table
        ]
        graph.op_groups = {
            "vit": [
                op for op in graph.body if not isinstance(op, PlaceholderOp)
            ]
        }
        graph.group_map_device = {"vit": DeviceType.CPU}
        driver = GraphDriver(graph)
        driver.subgraphs[0].lower_to_top_level_ir()
        (args.output / "subgraph0.triton.mlir").write_text(
            str(driver.subgraphs[0]._imported_module)
        )
        (args.output / "forward.triton.mlir").write_text(
            str(driver.construct_main_graph(True))
        )
        report["graphs"]["forward"] = {
            "ops_after": len(graph.body),
            "external_calls": len(calls),
        }
    except Exception as error:
        report["capture_error"] = f"{type(error).__name__}: {error}"
        # Keep the report actionable; do not claim a replacement if capture failed.
    # ``rewrite_linears_to_w8a8`` runs before the generic matcher, so its 3xN
    # records were appended first.  Runtime trace order must follow graph body
    # order; stable-sort by each external call's node position before emitting
    # the adapters and ordinal markers.
    if calls:
        body_order = (
            {node.name: i for i, node in enumerate(graph.body)}
            if "graph" in locals()
            else {}
        )
        calls.sort(key=lambda call: body_order.get(call.get("node"), 1 << 30))
        # The generic matcher records replacement metadata as it walks the
        # graph, while W8A8 emits three void calls per linear before that walk.
        # Rebuild one canonical, ordinal-ordered list so the UART trace
        # summarizer can validate all 391 adapter boundaries.
        previous = {
            entry.get("symbol"): entry for entry in report.get("replaced", [])
        }
        records = []
        for ordinal, call in enumerate(calls, 1):
            record = dict(previous.get(call["symbol"], {}))
            record.update(
                {
                    "ordinal": ordinal,
                    "node": call.get("node"),
                    "symbol": call["symbol"],
                    "case": call["case"]["name"],
                    "family": call.get("family"),
                }
            )
            if call.get("void"):
                record.update(
                    {
                        "shape": None,
                        "void": True,
                        "operands": [
                            operand.name for operand in call.get("operands", [])
                        ],
                    }
                )
            records.append(record)
        report["replaced"] = records
        generate_adapters(calls, args.output / "vit_triton_adapters.c")
    else:
        (args.output / "vit_triton_adapters.c").write_text(
            "/* no graph calls were replaced */\n"
        )
    if (
        args.w8a8
        and args.w8a8_only is None
        and report.get("w8a8_linear_count") != 73
    ):
        report["uncovered"].append(
            {
                "node": "model_dense_coverage",
                "reason": "W8A8 requires exactly 73 rewritten dense linears",
                "actual": report.get("w8a8_linear_count", 0),
            }
        )
    report["status"] = (
        "CAPTURE_FAILED"
        if report.get("capture_error")
        else "PASS"
        if calls and not report["uncovered"]
        else "PARTIAL"
        if calls
        else "NO_MATCHES"
    )
    report["distinct_cases"] = sorted({c["case"]["name"] for c in calls})
    report["adapter"] = str(args.output / "vit_triton_adapters.c")
    report["operator_trace"] = {
        "compile_define": "VIT_TRACE_OPS=1",
        "timing_compile_define": "VIT_TRACE_TIMING=1",
        "enabled_by_default": False,
        "call_count": len(calls),
        "marker": "[vit-op] ordinal/total symbol=... case=... BEGIN|END",
        "timing_marker": "[vit-op-cycle] ... BEGIN|END cycles=0x...",
    }
    (args.output / "triton-call-replacement.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if args.require_complete and report["status"] != "PASS":
        raise SystemExit(f"replacement incomplete: {report['status']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

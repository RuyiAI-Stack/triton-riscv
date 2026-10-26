#!/usr/bin/env python3
"""Capture google/vit-base-patch16-224 with Buddy's Dynamo frontend.

Only this script talks to HuggingFace/PyTorch. Later stages consume the saved
raw graph and MLIR, which makes it possible to audit every external Triton
replacement against a single fixed `[1,3,224,224]` input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _json(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def shape_of(op):
    meta = getattr(op, "tensor_meta", None)
    if isinstance(meta, dict) and meta.get("shape") is not None:
        try:
            return [int(x) for x in meta["shape"]]
        except (TypeError, ValueError):
            return str(meta["shape"])
    meta = getattr(op, "_tensor_meta", None)
    if isinstance(meta, dict) and meta.get("shape") is not None:
        try:
            return [int(x) for x in meta["shape"]]
        except (TypeError, ValueError):
            return str(meta["shape"])
    return None


def dtype_of(op):
    meta = getattr(op, "tensor_meta", None) or getattr(op, "_tensor_meta", None)
    return (
        str(meta.get("dtype"))
        if isinstance(meta, dict) and meta.get("dtype") is not None
        else None
    )


def describe_graph(graph):
    entries = []
    for index, op in enumerate(graph.body):
        entry = {
            "index": index,
            "name": op.name,
            "op_type": type(op).__name__,
            "arguments": [_json(v) for v in getattr(op, "args", [])],
            "keyword_arguments": {
                k: _json(v) for k, v in getattr(op, "kwargs", {}).items()
            },
            "parents": list(getattr(op, "_parents", []) or []),
            "children": list(getattr(op, "_children", []) or []),
        }
        shape = shape_of(op)
        dtype = dtype_of(op)
        if shape is not None or dtype is not None:
            entry["tensor_meta"] = {
                k: v
                for k, v in (("shape", shape), ("dtype", dtype))
                if v is not None
            }
        entries.append(entry)
    return entries


def load_model(config_path: Path, checkpoint: Path, dtype):
    import torch
    from safetensors.torch import load_file
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig.from_pretrained(str(config_path))
    model = ViTForImageClassification(config).eval()
    state = load_file(str(checkpoint), device="cpu")
    own = model.state_dict()
    # Transformers 5.x renamed the encoder modules (``encoder.layer`` /
    # ``attention.attention.query`` -> ``layers`` /
    # ``attention.q_proj``), while the pinned google checkpoint still uses the
    # 4.x names.  Normalize that harmless naming difference so graph capture
    # works in either supported Transformers environment.
    if set(own) != set(state):
        aliases = (
            ("vit.encoder.layer.", "vit.layers."),
            (".attention.attention.query.", ".attention.q_proj."),
            (".attention.attention.key.", ".attention.k_proj."),
            (".attention.attention.value.", ".attention.v_proj."),
            (".attention.output.dense.", ".attention.o_proj."),
            (".intermediate.dense.", ".mlp.fc1."),
            (".output.dense.", ".mlp.fc2."),
        )
        remapped = {}
        for name, tensor in state.items():
            normalized = name
            for old, new in aliases:
                normalized = normalized.replace(old, new)
            remapped[normalized] = tensor
        if set(remapped) == set(own):
            state = remapped
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=True)
    model = model.to(dtype=dtype)
    model.requires_grad_(False)

    class TraceableViT(torch.nn.Module):
        """Keep attention as matmul -> scale -> softmax -> matmul.

        HuggingFace may dispatch to fused scaled-dot-product attention.  That
        hides the QK/PV boundaries the FPGA kernels replace, so this wrapper
        preserves the mathematically identical eager ordering while reusing
        every official module and parameter.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            vit = self.inner.vit

            def project(module, value):
                # Keep bias as an explicit AddOp so the linear and bias_add
                # Triton ABIs remain separate.
                return (
                    torch.matmul(value, module.weight.transpose(-1, -2))
                    + module.bias
                )

            hidden = vit.embeddings(pixel_values)
            blocks = getattr(vit, "layers", None)
            if blocks is None:
                blocks = vit.encoder.layer
            for block in blocks:
                residual = hidden
                x = block.layernorm_before(hidden)
                # Transformers 4.x wraps Q/K/V in ``attention.attention``;
                # 5.x exposes them directly on ``ViTAttention``.
                attn = getattr(block.attention, "attention", block.attention)
                q = project(
                    getattr(attn, "q_proj", getattr(attn, "query", None)), x
                )
                k = project(
                    getattr(attn, "k_proj", getattr(attn, "key", None)), x
                )
                v = project(
                    getattr(attn, "v_proj", getattr(attn, "value", None)), x
                )
                batch, seq, width = q.shape
                heads = getattr(
                    attn, "num_attention_heads", config.num_attention_heads
                )
                dim = getattr(
                    attn, "attention_head_size", config.hidden_size // heads
                )
                q = q.view(batch, seq, heads, dim).transpose(1, 2)
                k = k.view(batch, seq, heads, dim).transpose(1, 2)
                v = v.view(batch, seq, heads, dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) * (dim**-0.5)
                probs = torch.softmax(scores, dim=-1)
                context = torch.matmul(probs, v)
                context = (
                    context.transpose(1, 2).contiguous().view(batch, seq, width)
                )
                output_proj = getattr(attn, "o_proj", None)
                if output_proj is None:
                    output_proj = block.attention.output.dense
                hidden = residual + project(output_proj, context)
                residual = hidden
                x = block.layernorm_after(hidden)
                mlp = getattr(block, "mlp", None)
                intermediate = (
                    getattr(mlp, "fc1", None) if mlp is not None else None
                )
                if intermediate is None:
                    intermediate = block.intermediate.dense
                x = project(intermediate, x)
                x = torch.nn.functional.gelu(x, approximate="none")
                output = getattr(mlp, "fc2", None) if mlp is not None else None
                if output is None:
                    output = block.output.dense
                x = project(output, x)
                hidden = residual + x
            hidden = vit.layernorm(hidden)
            return project(self.inner.classifier, hidden[:, 0, :])

    return (
        TraceableViT(model),
        config,
        {
            "tensor_count": len(state),
            "missing": missing,
            "unexpected": unexpected,
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
        },
    )


def capture_decompositions():
    """Keep matmul boundaries while retaining Buddy's supported decompositions.

    The trace wrapper spells projections as matmul plus an explicit bias add,
    so removing only ``matmul.default`` keeps the Triton linear and bias_add
    cases independently replaceable.  Buddy's native softmax/GELU/layernorm
    decompositions remain enabled because their fused op forms are not
    registered by every supported frontend build.
    """
    import torch
    from torch._inductor.decomposition import decompositions as base

    table = dict(base)
    names = ("matmul.default",)
    for name in names:
        op_name, overload = name.split(".", 1)
        try:
            op = getattr(torch.ops.aten, op_name)
            key = getattr(op, overload)
        except (AttributeError, RuntimeError):
            continue
        table.pop(key, None)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="directory containing config.json and preprocessor_config.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--save-mlir", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.ops import tosa

    dtype = getattr(torch, args.dtype)
    model, config, weights = load_model(
        args.assets / "config.json", args.checkpoint, dtype
    )
    pixel_values = torch.zeros((1, 3, 224, 224), dtype=dtype)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=capture_decompositions(),
        func_name="forward",
    )
    started = time.time()
    with torch.no_grad():
        graphs = compiler.importer(model, pixel_values=pixel_values)
    if len(graphs) != 1:
        raise RuntimeError(f"expected one graph, got {len(graphs)}")
    graph = graphs[0]
    raw = describe_graph(graph)
    (args.output / "raw-ops.json").write_text(json.dumps(raw, indent=2) + "\n")
    params = compiler.imported_params[graph]
    param_records = []
    for index, tensor in enumerate(params):
        array = tensor.detach().cpu().contiguous().numpy()
        param_records.append(
            {
                "index": index,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "elements": int(array.size),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    (args.output / "params.json").write_text(
        json.dumps(param_records, indent=2) + "\n"
    )
    report = {
        "model": "google/vit-base-patch16-224",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
        "dtype": args.dtype,
        "config": {
            k: _json(getattr(config, k))
            for k in (
                "image_size",
                "patch_size",
                "num_channels",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "layer_norm_eps",
                "num_labels",
            )
        },
        "weights": weights,
        "raw_op_count": len(raw),
        "placeholders": [
            op.name for op in graph.body if type(op).__name__ == "PlaceholderOp"
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if args.save_mlir:
        group = "subgraph0"
        graph.op_groups = {
            group: [
                op for op in graph.body if not isinstance(op, PlaceholderOp)
            ]
        }
        graph.group_map_device = {group: DeviceType.CPU}
        driver = GraphDriver(graph)
        driver.subgraphs[0].lower_to_top_level_ir()
        (args.output / "subgraph0.mlir").write_text(
            str(driver.subgraphs[0]._imported_module)
        )
        (args.output / "forward.mlir").write_text(
            str(driver.construct_main_graph(True))
        )
        report["mlir"] = "forward.mlir"
    (args.output / "import.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

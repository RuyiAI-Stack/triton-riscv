#!/usr/bin/env python3
"""Export pretrained timm MobileNetV3 to the existing fixed Triton kernel ABI.

The generated runner calls real compiled Triton adapters, with fused inference
batch normalization and a checked, non-aliasing activation memory plan. This
module does not provide replacement C implementations of any model operators.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402
from export import argument_shapes  # noqa: E402

MODEL = "mobilenetv3_small_100.lamb_in1k"


def align64(n):
    return (n + 63) & ~63


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Graph:
    def __init__(self, model, image):
        self.model = model
        self.values = {"input": image.detach().clone()}
        self.constants = {"input": image.detach().clone()}
        self.nodes = []

    def constant(self, name, value):
        assert name not in self.values, name
        self.values[name] = value.detach().contiguous().clone()
        self.constants[name] = self.values[name]
        return name

    def emit(self, name, case, inputs, value):
        # Both the physical tensor sizes and every adapter rank are checked.
        spec = describe(case)
        output = name + ".output"
        value = value.detach().contiguous().clone()
        physical = argument_shapes(spec)
        args = inputs + [output]
        arrays = [self.values[x] for x in inputs] + [value]
        if output in self.values:
            raise ValueError(f"{name}: duplicate graph output {output}")
        if not len(args) == len(physical) == len(spec["arguments"]):
            raise ValueError(f"{name}: argument count does not match {case} ABI")
        for arg, array, shape, metadata in zip(
            args, arrays, physical, spec["arguments"]
        ):
            if array.numel() != math.prod(shape) or len(shape) != metadata["rank"]:
                raise ValueError(
                    f"{name}: incompatible argument {arg}: {array.shape} vs {shape}"
                )
        self.values[output] = value
        self.nodes.append(
            {
                "name": name,
                "case": case,
                "args": args,
                "shapes": physical,
                "output": output,
            }
        )
        return output

    def activation(self, name, x, activation):
        import torch
        import torch.nn.functional as F

        if isinstance(activation, torch.nn.Identity):
            return x
        families = {
            torch.nn.ReLU: ("relu", F.relu),
            torch.nn.Hardswish: ("hardswish", F.hardswish),
            torch.nn.Hardsigmoid: ("hardsigmoid", F.hardsigmoid),
        }
        family, fn = families[type(activation)]
        return self.emit(
            name, f"{family}_count{self.values[x].numel()}", [x], fn(self.values[x])
        )

    def conv(self, name, x, conv, bn=None):
        import torch
        import torch.nn.functional as F

        tensor = self.values[x]
        n, ci, h, w = tensor.shape
        co = conv.out_channels
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        ph, pw = conv.padding
        assert n == 1 and h == w and kh == kw and sh == sw and ph == pw
        assert conv.dilation == (1, 1)
        weight = conv.weight.detach()
        bias = conv.bias.detach() if conv.bias is not None else torch.zeros(co)
        if bn is not None:
            assert not bn.training and isinstance(bn.drop, torch.nn.Identity)
            scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            weight = weight * scale.reshape(-1, 1, 1, 1)
            bias = (bias - bn.running_mean) * scale + bn.bias.detach()
        if conv.groups == ci and ci == co:
            case = f"dwconv_c{ci}_h{h}_k{kh}_s{sh}_p{ph}"
        elif kh == 1 and sh == 1 and ph == 0 and conv.groups == 1:
            case = f"pwconv_cin{ci}_cout{co}_h{h}_w{w}"
        elif name == "conv_stem" and conv.groups == 1:
            case = f"conv_stem_cin{ci}_cout{co}_h{h}_k{kh}_s{sh}_p{ph}"
        else:
            raise ValueError(f"unsupported convolution: {name}: {conv}")
        weights = self.constant(name + ".weight", weight)
        biases = self.constant(name + ".bias", bias)
        out = F.conv2d(
            tensor, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups
        )
        result = self.emit(name, case, [x, weights, biases], out)
        return (
            self.activation(name + ".act", result, bn.act) if bn is not None else result
        )

    def mean(self, name, x):
        tensor = self.values[x]
        _, c, h, w = tensor.shape
        return self.emit(
            name, f"mean_hw_c{c}_h{h}_w{w}", [x], tensor.mean((2, 3), keepdim=True)
        )

    def se(self, name, x, module):
        import torch

        if isinstance(module, torch.nn.Identity):
            return x
        _, c, h, w = self.values[x].shape
        y = self.mean(name + ".mean", x)
        y = self.conv(name + ".reduce", y, module.conv_reduce)
        y = self.activation(name + ".relu", y, module.act1)
        y = self.conv(name + ".expand", y, module.conv_expand)
        y = self.activation(name + ".gate", y, module.gate)
        return self.emit(
            name + ".mul",
            f"se_mul_c{c}_h{h}_w{w}",
            [x, y],
            self.values[x] * self.values[y],
        )

    def build(self):
        import torch
        import torch.nn.functional as F
        from timm.models._efficientnet_blocks import (
            ConvBnAct,
            DepthwiseSeparableConv,
            InvertedResidual,
        )

        m = self.model
        x = self.conv("conv_stem", "input", m.conv_stem, m.bn1)
        for stage_id, stage in enumerate(m.blocks):
            for block_id, block in enumerate(stage):
                name = f"blocks.{stage_id}.{block_id}"
                shortcut = x
                assert isinstance(block.aa, torch.nn.Identity)
                if isinstance(block, DepthwiseSeparableConv):
                    assert block.conv_s2d is None
                    x = self.conv(name + ".dw", x, block.conv_dw, block.bn1)
                    x = self.se(name + ".se", x, block.se)
                    x = self.conv(name + ".pw", x, block.conv_pw, block.bn2)
                elif isinstance(block, InvertedResidual):
                    assert block.conv_s2d is None
                    x = self.conv(name + ".pw", x, block.conv_pw, block.bn1)
                    x = self.conv(name + ".dw", x, block.conv_dw, block.bn2)
                    x = self.se(name + ".se", x, block.se)
                    x = self.conv(name + ".pwl", x, block.conv_pwl, block.bn3)
                elif isinstance(block, ConvBnAct):
                    x = self.conv(name + ".conv", x, block.conv, block.bn1)
                else:
                    raise ValueError(type(block))
                if block.has_skip:
                    x = self.emit(
                        name + ".add",
                        f"add_count{self.values[x].numel()}",
                        [x, shortcut],
                        self.values[x] + self.values[shortcut],
                    )
        assert m.global_pool.pool_type == "avg"
        x = self.mean("global_pool", x)
        x = self.conv("conv_head", x, m.conv_head)
        assert isinstance(m.norm_head, torch.nn.Identity)
        x = self.activation("act2", x, m.act2)
        weight = self.constant("classifier.weight", m.classifier.weight)
        bias = self.constant("classifier.bias", m.classifier.bias)
        x = self.emit(
            "classifier",
            "linear_m1_n1000_k1024",
            [x, weight, bias],
            F.linear(self.values[x].flatten(1), self.values[weight], self.values[bias]),
        )
        self.output = x
        return self

    def memory_plan(self):
        # Allocate output BEFORE releasing any input: adapters forbid all aliasing.
        uses = {}
        for i, node in enumerate(self.nodes):
            for name in node["args"][:-1]:
                uses[name] = i
        uses[self.output] = len(self.nodes)
        active, offsets = {}, {}
        high = 0
        for i, node in enumerate(self.nodes):
            output = node["output"]
            size = align64(self.values[output].numel() * 4)
            cursor = 0
            for start, length in sorted(active.values()):
                if cursor + size <= start:
                    break
                cursor = max(cursor, start + length)
            offsets[output] = cursor
            active[output] = (cursor, size)
            high = max(high, cursor + size)
            intervals = sorted(active.values())
            assert all(a + b <= c for (a, b), (c, d) in zip(intervals, intervals[1:]))
            active = {name: region for name, region in active.items() if uses[name] > i}
        return offsets, high


def write_runner(
    graph, output, constant_offsets, activation_offsets, golden_offset, top1
):
    cases = sorted({n["case"] for n in graph.nodes})
    lines = [
        "/* Generated by tools/export_model.py. Do not edit. */",
        '#include "support.h"',
        '#include "model_support.h"',
        "extern const unsigned char mobilenet_model_blob[];",
    ]
    for case in cases:
        args = describe(case)["arguments"]
        lines.append(
            f"extern void _mlir_ciface_kernel_{case}("
            + ", ".join(f"MemRef{a['rank']} *" for a in args)
            + ");"
        )
    lines += [
        "int launch(void) {",
        f"  model_start({len(graph.nodes)}u);",
        "  const uint64_t model_begin = nr_cycles();",
    ]
    for i, node in enumerate(graph.nodes):
        lines += [
            "  {",
            f"    model_op_begin({i + 1}u, {json.dumps(node['name'])}, "
            f"{json.dumps(node['case'])}, "
            f"{json.dumps('_mlir_ciface_kernel_' + node['case'])});",
            "    const uint64_t begin = nr_cycles();",
        ]
        for j, (name, shape) in enumerate(zip(node["args"], node["shapes"])):
            pointer = (
                f"(void *)(mobilenet_model_blob + {constant_offsets[name]}u)"
                if name in constant_offsets
                else f"workspace({activation_offsets[name]}u)"
            )
            dimensions = ", ".join(map(str, shape))
            lines.append(
                f"    MemRef{len(shape)} a{j} = make_{len(shape)}({pointer}, {dimensions});"
            )
        lines += [
            f"    _mlir_ciface_kernel_{node['case']}("
            + ", ".join(f"&a{j}" for j in range(len(node["args"])))
            + ");",
            "    model_op_end(nr_cycles() - begin);",
            "  }",
        ]
    lines += [
        f"  float *logits = workspace({activation_offsets[graph.output]}u);",
        f"  const float *golden = (const float *)(mobilenet_model_blob + {golden_offset}u);",
        f"  return model_verify(logits, golden, 1000u, {top1}u, nr_cycles() - model_begin);",
        "}",
        "",
    ]
    (output / "runner.c").write_text("\n".join(lines))
    (output / "model-cases.mk").write_text("MODEL_CASES := " + " ".join(cases) + "\n")
    (output / "model_blob.S").write_text(
        '.section .model_parameters,"a",@progbits\n.balign 64\n'
        ".global mobilenet_model_blob\nmobilenet_model_blob:\n"
        f'.incbin "{output.resolve() / "model-data.bin"}"\n'
        '.section .note.GNU-stack,"",@progbits\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/model")
    parser.add_argument(
        "--checkpoint", type=Path, help="local state_dict for the pretrained model"
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="optional RGB image, using timm validation preprocessing",
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import numpy as np
    import torch
    import timm
    from PIL import Image

    torch.set_num_threads(args.threads)
    model = timm.create_model(MODEL, pretrained=args.checkpoint is None).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=True)
    if args.image:
        from timm.data import create_transform, resolve_model_data_config

        transform = create_transform(
            **resolve_model_data_config(model), is_training=False
        )
        image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
        input_description = {
            "kind": "image",
            "path": str(args.image.resolve()),
            "sha256": sha256(args.image.read_bytes()),
            "preprocessing": resolve_model_data_config(model),
        }
    else:
        # Exact binary fractions, portable across torch/numpy versions and CPUs.
        index = torch.arange(3 * 224 * 224, dtype=torch.int64)
        image = (((index * 17) % 257 - 128).float() / 64.0).reshape(1, 3, 224, 224)
        input_description = {
            "kind": "deterministic-f32",
            "formula": "((i*17)%257-128)/64",
        }
    with torch.inference_mode():
        golden = model(image).detach()
        graph = Graph(model, image).build()
    folded = graph.values[graph.output]
    torch.testing.assert_close(folded, golden, atol=2e-4, rtol=2e-4)
    activation_offsets, arena_bytes = graph.memory_plan()
    assert arena_bytes <= 2 * 1024 * 1024, (
        "activation arena exceeds model linker reservation"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    constant_offsets = {}
    for name, tensor in {**graph.constants, "golden_logits": golden}.items():
        blob += bytes(align64(len(blob)) - len(blob))
        constant_offsets[name] = len(blob)
        blob += tensor.numpy().astype("<f4").tobytes()
    assert len(blob) <= 13 * 1024 * 1024, (
        "parameter blob exceeds model linker reservation"
    )
    (output / "model-data.bin").write_bytes(blob)
    golden.numpy().astype("<f4").tofile(output / "golden-logits.f32")
    top5 = torch.topk(golden[0], 5).indices.tolist()
    write_runner(
        graph,
        output,
        constant_offsets,
        activation_offsets,
        constant_offsets["golden_logits"],
        top5[0],
    )
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().contiguous().numpy().tobytes())
    manifest = {
        "schema_version": 1,
        "model": MODEL,
        "pretrained": True,
        "checkpoint": str(args.checkpoint.resolve())
        if args.checkpoint
        else "timm pretrained download",
        "state_dict_sha256": state_hash.hexdigest(),
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "pretrained_cfg": model.pretrained_cfg,
        "input": input_description,
        "numerics": "float32; inference batch normalization folded into convolution",
        "node_count": len(graph.nodes),
        "unique_kernel_count": len({n["case"] for n in graph.nodes}),
        "workspace_bytes": arena_bytes,
        "blob_bytes": len(blob),
        "blob_sha256": sha256(blob),
        "expected_top5": top5,
        "folded_graph_max_abs_error": float((folded - golden).abs().max()),
        "logits_atol": 0.002,
        "logits_rtol": 0.001,
        "nodes": graph.nodes,
        "constant_offsets": constant_offsets,
        "activation_offsets": activation_offsets,
        "used_cases": sorted({n["case"] for n in graph.nodes}),
        "unused_cases": sorted(
            {x["name"] for x in inventory()} - {n["case"] for n in graph.nodes}
        ),
    }
    (output / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez(output / "host-golden.npz", input=image.numpy(), logits=golden.numpy())
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "model",
                    "node_count",
                    "unique_kernel_count",
                    "workspace_bytes",
                    "blob_bytes",
                    "expected_top5",
                    "folded_graph_max_abs_error",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit a real timm EfficientNet-B3 topology using meta tensors only.

This preparation tool never downloads weights, initializes numerical model
data, builds a board image, or submits FPGA work. It records every logical
operation and the additional compiled connectors required by the current ABI.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402
from export import argument_shapes  # noqa: E402

MODEL = "efficientnet_b3.ra2_in1k"
TIMM_VERSION = "1.0.22"
INPUT_SHAPE = [1, 3, 288, 288]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ShapeGraph:
    def __init__(self, model, torch, cases):
        self.model, self.torch, self.cases = model, torch, cases
        self.values = {"input": {"shape": INPUT_SHAPE, "dtype": "f32"}}
        self.nodes, self.conv_shapes, self.parameters = [], {}, []
        self.missing = []
        self.by_name = {case["name"]: case for case in cases}

    def value(self, name):
        return self.values[name]["shape"]

    def emit(self, name, kind, inputs, shape, case=None, **details):
        if any(value not in self.values for value in inputs):
            raise ValueError(name + ": unbound or forward graph input")
        result = name + ".out"
        if result in self.values:
            raise ValueError("duplicate output " + result)
        shape = list(map(int, shape))
        if any(dim <= 0 for dim in shape):
            raise ValueError(name + ": nonpositive shape")
        node = {
            "index": len(self.nodes),
            "name": name,
            "kind": kind,
            "inputs": list(inputs),
            "input_shapes": [self.value(x) for x in inputs],
            "output": result,
            "shape": shape,
            "dtype": "f32",
            "case": case,
            **details,
        }
        self.nodes.append(node)
        self.values[result] = {"shape": shape, "dtype": "f32"}
        if case is None and kind not in ("reshape_view",):
            self.missing.append(
                {
                    "node": name,
                    "kind": kind,
                    "shape": shape,
                    "reason": details.get("reason", "no existing specialization"),
                }
            )
        return result

    def match_conv(self, ci, co, h, w, kh, kw, sh, sw, ph, pw, groups):
        matches = []
        for case in self.cases:
            s, f = case["constexprs"], case["family"]
            if f == "conv2d_i8":
                valid = (
                    groups == 1
                    and kh == kw == sh == sw == 1
                    and ph == pw == 0
                    and case["shape"] == [1, ci, h, w]
                    and (s["M"], s["N"], s["K"]) == (h * w, co, ci)
                )
            elif f in ("conv2d_f32", "dwconv2d_f32"):
                valid = (
                    (f == "dwconv2d_f32") == (groups == ci == co)
                    and (f == "dwconv2d_f32" or groups == 1)
                    and all(
                        s[key] == value
                        for key, value in {
                            "C": ci,
                            "H": h,
                            "W": w,
                            "KH": kh,
                            "KW": kw,
                            "SH": sh,
                            "SW": sw,
                            "PH": ph,
                            "PW": pw,
                        }.items()
                    )
                    and (f == "dwconv2d_f32" or s["OC"] == co)
                )
            else:
                valid = False
            if valid:
                matches.append(case)
        if len(matches) != 1:
            raise ValueError(
                "convolution shape has %d bindings: %s"
                % (len(matches), (ci, co, h, w, kh, sh, groups))
            )
        return matches[0]

    def activation(self, name, x, module):
        torch = self.torch
        if isinstance(module, torch.nn.Identity):
            return x
        if isinstance(module, torch.nn.SiLU):
            shape = self.value(x)
            case = "silu_" + "x".join(map(str, shape))
            return self.emit(
                name,
                "silu",
                [x],
                shape,
                case if case in self.by_name else None,
                requested_case=case,
                reason="missing SiLU shape specialization",
            )
        if isinstance(module, torch.nn.Sigmoid):
            return self.emit(
                name,
                "sigmoid",
                [x],
                self.value(x),
                reason="99-case inventory has no sigmoid specialization",
            )
        raise ValueError(name + ": unsupported activation " + str(type(module)))

    def conv(self, name, x, conv, bn=None):
        n, ci, h, w = self.value(x)
        co, (kh, kw), (sh, sw), (ph, pw) = (
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
        )
        if n != 1 or conv.dilation != (1, 1) or conv.padding_mode != "zeros":
            raise ValueError(name + ": unsupported convolution metadata")
        oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
        case = self.match_conv(ci, co, h, w, kh, kw, sh, sw, ph, pw, conv.groups)
        self.conv_shapes[name] = {"input": [n, ci, h, w], "output": [n, co, oh, ow]}
        parameter = {
            "node": name,
            "source_weight": name + ".weight",
            "weight_shape": list(conv.weight.shape),
            "source_bias": name + ".bias" if conv.bias is not None else None,
            "bias_shape": [co],
            "values_available": False,
        }
        if bn is not None:
            if bn.training or not isinstance(bn.drop, self.torch.nn.Identity):
                raise ValueError(name + ": batch normalization is not inference-only")
            parameter["batch_norm_fold"] = {
                "module": bn._integration_name,
                "eps": bn.eps,
                "formula": "scale=gamma/sqrt(running_var+eps); Wf=W*scale; bf=(b-running_mean)*scale+beta",
            }
        self.parameters.append(parameter)
        packed = case["family"] == "conv2d_i8"
        if packed:
            route = {
                "status": "requires_compiled_connectors",
                "before": "NCHW f32 -> per-spatial-row symmetric i8 quantize+pack X[HW,C]",
                "weights": "offline folded Weight[OC,C] row quantization plus WeightScale[OC]",
                "existing_entry_shapes": argument_shapes(case),
                "existing_bias_argument": "i32 zero placeholder; raw GEMM does not add bias",
                "after": "i32[HW,OC] * ActivationScale[HW] * WeightScale[OC] + folded f32 Bias[OC] -> NCHW f32",
                "rounding": "half away from zero, symmetric range [-127,127]",
            }
        else:
            route = {
                "status": "existing_public_nchw_w8a8_pipeline",
                "existing_entry_shapes": argument_shapes(case),
                "internal": "im2col -> per-patch/per-weight-row quant -> i8xi8->i32 MMA -> dequant+bias",
            }
        result = self.emit(
            name,
            "depthwise" if conv.groups == ci == co else "conv",
            [x],
            [n, co, oh, ow],
            case["name"],
            family=case["family"],
            weight_shape=list(conv.weight.shape),
            route=route,
        )
        return (
            self.activation(name + ".activation", result, bn.act)
            if bn is not None
            else result
        )

    def gap(self, name, x):
        _, channels, h, w = self.value(x)
        case = "gap_%dx%dx%d" % (h, w, channels)
        if case not in self.by_name:
            raise ValueError(name + ": missing GAP " + case)
        return self.emit(
            name,
            "gap",
            [x],
            [1, channels, 1, 1],
            case,
            descriptor_view="existing output [1,C] aliases the same storage as [1,C,1,1]",
        )

    def se(self, name, x, module):
        y = self.gap(name + ".pool", x)
        y = self.conv(name + ".conv_reduce", y, module.conv_reduce)
        y = self.activation(name + ".act1", y, module.act1)
        y = self.conv(name + ".conv_expand", y, module.conv_expand)
        y = self.activation(name + ".gate", y, module.gate)
        if self.value(x)[1] != self.value(y)[1] or self.value(y)[2:] != [1, 1]:
            raise ValueError(name + ": incompatible SE broadcast")
        return self.emit(
            name + ".multiply",
            "se_broadcast_multiply",
            [x, y],
            self.value(x),
            reason="99-case inventory has no channelwise broadcast multiply specialization",
        )

    def build(self):
        from timm.models._efficientnet_blocks import (
            DepthwiseSeparableConv,
            InvertedResidual,
        )

        m, torch = self.model, self.torch
        for name, module in m.named_modules():
            module._integration_name = name
        x = self.conv("conv_stem", "input", m.conv_stem, m.bn1)
        for stage_id, stage in enumerate(m.blocks):
            for block_id, block in enumerate(stage):
                name, shortcut = "blocks.%d.%d" % (stage_id, block_id), x
                if block.conv_s2d is not None or not isinstance(
                    block.aa, torch.nn.Identity
                ):
                    raise ValueError(
                        name + ": unexpected space-to-depth or antialiasing"
                    )
                if isinstance(block, DepthwiseSeparableConv):
                    x = self.conv(name + ".conv_dw", x, block.conv_dw, block.bn1)
                    x = self.se(name + ".se", x, block.se)
                    x = self.conv(name + ".conv_pw", x, block.conv_pw, block.bn2)
                elif isinstance(block, InvertedResidual):
                    x = self.conv(name + ".conv_pw", x, block.conv_pw, block.bn1)
                    x = self.conv(name + ".conv_dw", x, block.conv_dw, block.bn2)
                    x = self.se(name + ".se", x, block.se)
                    x = self.conv(name + ".conv_pwl", x, block.conv_pwl, block.bn3)
                else:
                    raise ValueError(name + ": unexpected block " + str(type(block)))
                if block.has_skip:
                    if self.value(x) != self.value(shortcut):
                        raise ValueError(name + ": residual shape mismatch")
                    case = "add_" + "x".join(map(str, self.value(x)))
                    if case not in self.by_name:
                        raise ValueError(name + ": missing residual " + case)
                    x = self.emit(
                        name + ".add", "add", [x, shortcut], self.value(x), case
                    )
        x = self.conv("conv_head", x, m.conv_head, m.bn2)
        if m.global_pool.pool_type != "avg" or m.drop_rate != 0.0:
            raise ValueError("unexpected classifier pooling/dropout")
        x = self.gap("global_pool", x)
        x = self.emit(
            "flatten",
            "reshape_view",
            [x],
            [1, 1536],
            view_only=True,
            reason="contiguous 1x1536 view; no data motion",
        )
        self.parameters.append(
            {
                "node": "classifier",
                "source_weight": "classifier.weight",
                "source_bias": "classifier.bias",
                "weight_shape": list(m.classifier.weight.shape),
                "bias_shape": list(m.classifier.bias.shape),
                "values_available": False,
            }
        )
        x = self.emit(
            "classifier",
            "linear_w8a8_chain",
            [x],
            [1, 1000],
            case="matmul_1x1000x1536",
            route={
                "status": "requires_classifier_bias_connector",
                "sequence": [
                    "quantize_1x1536",
                    "explicit_zero_accumulator",
                    "matmul_1x1000x1536",
                    "dequantize_1x1000",
                    "missing_f32_classifier_bias_add",
                ],
                "weights": "offline row-quantize classifier.weight[1000,1536]",
                "accumulator": "existing matmul is C += A@B.T; clear C on every invocation",
                "retired_model_route": "linear_1x1000x1536 F32 is not allowed by user matrix policy",
            },
        )
        self.output = x
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timm-path", type=Path, default=ROOT / "build/model-integration/python-deps"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/model-integration")
    args = parser.parse_args()
    sys.path.insert(0, str(args.timm_path.resolve()))
    import torch
    import timm

    if timm.__version__ != TIMM_VERSION:
        raise ValueError("shape audit requires pinned timm==" + TIMM_VERSION)
    cases = inventory()
    if len(cases) != 99:
        raise ValueError("expected 99 current cases")
    inventory_report = []
    for case in cases:
        shapes = argument_shapes(case)
        if len(shapes) != len(case["arguments"]) or any(
            len(s) != a["rank"] or math.prod(s) <= 0
            for s, a in zip(shapes, case["arguments"])
        ):
            raise ValueError("invalid physical ABI: " + case["name"])
        inventory_report.append(
            {
                "case": case["name"],
                "family": case["family"],
                "argument_shapes": shapes,
                "arguments": case["arguments"],
            }
        )
    with torch.device("meta"):
        model = timm.create_model(MODEL, pretrained=False).eval()
    graph = ShapeGraph(model, torch, cases).build()
    observed = {}
    handles = []

    def hook(name):
        def observe(module, inputs, output):
            observed[name] = {
                "input": list(inputs[0].shape),
                "output": list(output.shape),
            }

        return observe

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        actual = model(torch.empty(INPUT_SHAPE, device="meta"))
    for handle in handles:
        handle.remove()
    if observed != graph.conv_shapes or list(actual.shape) != graph.value(graph.output):
        raise ValueError("manual topology differs from official meta forward")
    used = Counter(node["case"] for node in graph.nodes if node["case"])
    used.update({"quantize_1x1536": 1, "dequantize_1x1000": 1})
    all_names = {case["name"] for case in cases}
    checkpoint_filename = Path(model.pretrained_cfg["url"]).name
    checkpoint_candidates = [
        Path.home() / ".cache/torch/hub/checkpoints" / checkpoint_filename,
        REPO / "build/model-assets" / checkpoint_filename,
    ]
    hf = Path.home() / ".cache/huggingface/hub/models--timm--efficientnet_b3.ra2_in1k"
    if hf.exists():
        checkpoint_candidates.extend(hf.glob("snapshots/*/*.safetensors"))
        checkpoint_candidates.extend(hf.glob("snapshots/*/*.bin"))
    checkpoint_candidates = [str(p) for p in checkpoint_candidates if p.is_file()]
    missing_kinds = Counter(node["kind"] for node in graph.missing)
    source_files = [
        Path(timm.__file__).parent / "models/efficientnet.py",
        Path(timm.__file__).parent / "models/_efficientnet_blocks.py",
    ]
    report = {
        "schema": 1,
        "model": MODEL,
        "mode": "meta_shape_only_no_numerical_values",
        "topology_status": "PASS",
        "production_ready": False,
        "pretrained_weights_loaded": False,
        "numerical_validation_executed": False,
        "board_submitted": False,
        "input": {"shape": INPUT_SHAPE, "dtype": "f32", "values_available": False},
        "output": {
            "shape": graph.value(graph.output),
            "dtype": "f32",
            "value": graph.output,
        },
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "timm_sources": {str(p): digest(p) for p in source_files},
        "pretrained_cfg": model.pretrained_cfg,
        "original_parameter_elements": sum(p.numel() for p in model.parameters()),
        "state_dict_tensors": len(model.state_dict()),
        "block_repeats": [len(stage) for stage in model.blocks],
        "official_meta_forward_conv_shape_matches": len(observed),
        "logical_nodes": len(graph.nodes),
        "node_kind_counts": dict(Counter(n["kind"] for n in graph.nodes)),
        "inventory_cases": len(cases),
        "bound_unique_cases": len(used),
        "used_case_counts": dict(sorted(used.items())),
        "unused_inventory_cases": sorted(all_names - used.keys()),
        "missing_logical_binding_counts": dict(missing_kinds),
        "missing_logical_bindings": graph.missing,
        "packed_pointwise_graph_calls": sum(
            n.get("family") == "conv2d_i8" for n in graph.nodes
        ),
        "packed_pointwise_unique_cases": len(
            {n["case"] for n in graph.nodes if n.get("family") == "conv2d_i8"}
        ),
        "checkpoint_files_found_in_standard_locations": checkpoint_candidates,
        "checkpoint_search_scope": "current-user standard torch/HF caches plus repo build/model-assets; no unbounded system scan",
        "parameters": graph.parameters,
        "nodes": graph.nodes,
        "inventory_abi": inventory_report,
        "limitations": [
            "No checkpoint values, image values, reference logits, or executable graph are produced.",
            "99 specializations are a reusable inventory, not 99 model execution nodes.",
            "Missing sigmoid/SE multiply/SiLU specializations and packed-i8 connectors are explicit blockers.",
            "BatchNorm folding and offline W8A8 weights remain data preparation requests until real weights are loaded.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "shape-graph-manifest.json"
    out.write_text(json.dumps(report, indent=2) + chr(10))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "topology_status",
                    "production_ready",
                    "logical_nodes",
                    "node_kind_counts",
                    "inventory_cases",
                    "bound_unique_cases",
                    "unused_inventory_cases",
                    "missing_logical_binding_counts",
                    "packed_pointwise_graph_calls",
                    "packed_pointwise_unique_cases",
                    "official_meta_forward_conv_shape_matches",
                    "checkpoint_files_found_in_standard_locations",
                )
            },
            indent=2,
        )
    )
    print("manifest: " + str(out))


if __name__ == "__main__":
    main()

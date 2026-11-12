#!/usr/bin/env python3
"""Real-weight CPU references for original, BN-folded, and W8A8 EfficientNet.

The W8A8 interpreter describes the intended compiled graph semantics; it does
not claim to run compiled Triton code. No reference logits are fabricated.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "build/model-integration"
sys.path.insert(0, str(DEFAULT / "python-deps"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
import timm  # noqa: E402

MODEL = "efficientnet_b3.ra2_in1k"
EXPECTED_CHECKPOINT_SHA256 = (
    "279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469"
)
EXPECTED_CONFIG_SHA256 = (
    "d3413ca359c4c5283bbabbba46ef66a265a14031b0b86132de9aa59f84b8d363"
)
HF_REVISION = "0366a75518620e0f2077789202073759f2d61393"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def load_model(assets):
    if timm.__version__ != "1.0.22":
        raise ValueError("reference requires timm==1.0.22")
    checkpoint, config = assets / "model.safetensors", assets / "config.json"
    if (
        sha(checkpoint) != EXPECTED_CHECKPOINT_SHA256
        or sha(config) != EXPECTED_CONFIG_SHA256
    ):
        raise ValueError(
            "checkpoint/config content differs from pinned official assets"
        )
    with torch.device("meta"):
        model = timm.create_model(MODEL, pretrained=False).eval()
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    if any(value.is_meta for value in model.state_dict().values()):
        raise ValueError("checkpoint did not provide all model values")
    return model, json.loads(config.read_text())


def quantize_rows(value):
    value = value.float().contiguous()
    maximum = value.abs().amax(dim=-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    ratio = torch.clamp(value / scale.unsqueeze(-1), -127.0, 127.0)
    base = ratio.to(torch.int32)
    fraction = ratio - base.float()
    adjustment = torch.where(fraction.abs() >= 0.5, torch.where(ratio >= 0, 1, -1), 0)
    return (base + adjustment).to(torch.int8), scale


def folded_parameters(model, graph):
    modules = dict(model.named_modules())
    result = {}
    for item in graph["parameters"]:
        name = item["node"]
        module = modules[name]
        weight = module.weight.detach().float().clone()
        bias = (
            module.bias.detach().float().clone()
            if module.bias is not None
            else torch.zeros(weight.shape[0], dtype=torch.float32)
        )
        if "batch_norm_fold" in item:
            bn = modules[item["batch_norm_fold"]["module"]]
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            weight = weight * scale.reshape([-1] + [1] * (weight.ndim - 1))
            bias = (bias - bn.running_mean.detach()) * scale + bn.bias.detach()
        qw, ws = quantize_rows(weight.reshape(weight.shape[0], -1))
        result[name] = {
            "weight": weight.contiguous(),
            "bias": bias.contiguous(),
            "qweight": qw,
            "weight_scale": ws,
        }
    return result


def integer_mm(left, right_transposed):
    # _int_mm is an exact signed i8 -> i32 CPU implementation. The fallback
    # remains integer arithmetic, never an f32 matmul approximation.
    if hasattr(torch, "_int_mm"):
        return torch._int_mm(left.contiguous(), right_transposed.contiguous())
    return left.to(torch.int32) @ right_transposed.to(torch.int32)


def w8a8_conv(value, module, parameters):
    _, channels, h, w = value.shape
    bias = parameters["bias"]
    qw, ws = parameters["qweight"], parameters["weight_scale"]
    patches = F.unfold(
        value,
        module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )[0]
    kh, kw = module.kernel_size
    oh = (h + 2 * module.padding[0] - kh) // module.stride[0] + 1
    ow = (w + 2 * module.padding[1] - kw) // module.stride[1] + 1
    if module.groups == channels == module.out_channels:
        patch = patches.reshape(channels, kh * kw, oh * ow).transpose(1, 2).contiguous()
        qpatch, ps = quantize_rows(patch)
        acc = (qpatch.to(torch.int32) * qw[:, None, :].to(torch.int32)).sum(
            -1, dtype=torch.int32
        )
        out = acc.float() * ws[:, None]
        out = out * ps + bias[:, None]
        return out.reshape(1, channels, oh, ow).contiguous()
    if module.groups != 1:
        raise ValueError("unsupported grouped convolution")
    patch = patches.transpose(0, 1).contiguous()
    qpatch, ps = quantize_rows(patch)
    acc = integer_mm(qpatch, qw.T)
    out = acc.float() * ws[None, :]
    out = out * ps[:, None] + bias[None, :]
    return out.T.reshape(1, module.out_channels, oh, ow).contiguous()


def interpret(model, graph, image, parameters, quantized):
    modules = dict(model.named_modules())
    uses = Counter(value for node in graph["nodes"] for value in node["inputs"])
    uses[graph["output"]["value"]] += 1
    values, traces = {"input": image}, []
    for node in graph["nodes"]:
        x = [values[value] for value in node["inputs"]]
        kind, name, begin = node["kind"], node["name"], time.monotonic()
        if kind in ("conv", "depthwise"):
            module, p = modules[name], parameters[name]
            out = (
                w8a8_conv(x[0], module, p)
                if quantized
                else F.conv2d(
                    x[0],
                    p["weight"],
                    p["bias"],
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )
            )
        elif kind == "silu":
            out = F.silu(x[0])
        elif kind == "sigmoid":
            out = torch.sigmoid(x[0])
        elif kind == "gap":
            out = x[0].mean((2, 3), keepdim=True)
        elif kind == "se_broadcast_multiply":
            out = x[0] * x[1]
        elif kind == "add":
            out = x[0] + x[1]
        elif kind == "reshape_view":
            out = x[0].reshape(node["shape"])
        elif kind == "linear_w8a8_chain":
            p = parameters[name]
            if quantized:
                qx, xs = quantize_rows(x[0])
                # Existing matmul is additive: model contract requires fresh
                # zero Acc, so the intended mathematical result is this dot.
                acc = integer_mm(qx, p["qweight"].T)
                out = acc.float() * xs[:, None]
                out = out * p["weight_scale"][None, :]
                out = out + p["bias"][None, :]
            else:
                out = F.linear(x[0], p["weight"], p["bias"])
        else:
            raise ValueError("unsupported graph node " + kind)
        out = out.float().contiguous()
        if list(out.shape) != node["shape"] or not torch.isfinite(out).all():
            raise ValueError(name + ": result shape/nonfinite failure")
        values[node["output"]] = out
        traces.append(
            {
                "node": name,
                "case": node["case"],
                "shape": list(out.shape),
                "output_sha256": array_sha(out),
                "min": float(out.min()),
                "max": float(out.max()),
                "host_seconds": time.monotonic() - begin,
            }
        )
        for value in node["inputs"]:
            uses[value] -= 1
            if uses[value] == 0:
                del values[value]
        if len(traces) % 40 == 0:
            print(
                ("w8a8" if quantized else "folded")
                + " "
                + str(len(traces))
                + "/"
                + str(len(graph["nodes"])),
                flush=True,
            )
    return values[graph["output"]["value"]], traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT / "assets")
    parser.add_argument(
        "--graph", type=Path, default=DEFAULT / "shape-graph-manifest.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT / "reference")
    parser.add_argument(
        "--input-f32",
        type=Path,
        help="already preprocessed little-endian f32 [1,3,288,288]",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    graph = json.loads(args.graph.read_text())
    if (
        graph.get("topology_status") != "PASS"
        or graph.get("mode") != "meta_shape_only_no_numerical_values"
    ):
        raise ValueError("a validated shape-only graph is required")
    model, config = load_model(args.assets)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input_f32:
        import numpy as np

        array = np.fromfile(args.input_f32, dtype="<f4")
        if array.size != 3 * 288 * 288:
            raise ValueError("input has incorrect f32 byte count")
        image = torch.from_numpy(array.reshape(1, 3, 288, 288).copy())
        input_record = {
            "kind": "user_preprocessed_f32",
            "path": str(args.input_f32.resolve()),
            "sha256": sha(args.input_f32),
        }
    else:
        index = torch.arange(3 * 288 * 288, dtype=torch.int64)
        image = (((index * 17) % 257 - 128).float() / 64.0).reshape(1, 3, 288, 288)
        input_record = {
            "kind": "declared_deterministic_regression_tensor_not_real_image",
            "formula": "((i*17)%257-128)/64",
            "shape": [1, 3, 288, 288],
        }
    if not torch.isfinite(image).all():
        raise ValueError("nonfinite input")
    input_record["tensor_sha256"] = array_sha(image)
    with torch.inference_mode():
        original = model(image).float().contiguous()
        parameters = folded_parameters(model, graph)
        folded, folded_trace = interpret(model, graph, image, parameters, False)
        folded_error = float((folded - original).abs().max())
        torch.testing.assert_close(folded, original, atol=2e-4, rtol=2e-4)
        quantized, quantized_trace = interpret(model, graph, image, parameters, True)
    tensors = {
        "input.f32": image,
        "original-logits.f32": original,
        "folded-logits.f32": folded,
        "w8a8-logits.f32": quantized,
    }
    for name, tensor in tensors.items():
        tensor.numpy().astype("<f4").tofile(args.output / name)
    packed_parameters = {}
    parameter_layout = []
    for name, values in parameters.items():
        for field, tensor in values.items():
            packed_parameters[name + "." + field] = tensor.numpy()
            parameter_layout.append(
                {
                    "name": name + "." + field,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "sha256": array_sha(tensor),
                }
            )
    import numpy as np

    np.savez(args.output / "folded-and-quantized-parameters.npz", **packed_parameters)
    report = {
        "status": "PASS",
        "scope": "real-weight CPU graph semantics only; not compiled Triton/FPGA evidence",
        "model": MODEL,
        "hf_revision": HF_REVISION,
        "checkpoint_sha256": sha(args.assets / "model.safetensors"),
        "config_sha256": sha(args.assets / "config.json"),
        "graph_sha256": sha(args.graph),
        "reference_source_sha256": sha(__file__),
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "input": input_record,
        "pretrained_cfg_from_pinned_asset": config["pretrained_cfg"],
        "logical_nodes_executed_per_variant": len(graph["nodes"]),
        "bn_fold_vs_original_max_abs_error": folded_error,
        "w8a8_vs_original_max_abs_error": float((quantized - original).abs().max()),
        "w8a8_vs_original_mean_abs_error": float((quantized - original).abs().mean()),
        "original_top5": original[0].topk(5).indices.tolist(),
        "w8a8_top5": quantized[0].topk(5).indices.tolist(),
        "output_hashes": {name: array_sha(tensor) for name, tensor in tensors.items()},
        "numerics": "all matrix products in W8A8 interpreter use symmetric per-patch/per-weight-row i8 quantization and exact integer i32 arithmetic; half away from zero",
        "w8a8_accuracy_acceptance": "not declared; differences are measurements on one regression tensor, not ImageNet accuracy",
        "parameter_layout": parameter_layout,
    }
    (args.output / "reference-report.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    (args.output / "folded-node-trace.json").write_text(
        json.dumps(folded_trace, indent=2) + chr(10)
    )
    (args.output / "w8a8-node-trace.json").write_text(
        json.dumps(quantized_trace, indent=2) + chr(10)
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "logical_nodes_executed_per_variant",
                    "bn_fold_vs_original_max_abs_error",
                    "w8a8_vs_original_max_abs_error",
                    "original_top5",
                    "w8a8_top5",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

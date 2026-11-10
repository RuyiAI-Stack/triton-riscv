#!/usr/bin/env python3
"""Independent ELECTRA inference and concrete W8A8 routing on the host.

With no assets this checks random weights from the saved real configuration.
No emitted BOSCAME model or FPGA execution is claimed by this script.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraForPreTraining

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402


def quantize(x):
    top = x.abs().amax(-1, keepdim=True)
    scale = torch.where(top == 0, torch.ones_like(top), top / 127)
    v = (x / scale).clamp(-127, 127)
    base = v.trunc()
    q = base + torch.where((v - base).abs() >= 0.5, v.sign(), 0)
    return q.to(torch.int32), scale


def forward(
    state,
    ids,
    types,
    mask,
    mode,
    trace,
    snapshots=None,
    initial_hidden=None,
    first_layer=0,
    last_layer=12,
    return_hidden=False,
):
    def call(name):
        trace.append(name)

    def product(a, b, factor=1.0, transpose_rhs=False):
        m, k = a.shape
        n = b.shape[0]
        call(f"quantize_{m}x{k}")
        call(f"quantize_transpose_{k}x{n}" if transpose_rhs else f"quantize_{n}x{k}")
        call(f"matmul_i8_{m}x{n}x{k}")
        call(f"dequantize_{m}x{n}")
        if mode == "fp32":
            return (a @ b.T) * factor
        aq, ar = quantize(a)
        bq, br = quantize(b)
        return ((aq @ bq.T).float() * ar) * br.T * factor

    def linear(x, prefix):
        y = product(x, state[prefix + ".weight"])
        call(f"bias_add_128x{y.shape[-1]}")
        return y + state[prefix + ".bias"]

    def norm(x, prefix):
        call("layernorm_128x768")
        return F.layer_norm(
            x, (768,), state[prefix + ".weight"], state[prefix + ".bias"], 1e-12
        )

    def split(x):
        call("pack_heads_12x128x64")
        return x.reshape(128, 12, 64).transpose(0, 1)

    ep = "electra.embeddings."
    call("embedding_128x768")
    x = (
        state[ep + "word_embeddings.weight"][ids]
        + state[ep + "token_type_embeddings.weight"][types]
    )
    x = x + state[ep + "position_embeddings.weight"][:128]
    x = norm(x, ep + "LayerNorm")
    if snapshots is not None:
        snapshots["embedding"] = x.detach().cpu().clone()
    if initial_hidden is not None:
        x = initial_hidden
    for layer in range(first_layer, last_layer):
        p = f"electra.encoder.layer.{layer}."
        q, k, v = [
            split(linear(x, p + "attention.self." + name))
            for name in ("query", "key", "value")
        ]
        scores = torch.stack([product(q[h], k[h], 0.125) for h in range(12)])
        call("masked_softmax_12x128x128")
        probs = torch.softmax(
            scores.masked_fill(mask[None, None, :] == 0, -torch.inf), -1
        )
        heads = torch.stack(
            [product(probs[h], v[h].T, transpose_rhs=True) for h in range(12)]
        )
        call("unpack_heads_12x128x64")
        context = heads.transpose(0, 1).reshape(128, 768)
        y = linear(context, p + "attention.output.dense")
        call("add_128x768")
        x = norm(y + x, p + "attention.output.LayerNorm")
        y = linear(x, p + "intermediate.dense")
        call("gelu_128x3072")
        y = F.gelu(y, approximate="none")
        y = linear(y, p + "output.dense")
        call("add_128x768")
        x = norm(y + x, p + "output.LayerNorm")
        if snapshots is not None:
            snapshots[f"layer_{layer:02d}"] = x.detach().cpu().clone()
    if return_hidden:
        return x
    y = linear(x, "discriminator_predictions.dense")
    call("gelu_128x768")
    y = F.gelu(y, approximate="none")
    result = linear(y, "discriminator_predictions.dense_prediction").squeeze(-1)
    if snapshots is not None:
        snapshots["logits"] = result.detach().cpu().clone()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "build/model-host.json")
    parser.add_argument(
        "--w8a8",
        action="store_true",
        help="also measure host quantization difference; not an FPGA result",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    if args.assets:
        model = ElectraForPreTraining.from_pretrained(
            args.assets, local_files_only=True, attn_implementation="eager"
        ).eval()
        source = str(args.assets.resolve())
    else:
        config = ElectraConfig.from_json_file(ROOT / "model/config.json")
        config._attn_implementation = "eager"
        model = ElectraForPreTraining(config).eval()
        source = (
            "random weights from saved upstream config; no pretrained accuracy claim"
        )
    required = {
        "hidden_size": 768,
        "embedding_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "position_embedding_type": "absolute",
    }
    for key, want in required.items():
        if getattr(model.config, key) != want:
            raise ValueError(f"unsupported checkpoint config: {key}")
    state = model.state_dict()
    results = []
    with torch.inference_mode():
        for scenario in ("padding-and-segments", "all-valid"):
            ids = torch.arange(128, dtype=torch.long) * 239 % 30522
            ids[-1] = 30521
            mask = (
                (torch.arange(128) < 117).long()
                if scenario == "padding-and-segments"
                else torch.ones(128, dtype=torch.long)
            )
            ids[mask == 0] = 0
            types = (torch.arange(128) >= 64).long()
            trace = []
            got = forward(state, ids, types, mask, "fp32", trace)
            want = model(
                input_ids=ids[None, :],
                token_type_ids=types[None, :],
                attention_mask=mask[None, :],
            ).logits[0]
            # HF may fuse addmm's bias, while the public kernel chain has a
            # separate bias_add. Twelve layers amplify the FP32 rounding gap.
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
            missing = sorted(set(trace) - {c["name"] for c in inventory()})
            if missing:
                raise ValueError("missing routed operators: " + str(missing))
            item = {
                "scenario": scenario,
                "status": "PASS",
                "max_abs_error": float((got - want).abs().max()),
                "output_shape": list(got.shape),
                "operator_calls": len(trace),
            }
            if args.w8a8:
                quantized = forward(state, ids, types, mask, "w8a8", [])
                item["host_w8a8_max_abs_error"] = float((quantized - want).abs().max())
            results.append(item)
    report = {
        "model": "google/electra-base-discriminator",
        "weights": source,
        "validation": "independent host FP32 forward vs Transformers; W8A8 routing coverage",
        "board_executed": False,
        "emitted_model_executed": False,
        "cases": results,
        "operator_inventory": sorted(set(trace)),
        "ordered_calls": trace,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("ordered_calls", "operator_inventory")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

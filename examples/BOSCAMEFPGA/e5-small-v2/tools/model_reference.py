#!/usr/bin/env python3
"""Reviewable fixed-shape BERT forward and W8A8 operator routing on the host.

The host backend evaluates the arithmetic contracts using PyTorch. It does not
execute compiled BOSCAME artifacts. --self-test compares the FP32 decomposition
with Hugging Face's independent BertModel and checks the complete W8A8 schedule.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402

SEQ, HIDDEN, HEADS, DIM, INTERMEDIATE = 128, 384, 12, 32, 1536


def average_pool(hidden, mask):
    """Include every valid token, including CLS/SEP, as in the E5 model card."""
    return (hidden * mask[:, None]).sum(0) / mask.sum().clamp_min(1e-9)


def quantize_rows(x):
    maximum = x.abs().amax(-1)
    scale = torch.where(maximum == 0, 1.0, maximum / 127.0)
    value = (x / scale[:, None]).clamp(-127, 127)
    base = value.trunc().to(torch.int32)
    delta = value - base.to(torch.float32)
    adjustment = torch.where(delta.abs() >= 0.5, torch.where(value >= 0, 1, -1), 0)
    return (base + adjustment).to(torch.int8), scale


class HostContractRuntime:
    def __init__(self):
        self.trace = []
        self.scope = "embedding"

    def record(self, name, **bindings):
        case = describe(name)
        self.trace.append(
            {"scope": self.scope, "case": name, "grid": case["grid"], **bindings}
        )

    def product(self, left, right, transpose_right=False, factor=1.0):
        """right is physical [N,K]; PV starts from logical V[K,N]."""
        m, k = left.shape
        physical = right.T.contiguous() if transpose_right else right
        n = physical.shape[0]
        self.record(f"quantize_{m}x{k}", operand="left")
        lq, ls = quantize_rows(left)
        if transpose_right:
            self.record(f"quantize_transpose_{k}x{n}", operand="right")
        else:
            self.record(f"quantize_{n}x{k}", operand="right")
        rq, rs = quantize_rows(physical)
        self.record(f"matmul_i8_{m}x{n}x{k}")
        # Accumulation is INT32. Input casts prevent PyTorch INT8 output math.
        acc = lq.to(torch.int32) @ rq.to(torch.int32).T
        self.record(f"dequantize_{m}x{n}", factor=factor)
        out = (acc.to(torch.float32) * ls[:, None]) * rs[None, :]
        return out * factor

    def linear(self, x, weight, bias, binding):
        y = self.product(x, weight)
        self.record(f"bias_add_{y.shape[0]}x{y.shape[1]}", weight=binding)
        return y + bias


@torch.inference_mode()
def forward(state, config, ids, mask, token_types=None, w8a8=False, checkpoints=None):
    if tuple(ids.shape) != (SEQ,) or tuple(mask.shape) != (SEQ,):
        raise ValueError("fixed contract requires input_ids/attention_mask [128]")
    if ids.min() < 0 or ids.max() >= config.vocab_size:
        raise ValueError("input_ids are outside the checkpoint vocabulary")
    if not torch.all((mask == 0) | (mask == 1)) or not mask.any():
        raise ValueError("attention_mask must be binary with at least one valid token")
    if token_types is None:
        token_types = torch.zeros_like(ids)
    if (
        tuple(token_types.shape) != (SEQ,)
        or token_types.min() < 0
        or token_types.max() >= 2
    ):
        raise ValueError("token_type_ids must be [128] with values 0 or 1")
    runtime = HostContractRuntime()
    runtime.record(
        "embedding_lookup_128x384", weight="embeddings.word_embeddings.weight"
    )
    word = state["embeddings.word_embeddings.weight"][ids]
    runtime.record("embedding_add_128x384")
    position = state["embeddings.position_embeddings.weight"][:SEQ]
    segment = state["embeddings.token_type_embeddings.weight"][token_types]
    x = (word + position) + segment
    runtime.record("layernorm_128x384", weight="embeddings.LayerNorm")
    x = F.layer_norm(
        x,
        (HIDDEN,),
        state["embeddings.LayerNorm.weight"],
        state["embeddings.LayerNorm.bias"],
        config.layer_norm_eps,
    )
    if checkpoints is not None:
        checkpoints.append(x.clone())

    def dense(value, name):
        if w8a8:
            return runtime.linear(
                value, state[name + ".weight"], state[name + ".bias"], name
            )
        return F.linear(value, state[name + ".weight"], state[name + ".bias"])

    for layer in range(config.num_hidden_layers):
        runtime.scope = f"encoder.layer.{layer}"
        prefix = runtime.scope
        qkv = [
            dense(x, prefix + ".attention.self." + role)
            for role in ("query", "key", "value")
        ]
        for _ in qkv:
            runtime.record("split_heads_128x384")
        q, k, v = [
            value.reshape(SEQ, HEADS, DIM).permute(1, 0, 2).contiguous()
            for value in qkv
        ]
        if w8a8:
            scores = torch.stack(
                [runtime.product(q[h], k[h], factor=DIM**-0.5) for h in range(HEADS)]
            )
        else:
            scores = (q @ k.transpose(-1, -2)) * DIM**-0.5
        runtime.record("attention_mask_12x128x128")
        scores = scores + torch.where(mask.bool(), 0.0, torch.finfo(torch.float32).min)
        runtime.record("softmax_12x128x128")
        probabilities = scores.softmax(-1)
        if w8a8:
            context = torch.stack(
                [
                    runtime.product(probabilities[h], v[h], transpose_right=True)
                    for h in range(HEADS)
                ]
            )
        else:
            context = probabilities @ v
        runtime.record("merge_heads_12x128x32")
        context = context.permute(1, 0, 2).reshape(SEQ, HIDDEN).contiguous()
        attention = dense(context, prefix + ".attention.output.dense")
        runtime.record("add_128x384")
        attention = attention + x
        runtime.record(
            "layernorm_128x384", weight=prefix + ".attention.output.LayerNorm"
        )
        x = F.layer_norm(
            attention,
            (HIDDEN,),
            state[prefix + ".attention.output.LayerNorm.weight"],
            state[prefix + ".attention.output.LayerNorm.bias"],
            config.layer_norm_eps,
        )
        intermediate = dense(x, prefix + ".intermediate.dense")
        runtime.record("gelu_128x1536")
        intermediate = F.gelu(intermediate, approximate="none")
        out = dense(intermediate, prefix + ".output.dense")
        runtime.record("add_128x384")
        out = out + x
        runtime.record("layernorm_128x384", weight=prefix + ".output.LayerNorm")
        x = F.layer_norm(
            out,
            (HIDDEN,),
            state[prefix + ".output.LayerNorm.weight"],
            state[prefix + ".output.LayerNorm.bias"],
            config.layer_norm_eps,
        )
        if checkpoints is not None:
            checkpoints.append(x.clone())
    runtime.scope = "sentence_embedding"
    runtime.record("mean_pool_128x384")
    pooled = average_pool(x, mask)
    runtime.record("l2_normalize_1x384")
    embedding = F.normalize(pooled, p=2, dim=0, eps=1e-12)
    return x, embedding, runtime.trace


def load_model(assets=None):
    if assets:
        config = BertConfig.from_pretrained(assets, local_files_only=True)
        model = BertModel.from_pretrained(
            assets, local_files_only=True, attn_implementation="eager"
        )
    else:
        config = BertConfig.from_json_file(str(ROOT / "model/config.json"))
        config._attn_implementation = "eager"
        torch.manual_seed(20261103)
        model = BertModel(config)
    expected = (HIDDEN, HEADS, INTERMEDIATE, 12, 30522, 512)
    actual = (
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        config.num_hidden_layers,
        config.vocab_size,
        config.max_position_embeddings,
    )
    if (
        actual != expected
        or config.hidden_act != "gelu"
        or config.layer_norm_eps != 1e-12
    ):
        raise ValueError("checkpoint config differs from fixed operator specialization")
    return model.float().eval(), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--assets",
        type=Path,
        help="local pinned checkpoint; otherwise seeded random weights",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/modelcheck")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, config = load_model(args.assets)
    args.output.mkdir(parents=True, exist_ok=True)
    ids = (torch.arange(SEQ, dtype=torch.int64) * 227 + 101) % config.vocab_size
    ids[0], ids[78] = 101, 102
    mask = ((torch.arange(SEQ) < 79) & (torch.arange(SEQ) % 17 != 5)).to(torch.int32)
    ids[mask == 0] = config.pad_token_id
    types = (torch.arange(SEQ) // 53 % 2).to(torch.int64)
    state = model.state_dict()
    with torch.inference_mode():
        reference = model(
            input_ids=ids[None], attention_mask=mask[None], token_type_ids=types[None]
        ).last_hidden_state[0]
        hidden, embedding, _ = forward(state, config, ids, mask, types)
        expected_embedding = F.normalize(average_pool(reference, mask), dim=0)
        torch.testing.assert_close(hidden, reference, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(embedding, expected_embedding, atol=2e-6, rtol=2e-5)
        quantized_hidden, quantized_embedding, trace = forward(
            state, config, ids, mask, types, w8a8=True
        )
        assert (
            torch.isfinite(quantized_hidden).all()
            and torch.isfinite(quantized_embedding).all()
        )
        torch.testing.assert_close(
            quantized_embedding.norm(), torch.tensor(1.0), atol=2e-6, rtol=0
        )
    coverage = Counter(call["case"] for call in trace)
    missing = sorted({case["name"] for case in inventory()} - set(coverage))
    if missing:
        raise AssertionError("unrouted operator cases: " + ", ".join(missing))
    torch.save(
        {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": types,
            "fp32_embedding": embedding,
            "w8a8_embedding": quantized_embedding,
        },
        args.output / "outputs.pt",
    )
    (args.output / "operator_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    report = {
        "status": "PASS",
        "validation": "host PyTorch contract decomposition; no emitted BOSCAME or FPGA execution",
        "weights": str(args.assets)
        if args.assets
        else "seeded random full-size 12-layer BertModel",
        "fp32_hidden_max_abs_error": (hidden - reference).abs().max().item(),
        "fp32_embedding_max_abs_error": (embedding - expected_embedding)
        .abs()
        .max()
        .item(),
        "w8a8_embedding_cosine_to_fp32": F.cosine_similarity(
            embedding, quantized_embedding, dim=0
        ).item(),
        "w8a8_hidden_max_abs_error": (quantized_hidden - hidden).abs().max().item(),
        "operator_calls": len(trace),
        "operator_coverage": dict(sorted(coverage.items())),
        "checkpoint_sha256": __import__("hashlib")
        .sha256((args.assets / "model.safetensors").read_bytes())
        .hexdigest()
        if args.assets
        else None,
        "checkpoint_accuracy_validated": False,
        "board_executed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

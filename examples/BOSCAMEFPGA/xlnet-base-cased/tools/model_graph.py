"""Fixed XLNet content-stream graph and standalone primitive routing."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "model/config.json").read_text())


def linear(m, n, k):
    return [
        f"quantize_{m}x{k}",
        f"quantize_{n}x{k}",
        f"matmul_i8_{m}x{n}x{k}",
        f"dequantize_{m}x{n}",
    ]


def graph():
    hidden_projection = linear(128, 768, 768)
    block = {
        "q_k_v_projection": hidden_projection,
        "relative_projection": linear(256, 768, 768),
        "split_qkv": ["split_heads_12x128x64"],
        "split_relative": ["split_heads_12x256x64"],
        "query_bias_rw_rr_rs": ["head_bias_add_12x128x64"],
        "content_ac_per_head": linear(128, 128, 64),
        "position_bd_per_head": linear(128, 256, 64),
        "relative_shift": ["relative_shift_12x128x128"],
        "segment_ef_per_head": linear(128, 2, 64),
        "segment_gather_optional": ["segment_gather_12x128x128"],
        "score_scale_mask": ["attention_score_12x128x128"],
        "probability": ["softmax_12x128x128"],
        # PV uses V[128,64] -> physical right operand [64,128].
        "pv_per_head": [
            "quantize_128x128",
            "quantize_transpose_128x64",
            "matmul_i8_128x64x128",
            "dequantize_128x64",
        ],
        "merge": ["merge_heads_128x768"],
        "output_projection": hidden_projection,
        "attention_residual_norm": ["add_128x768", "layernorm_128x768"],
        "ffn_intermediate": linear(128, 3072, 768)
        + ["bias_add_128x3072", "gelu_128x3072"],
        "ffn_output": linear(128, 768, 3072)
        + ["bias_add_128x768", "add_128x768", "layernorm_128x768"],
    }
    return {
        "model_id": "xlnet/xlnet-base-cased",
        "architecture": "XLNetLMHeadModel",
        "input_ids": [1, 128],
        "attention_mask": [1, 128],
        "internal_non_target_mask": [128, 128],
        "token_type_ids": [1, 128],
        "options": {
            "training": False,
            "use_mems": False,
            "mems": None,
            "perm_mask": None,
            "target_mapping": None,
        },
        "last_hidden_state": [1, 128, 768],
        "logits_optional": [1, 128, 32000],
        "embedding": ["embedding_128x768"],
        "relative_position_constant": {
            "shape": [256, 768],
            "position_sequence": "128,127,...,-127",
            "layout": "concat(sin,cos), 384 each",
        },
        "num_layers": CONFIG["n_layer"],
        "block": block,
        "lm_head_optional": {
            "tiles": 32,
            "tile_width": 1024,
            "last_valid_columns": 256,
            "kernels": linear(128, 1024, 768) + ["bias_add_128x1024"],
            "weights": "tied transformer.word_embedding.weight; pad last tile to1024; copy first256 only",
        },
        "precision": "FP32 embedding/layout/nonlinear; W8A8 signed-i8 overwrite matrix products; no accuracy claim",
    }


def operator_names():
    g = graph()
    return sorted(
        set(
            g["embedding"]
            + [n for values in g["block"].values() for n in values]
            + g["lm_head_optional"]["kernels"]
        )
    )


if __name__ == "__main__":
    print(json.dumps(graph(), indent=2))

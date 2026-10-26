"""Static ViT-B/16 Triton kernel specializations.

The metadata in each case directory is intentionally small and stable;
it describes the operation shape used by the model graph replacement pass.  No
PyTorch or Triton import is needed to enumerate cases.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cdiv(value, divisor):
    return (value + divisor - 1) // divisor


def power_of_two(value):
    return 1 << (value - 1).bit_length()


def _case(
    name, kind, arguments, constexprs, grid, shape, roles=(), kernel=None
):
    pointer_type = {"f32": "*fp32", "i32": "*i32", "i8": "*i8"}
    return {
        "name": name,
        "family": kind,
        "kernel": kernel or kind,
        "kernel_module": "kernel",
        "kernel_source": str(ROOT / name / "kernel.py"),
        "symbol": "triton_" + name,
        "arguments": arguments,
        "signature": {
            arg["name"]: pointer_type[arg["dtype"]] for arg in arguments
        },
        "constexprs": constexprs,
        "grid": list(grid),
        "directory": str(ROOT / name),
        "launch_source": str(ROOT / name / "launch.c"),
        "shape": list(shape),
        "roles": list(roles),
        "validation": "Independent launch oracle; contiguous tensors",
        **(
            {"accumulation": "overwrite"}
            if kind == "matmul_i8_overwrite"
            else {}
        ),
        **(
            {
                "math_accuracy": {
                    "erf" if kind == "gelu" else "exp": {
                        "domain": "all-f32",
                        "special_values": "ieee",
                        "max_abs_error": 2e-6
                        if kind == "gelu"
                        else 8 * 2.0**-149,
                        "max_rel_error": 0.0 if kind == "gelu" else 2e-6,
                    }
                }
            }
            if kind in ("gelu", "softmax")
            else {}
        ),
    }


def _arg(name, rank, dtype="f32"):
    return {"name": name, "rank": rank, "dtype": dtype}


def describe(name):
    path = ROOT / name / "metadata.json"
    if not path.is_file():
        raise ValueError("unknown case: " + name)
    metadata = json.loads(path.read_text())
    kind = metadata["kind"]
    shape = metadata["shape"]
    if kind in ("matmul_i8", "matmul_i8_overwrite"):
        m, n, k = shape
        return _case(
            name,
            kind,
            [_arg("A", 2, "i8"), _arg("B", 2, "i8"), _arg("C", 2, "i32")],
            {
                "M": m,
                "VALID_M": m,
                "N": n,
                "K": k,
                "BM": 32,
                "BN": 64,
                "BK": 128,
            },
            (cdiv(m, 32), cdiv(n, 64), 1),
            shape,
            roles=metadata.get("roles", ()),
            kernel=kind,
        )
    if kind == "per_token_quantization":
        rows, width = shape
        return _case(
            name,
            kind,
            [_arg("X", 2), _arg("Q", 2, "i8"), _arg("Scale", 1)],
            {"WIDTH": width, "BLOCK": power_of_two(width)},
            (rows, 1, 1),
            shape,
            kernel="quantize",
        )
    if kind == "dequantization":
        rows, width = shape
        return _case(
            name,
            kind,
            [
                _arg("X", 2, "i32"),
                _arg("Row", 1),
                _arg("Column", 1),
                _arg("Out", 2),
            ],
            {"ROWS": rows, "COLS": width, "BLOCK": 128},
            (cdiv(rows * width, 128), 1, 1),
            shape,
            kernel="dequantize",
        )
    if kind == "linear":
        m, n, k = shape
        return _case(
            name,
            kind,
            [_arg("A", 2), _arg("B", 2), _arg("C", 2)],
            {
                "M": m,
                "N": n,
                "K": k,
                "BM": power_of_two(min(m, 16)),
                "BN": 16,
                "BK": power_of_two(min(k, 64)),
            },
            (cdiv(m, power_of_two(min(m, 16))), cdiv(n, 16), 1),
            shape,
        )
    if kind == "patch_embed":
        # The external ABI remains NCHW/OIHW; the composed adapter uses this
        # case to materialize a row-major [196,768] im2col matrix before the
        # quantize -> signed i8 matmul -> dequantize projection.
        return _case(
            name,
            kind,
            [_arg("X", 4), _arg("Weight", 4), _arg("Bias", 1), _arg("Out", 4)],
            {
                "HEIGHT": 224,
                "WIDTH": 224,
                "CHANNELS": 3,
                "PATCH": 16,
                "HIDDEN": 768,
                "GRID": 14,
                "BLOCK": 1024,
            },
            (196, 1, 1),
            shape,
        )
    if kind == "layernorm":
        rows, width = shape
        return _case(
            name,
            kind,
            [_arg("X", 2), _arg("Weight", 1), _arg("Bias", 1), _arg("Out", 2)],
            {
                "ROWS": rows,
                "WIDTH": width,
                "BLOCK": power_of_two(width),
                "EPS": 1.0e-12,
            },
            (rows, 1, 1),
            shape,
        )
    if kind == "gelu":
        rows, width = shape
        count = rows * width
        return _case(
            name,
            kind,
            [_arg("X", 2), _arg("Out", 2)],
            {"COUNT": count, "BLOCK": 128},
            (cdiv(count, 128), 1, 1),
            shape,
        )
    if kind == "bias_add":
        rows, width = shape
        block = 1024
        return _case(
            name,
            kind,
            [_arg("X", 2), _arg("Bias", 1), _arg("Out", 2)],
            {"ROWS": rows, "WIDTH": width, "BLOCK": block},
            (rows, cdiv(width, block), 1),
            shape,
        )
    if kind == "add":
        rows, width = shape
        count = rows * width
        return _case(
            name,
            kind,
            [_arg("X", 2), _arg("Y", 2), _arg("Out", 2)],
            {"COUNT": count, "BLOCK": 128},
            (cdiv(count, 128), 1, 1),
            shape,
        )
    if kind in ("attention_qk", "attention_pv"):
        heads, m, n, k = shape
        return _case(
            name,
            kind,
            [_arg("A", 3), _arg("B", 3), _arg("C", 3)],
            {
                "HEADS": heads,
                "M": m,
                "N": n,
                "K": k,
                "BM": 16,
                "BN": 16,
                "BK": power_of_two(min(k, 64)),
                "SCALE": 1.0 / (k**0.5) if kind == "attention_qk" else 1.0,
            },
            (cdiv(m, 16), cdiv(n, 16), heads),
            shape,
            kernel="attention_dot",
        )
    if kind == "softmax":
        heads, rows, length = shape
        return _case(
            name,
            kind,
            [_arg("X", 3), _arg("Out", 3)],
            {
                "ROWS": heads * rows,
                "LENGTH": length,
                "BLOCK": power_of_two(length),
            },
            (heads * rows, 1, 1),
            shape,
        )
    raise ValueError(f"unsupported ViT metadata kind: {kind}")


def inventory():
    return [
        describe(path.parent.name)
        for path in sorted(ROOT.glob("*/metadata.json"))
    ]


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))

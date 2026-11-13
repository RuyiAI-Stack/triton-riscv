"""Fail-closed production graph gates; independent host baselines bypass none."""

import re

EXPECTED_MATRIX_ROUTES = {
    "dense_w8a8": 73,
    "qk_w8a8": 12,
    "pv_w8a8": 12,
    "patch_w8a8": 1,
}


ATTENTION_PRIMITIVES = {
    "attention_qk": (
        "quantize_197x64",
        "quantize_transpose_64x197",
        "matmul_i8_197x197x64",
        "dequantize_197x197",
        "unpack_heads_12x197x197",
    ),
    "attention_pv": (
        "quantize_197x197",
        "quantize_transpose_197x64",
        "matmul_i8_197x64x197",
        "dequantize_197x64",
        "pack_heads_12x197x197",
    ),
}


def require_attention_primitive_chain(chain, calls):
    """Require explicit graph calls, never an executable attention container."""
    family = chain.get("family")
    if (
        family not in ATTENTION_PRIMITIVES
        or chain.get("implementation") != "model_primitives_w8a8"
        or chain.get("quantization") != "symmetric_i8"
        or "composition" in chain
        or "dependencies" in chain
    ):
        raise ValueError("production attention requires explicit model primitive calls")
    expected = ATTENTION_PRIMITIVES[family]
    selected = [call for call in calls if call.get("attention_id") == chain.get("id")]
    sequence = [(head, case) for head in range(12) for case in expected[:4]]
    layout = (None, expected[4])
    sequence = sequence + [layout] if family == "attention_qk" else [layout] + sequence
    actual = [
        (
            call.get("head"),
            call.get("case", {}).get("name")
            if isinstance(call.get("case"), dict)
            else call.get("case"),
        )
        for call in selected
    ]
    if actual != sequence or chain.get("call_symbols") != [
        call.get("symbol") for call in selected
    ]:
        raise ValueError(
            "attention requires all 12 heads and four ordered primitive calls per head plus layout staging"
        )
    if chain.get("qk_factor") != (0.125 if family == "attention_qk" else 1.0):
        raise ValueError("attention scale must occur once in the QK dequantizer")
    if not all(call.get("void") is True and call.get("views") for call in selected):
        raise ValueError(
            "attention primitive calls require explicit output buffers and descriptor views"
        )
    if any(call.get("family") != "attention_primitive" for call in selected):
        raise ValueError("attention may not dispatch an attention executable")
    buffers = chain.get("buffers", {})
    if (
        set(buffers) != {"q", "t", "row", "column", "acc", "padded", "out"}
        or len(set(buffers.values())) != 7
    ):
        raise ValueError("attention scratch/output roots must be distinct")
    inputs = chain.get("inputs", [])
    qk = family == "attention_qk"
    n, k = (197, 64) if qk else (64, 197)

    def shape3(shape):
        return shape[1:] if len(shape) == 4 and shape[0] == 1 else shape

    if (
        len(inputs) != 2
        or [shape3(shape) for shape in chain.get("input_shapes", [])]
        != [[12, 197, k], [12, k, n]]
        or shape3(chain.get("result_shape", [])) != [12, 197, n]
        or chain.get("result") != buffers["out"]
        or chain.get("workspace_alignment") != 64
    ):
        raise ValueError(
            "attention input/output shape or aligned workspace contract differs"
        )
    for call in selected:
        head, stage = call.get("head"), call.get("stage")
        if stage == "quantize_left":
            roots, offsets, shapes, dtypes = (
                [inputs[0] if qk else buffers["padded"], buffers["q"], buffers["row"]],
                [head * (12608 if qk else 38816), 0, 0],
                [[197, k], [197, k], [197]],
                ["f32", "i8", "f32"],
            )
        elif stage == "quantize_right_transpose":
            roots, offsets, shapes, dtypes = (
                [inputs[1], buffers["t"], buffers["column"]],
                [head * 12608, 0, 0],
                [[k, n], [n, k], [n]],
                ["f32", "i8", "f32"],
            )
        elif stage == "matmul":
            roots, offsets, shapes, dtypes = (
                [buffers["q"], buffers["t"], buffers["acc"]],
                [0, 0, 0],
                [[197, k], [n, k], [197, n]],
                ["i8", "i8", "i32"],
            )
        elif stage == "dequantize":
            roots, offsets, shapes, dtypes = (
                [
                    buffers["acc"],
                    buffers["row"],
                    buffers["column"],
                    buffers["padded"] if qk else buffers["out"],
                ],
                [0, 0, 0, head * (38816 if qk else 12608)],
                [[197, n], [197], [n], [197, n]],
                ["i32", "f32", "f32", "f32"],
            )
        elif stage == ("unpack" if qk else "pack"):
            roots, offsets, shapes, dtypes = (
                (
                    [buffers["padded"], buffers["out"]]
                    if qk
                    else [inputs[0], buffers["padded"]]
                ),
                [0, 0],
                ([[12, 38816], [12, 38809]] if qk else [[12, 38809], [12, 38816]]),
                ["f32", "f32"],
            )
        else:
            raise ValueError("invalid attention primitive stage")
        destinations = (
            [1, 2]
            if stage in ("quantize_left", "quantize_right_transpose")
            else [2]
            if stage == "matmul"
            else [3]
            if stage == "dequantize"
            else [1]
        )
        if call.get("destination_indices") != destinations:
            raise ValueError(
                "attention primitive writable destination contract differs"
            )
        operands = call.get("operands", [])
        operands = [
            value.name if hasattr(value, "name") else value for value in operands
        ]
        expected_views = [
            {"root": root, "element_offset": offset, "shape": shape, "dtype": dtype}
            for root, offset, shape, dtype in zip(roots, offsets, shapes, dtypes)
        ]
        if operands != roots or call.get("views") != expected_views:
            raise ValueError(
                "attention primitive descriptor binding, scale axis or head offset differs"
            )


def require_production_replacement(report):
    if report.get("status") != "PASS" or report.get("production_ready") is not True:
        raise ValueError("a complete production-ready PASS replacement is required")
    if report.get("w8a8_enabled") is not True:
        raise ValueError("FPGA production requires W8A8 matrix products")
    if report.get("w8a8_linear_count") != 73:
        raise ValueError("production requires all 73 W8A8 dense linears")
    if report.get("residual_matrix_products") != []:
        raise ValueError("replacement has residual or unaudited matrix products")
    if report.get("matrix_routes") != EXPECTED_MATRIX_ROUTES:
        raise ValueError(
            "production matrix routes do not cover patch, dense, QK and PV"
        )
    chains = report.get("attention_expansions", [])
    for family in ATTENTION_PRIMITIVES:
        selected = [chain for chain in chains if chain.get("family") == family]
        if len(selected) != 12:
            raise ValueError(
                "production requires 12 explicit primitive expansions per attention family"
            )
        for chain in selected:
            require_attention_primitive_chain(chain, report.get("replaced", []))
    if len({chain.get("id") for chain in chains}) != 24:
        raise ValueError("attention expansion identities must be unique")
    primitive_calls = [
        call
        for call in report.get("replaced", [])
        if call.get("family") == "attention_primitive"
    ]
    if len(primitive_calls) != 24 * 49 or any(
        call.get("attention_id") not in {chain["id"] for chain in chains}
        for call in primitive_calls
    ):
        raise ValueError(
            "production requires exactly 1176 explicit attention primitive/layout calls"
        )
    if any(
        name.startswith(("linear_", "attention_qk_", "attention_pv_"))
        for name in report.get("distinct_cases", [])
    ):
        raise ValueError("retired executable case in production graph")


def require_attention_manifest(report, manifest):
    if (
        manifest.get("schema") != 1
        or manifest.get("implementation") != "model_primitives_w8a8"
        or manifest.get("chains") != report.get("attention_expansions")
    ):
        raise ValueError("attention primitive manifest does not match graph expansions")
    calls = manifest.get("calls", [])
    selected = [
        call
        for call in report.get("replaced", [])
        if call.get("family") == "attention_primitive"
    ]
    keys = (
        "symbol",
        "node",
        "case",
        "family",
        "attention_id",
        "attention_family",
        "stage",
        "head",
        "void",
        "views",
        "destination_indices",
        "operands",
    )
    if [{key: call.get(key) for key in keys} for call in calls] != [
        {key: call.get(key) for key in keys} for call in selected
    ]:
        raise ValueError(
            "attention primitive manifest differs from actual replacement calls"
        )
    for chain in manifest["chains"]:
        require_attention_primitive_chain(chain, calls)


def require_no_matrix_fallback(source):
    """Check before converting named tensor/linalg operations to scalar loops."""
    forbidden = re.search(
        r"(?:tosa\.matmul|linalg\.[a-z_]*matmul[a-z_]*|"
        r"(?:stablehlo|mhlo)\.[a-z_]*dot[a-z_]*|vector\.contract|(?:\b|_)linear_\d|"
        r"(?:\b|_)attention_(?:qk|pv)_\d)",
        source,
    )
    if forbidden:
        raise ValueError(
            "residual matrix multiply / retired FP32 external call: " + forbidden[0]
        )

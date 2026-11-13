"""Independent W8A8 references for the EfficientNet convolution cases.

Only finite FP32 inputs are supported. NaN and infinity do not have a defined
quantization contract in this helper.
"""

import math

import numpy as np

KPAD = 64


def is_quantized_conv(case):
    """Whether the public case binds to the im2col + W8A8 implementation."""
    bindings = {
        "conv2d_f32": (
            "conv2d_im2col_w8a8",
            "im2col_w8a8",
            "quantized_conv_kernels",
        ),
        "dwconv2d_f32": (
            "dwconv2d_im2col_w8a8",
            "im2col_w8a8_depthwise",
            "quantized_conv_kernels",
        ),
    }
    return bindings.get(case.get("family")) == (
        case.get("kernel"),
        case.get("implementation"),
        case.get("kernel_module"),
    )


def _family(case):
    family = case.get("family")
    if family not in ("conv2d_f32", "dwconv2d_f32"):
        raise ValueError("expected an FP32 convolution case")
    if not is_quantized_conv(case):
        raise ValueError("case implementation is not im2col_w8a8")
    return family


def _spec(case):
    _family(case)
    return case["constexprs"]


def _finite_f32(value, name):
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(name + " must contain only finite FP32 values")
    return array


def _logical_k(case):
    spec = _spec(case)
    channels = spec["C"] if case["family"] == "conv2d_f32" else 1
    return channels * spec["KH"] * spec["KW"]


def _spatial_size(case):
    spec = _spec(case)
    return spec["OH"] * spec["OW"]


def _check_arrays(case, arrays):
    spec = _spec(case)
    x = _finite_f32(arrays["X"], "X")
    weight = _finite_f32(arrays["Weight"], "Weight")
    bias = _finite_f32(arrays["Bias"], "Bias")
    expected_x = (1, spec["C"], spec["H"], spec["W"])
    if x.shape != expected_x:
        raise ValueError(f"X shape {x.shape} != {expected_x}")
    if case["family"] == "conv2d_f32":
        expected_weight = (spec["OC"], spec["C"], spec["KH"], spec["KW"])
        output_channels = spec["OC"]
    else:
        expected_weight = (spec["C"], 1, spec["KH"], spec["KW"])
        output_channels = spec["C"]
    if weight.shape != expected_weight:
        raise ValueError(f"Weight shape {weight.shape} != {expected_weight}")
    if bias.shape != (output_channels,):
        raise ValueError(f"Bias shape {bias.shape} != {(output_channels,)}")
    return x, weight, bias


def quantize_rows(rows, kpad=KPAD):
    """Quantize finite ``[rows, logicalK]`` FP32 values to signed int8.

    Each row uses ``max(abs(row))/127`` (or scale 1 for a zero row), FP32
    round-to-nearest division, truncation followed by ties-away-from-zero, and
    saturation to [-127, 127]. Logical K is zero-padded to ``kpad``.
    Returns ``(Q[int8, rows x kpad], scale[float32, rows])``.
    """
    value = _finite_f32(rows, "rows")
    if value.ndim != 2 or value.shape[1] == 0:
        raise ValueError("rows must be a nonempty 2-D array")
    if type(kpad) is not int or not value.shape[1] <= kpad <= KPAD:
        raise ValueError("kpad must satisfy logical K <= kpad <= 64")
    maximum = np.max(np.abs(value), axis=1).astype(np.float32)
    scale = np.where(
        maximum == 0,
        np.float32(1),
        np.divide(maximum, np.float32(127), dtype=np.float32),
    ).astype(np.float32)
    if np.any((maximum != 0) & (scale == 0)):
        raise ValueError("nonzero row scale underflows FP32")
    scaled = np.divide(value, scale[:, None], dtype=np.float32)
    saturated = np.clip(scaled, np.float32(-127), np.float32(127))
    base = saturated.astype(np.int32)
    fraction = np.subtract(saturated, base.astype(np.float32), dtype=np.float32)
    adjustment = np.where(
        np.abs(fraction) >= np.float32(0.5),
        np.where(saturated >= 0, 1, -1),
        0,
    )
    quantized = (base + adjustment).astype(np.int8)
    result = np.zeros((value.shape[0], kpad), dtype=np.int8)
    result[:, : value.shape[1]] = quantized
    return result, scale


def patch_rows(case, arrays, group, start, count=64):
    """Gather an im2col tile using the original NCHW and padding semantics.

    For ordinary convolution ``group`` must be zero and logical K is flattened
    as ``(ic, kh, kw)``. For depthwise convolution ``group`` selects the input
    channel and K is flattened as ``(kh, kw)``. Rows beyond the output image
    and taps outside the input image are zero. The returned K dimension is 64.
    """
    x, _, _ = _check_arrays(case, arrays)
    return _patch_rows_validated(case, x, group, start, count)


def _patch_rows_validated(case, x, group, start, count):
    spec = _spec(case)
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if type(count) is not int or count <= 0 or count > KPAD:
        raise ValueError("count must be in [1, 64]")
    rows = np.zeros((count, KPAD), dtype=np.float32)
    spatial = _spatial_size(case)
    if case["family"] == "conv2d_f32":
        if group != 0:
            raise ValueError("ordinary convolution has only group=0")
        channels = spec["C"]
    else:
        if type(group) is not int or not 0 <= group < spec["C"]:
            raise ValueError("depthwise group must be a valid channel")
        channels = 1

    logical_k = channels * spec["KH"] * spec["KW"]
    for row_index in range(count):
        position = start + row_index
        if position >= spatial:
            continue
        oh, ow = divmod(position, spec["OW"])
        for local_channel in range(channels):
            channel = local_channel if channels > 1 else group
            for kh in range(spec["KH"]):
                ih = oh * spec["SH"] - spec["PH"] + kh
                if not 0 <= ih < spec["H"]:
                    continue
                for kw in range(spec["KW"]):
                    iw = ow * spec["SW"] - spec["PW"] + kw
                    if not 0 <= iw < spec["W"]:
                        continue
                    k = (local_channel * spec["KH"] + kh) * spec["KW"] + kw
                    rows[row_index, k] = x[0, channel, ih, iw]
    if logical_k > KPAD:
        raise ValueError("logical K exceeds the 64-element W8A8 tile")
    return rows


def weight_rows(case, arrays):
    """Return FP32 filter rows ``[OC, logicalK]`` in im2col K order.

    For depthwise convolution OC is C and each row contains that channel's
    spatial filter. Bias is not folded into these rows.
    """
    _, weight, _ = _check_arrays(case, arrays)
    return _weight_rows_validated(case, weight)


def _weight_rows_validated(case, weight):
    _spec(case)
    if case["family"] == "conv2d_f32":
        return np.ascontiguousarray(weight.reshape(weight.shape[0], -1))
    return np.ascontiguousarray(weight[:, 0].reshape(weight.shape[0], -1))


def _dequantize_accumulator(accumulator, weight_scale, patch_scale, bias):
    """Mirror candidate order: int32 -> f32, * weight scale, * patch scale, + bias."""
    value = np.asarray(accumulator, dtype=np.int32).astype(np.float32)
    value = np.multiply(value, weight_scale[:, None], dtype=np.float32)
    value = np.multiply(value, patch_scale[None, :], dtype=np.float32)
    return np.add(value, bias[:, None], dtype=np.float32)


def stage_reference(case, arrays, group=0, start=0, count=64):
    """Return expected im2col, quantization, INT32 dot, and dequant for one tile.

    ``valid_rows`` identifies output positions in the image. The final tile may
    contain extra zero-patch rows; callers should compare only valid outputs.
    For depthwise cases, ``group`` is the output channel. Returned Out and Acc
    have shape ``[1, count]``; ordinary convolution returns ``[OC, count]``.
    """
    x, weight, bias = _check_arrays(case, arrays)
    weights = _weight_rows_validated(case, weight)
    weight_q, weight_scale = quantize_rows(weights, KPAD)
    return _stage_reference_validated(
        case, x, weights, weight_q, weight_scale, bias, group, start, count
    )


def _stage_reference_validated(
    case, x, weights, weight_q, weight_scale, bias, group, start, count
):
    logical_k = _logical_k(case)
    if logical_k > KPAD:
        raise ValueError("logical K exceeds the 64-element W8A8 tile")
    spatial = _spatial_size(case)
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if type(count) is not int or count <= 0 or count > KPAD:
        raise ValueError("count must be in [1, 64]")

    patches = _patch_rows_validated(case, x, group, start, count)
    patch_q, patch_scale = quantize_rows(patches[:, :logical_k], KPAD)
    if case["family"] == "dwconv2d_f32":
        weight_f32 = weights[group : group + 1]
        weight_quantized = weight_q[group : group + 1]
        row_scale = weight_scale[group : group + 1]
        row_bias = bias[group : group + 1]
        accumulator64 = np.sum(
            weight_quantized.astype(np.int64) * patch_q.astype(np.int64),
            axis=1,
            dtype=np.int64,
        )[None, :]
    else:
        weight_f32 = weights
        weight_quantized = weight_q
        row_scale = weight_scale
        row_bias = bias
        accumulator64 = weight_quantized.astype(np.int64) @ patch_q.astype(np.int64).T
    accumulator = accumulator64.astype(np.int32)
    output = _dequantize_accumulator(accumulator, row_scale, patch_scale, row_bias)
    return {
        "patch_f32": patches,
        "patch_q": patch_q,
        "patch_scale": patch_scale,
        "weight_f32": weight_f32,
        "weight_q": weight_quantized,
        "weight_scale": row_scale,
        "Acc": accumulator,
        "Out": output,
        "valid_rows": max(0, min(count, spatial - start)),
    }


def reference(case, arrays):
    """Return the full NCHW W8A8-convolution result as ``{"Out": f32}``."""
    spec = _spec(case)
    x, weight, bias = _check_arrays(case, arrays)
    weights = _weight_rows_validated(case, weight)
    weight_q, weight_scale = quantize_rows(weights, KPAD)
    spatial = _spatial_size(case)
    if case["family"] == "conv2d_f32":
        channels_out = spec["OC"]
        output = np.empty((channels_out, spatial), dtype=np.float32)
        for start in range(0, spatial, KPAD):
            stage = _stage_reference_validated(
                case,
                x,
                weights,
                weight_q,
                weight_scale,
                bias,
                group=0,
                start=start,
                count=min(KPAD, spatial - start),
            )
            valid = stage["valid_rows"]
            output[:, start : start + valid] = stage["Out"][:, :valid]
        result = output.reshape(1, channels_out, spec["OH"], spec["OW"])
    else:
        channels_out = spec["C"]
        output = np.empty((channels_out, spatial), dtype=np.float32)
        for group in range(channels_out):
            for start in range(0, spatial, KPAD):
                stage = _stage_reference_validated(
                    case,
                    x,
                    weights,
                    weight_q,
                    weight_scale,
                    bias,
                    group=group,
                    start=start,
                    count=min(KPAD, spatial - start),
                )
                valid = stage["valid_rows"]
                output[group, start : start + valid] = stage["Out"][0, :valid]
        result = output.reshape(1, channels_out, spec["OH"], spec["OW"])
    return {"Out": result}


def _gamma(operations):
    unit_roundoff = np.finfo(np.float32).eps / 2
    product = operations * unit_roundoff
    if product >= 1:
        return math.inf
    return product / (1 - product)


def quantization_error_metrics(case, arrays, actual, mask=None):
    """Compare W8A8 output with the independent original FP32 convolution.

    The per-output quantization bound is
    ``K*xmax*wmax*(1/127 + 1/(4*127**2))`` plus an FP32 rounding allowance for
    scale/divide/dequant and the original ordered FP32 convolution. A violation
    raises ``AssertionError``. ``max_scaled_abs`` is the maximum absolute error
    divided by ``max(1, abs(reference))``; ``max_bound_fraction`` is the maximum
    absolute-error/per-output-bound ratio used by the pass check.
    """
    spec = _spec(case)
    x, weight, bias = _check_arrays(case, arrays)
    if isinstance(actual, dict):
        actual = actual["Out"]
    got = np.asarray(actual, dtype=np.float32)
    output_channels = spec["C"] if case["family"] == "dwconv2d_f32" else spec["OC"]
    expected_shape = (1, output_channels, spec["OH"], spec["OW"])
    if got.shape != expected_shape:
        raise ValueError(f"actual Out shape {got.shape} != {expected_shape}")
    selected = (
        np.ones(expected_shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if selected.shape != expected_shape or not np.any(selected):
        raise ValueError("mask must select at least one output with the expected shape")
    if not np.all(np.isfinite(got[selected])):
        raise ValueError("selected actual Out must contain only finite FP32 values")

    # This is the original FP32 convolution, independent of the W8A8 path.
    from oracles import _conv2d_nchw, _dwconv2d_nchw

    validated_arrays = dict(arrays)
    validated_arrays.update(X=x, Weight=weight, Bias=bias)
    reference_f32 = (
        _conv2d_nchw(case, validated_arrays)
        if case["family"] == "conv2d_f32"
        else _dwconv2d_nchw(case, validated_arrays)
    )
    error = np.zeros(expected_shape, dtype=np.float64)
    error[selected] = np.abs(
        got[selected].astype(np.float64) - reference_f32[selected].astype(np.float64)
    )
    bound = np.empty(expected_shape, dtype=np.float64)
    logical_k = _logical_k(case)
    factor = 1 / 127 + 1 / (4 * 127**2)
    original_rounding = _gamma(2 * logical_k + 1)
    dequant_rounding = _gamma(4)
    scale_rounding = 8 * (np.finfo(np.float32).eps / 2)
    spatial = _spatial_size(case)
    weights = _weight_rows_validated(case, weight)

    if case["family"] == "conv2d_f32":
        xmax = np.empty(spatial, dtype=np.float64)
        for start in range(0, spatial, KPAD):
            count = min(KPAD, spatial - start)
            patches = _patch_rows_validated(case, x, 0, start, count)
            xmax[start : start + count] = np.max(np.abs(patches[:, :logical_k]), axis=1)
        wmax = np.max(np.abs(weights), axis=1).astype(np.float64)
        magnitude = logical_k * wmax[:, None] * xmax[None, :]
        quant_bound = magnitude * factor
        bias_magnitude = np.abs(bias.astype(np.float64))[:, None]
        rounding = (
            original_rounding * (magnitude + bias_magnitude)
            + dequant_rounding * (magnitude + quant_bound + bias_magnitude)
            + scale_rounding * magnitude
        )
        bound[0] = (quant_bound + rounding).reshape(spec["OC"], spec["OH"], spec["OW"])
    else:
        for group in range(spec["C"]):
            wmax = float(np.max(np.abs(weights[group])))
            bias_magnitude = abs(float(bias[group]))
            for start in range(0, spatial, KPAD):
                count = min(KPAD, spatial - start)
                patches = _patch_rows_validated(case, x, group, start, count)
                xmax = np.max(np.abs(patches[:, :logical_k]), axis=1).astype(np.float64)
                magnitude = logical_k * wmax * xmax
                quant_bound = magnitude * factor
                rounding = (
                    original_rounding * (magnitude + bias_magnitude)
                    + dequant_rounding * (magnitude + quant_bound + bias_magnitude)
                    + scale_rounding * magnitude
                )
                bound[0, group, :, :].reshape(-1)[start : start + count] = (
                    quant_bound + rounding
                )

    scaled = error / np.maximum(1.0, np.abs(reference_f32.astype(np.float64)))
    fraction = np.zeros_like(error)
    positive_bound = bound > 0
    fraction[positive_bound] = error[positive_bound] / bound[positive_bound]
    fraction[(~positive_bound) & (error > 0)] = np.inf
    selected_error = error[selected]
    selected_bound = bound[selected]
    passed = bool(np.all(selected_error <= selected_bound))
    result = {
        "reference": "oracles._conv2d_nchw/_dwconv2d_nchw, original FP32 semantics",
        "bound_formula": "K*xmax*wmax*(1/127 + 1/(4*127^2)) + per-output FP32 rounding safety",
        "max_abs": float(np.max(selected_error, initial=0)),
        "rmse": float(np.sqrt(np.mean(np.square(selected_error))))
        if selected_error.size
        else 0.0,
        "max_scaled_abs": float(np.max(scaled[selected], initial=0)),
        "bound": float(np.max(selected_bound, initial=0)),
        "max_bound": float(np.max(selected_bound, initial=0)),
        "max_bound_fraction": float(np.max(fraction[selected], initial=0)),
        "bound_pass": passed,
        "finite_outputs": int(np.count_nonzero(selected)),
    }
    if not passed:
        raise AssertionError(
            "W8A8 output exceeds its per-output quantization bound: "
            f"max_abs={result['max_abs']}, max_bound={result['max_bound']}, "
            f"max_bound_fraction={result['max_bound_fraction']}"
        )
    return result

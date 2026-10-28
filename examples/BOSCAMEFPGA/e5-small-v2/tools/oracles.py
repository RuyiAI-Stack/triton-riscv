"""Independent NumPy references for the ported ViT operator semantics."""

import math

import numpy as np

DTYPES = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"), "i8": np.dtype("i1")}
NONLINEAR_TOLERANCE = 2e-5


def gelu_boundary_inputs():
    """Finite boundaries and IEEE values, independent of target coefficients."""
    invsqrt2 = np.float32(0.7071067811865475)
    joins = np.arange(0.5, 4.5, 0.5, dtype=np.float32) / invsqrt2
    points = np.concatenate(
        (
            np.linspace(-12, 12, 2049, dtype=np.float32),
            joins,
            -joins,
            np.array(
                [
                    -np.inf,
                    np.inf,
                    np.nan,
                    -0.0,
                    0.0,
                    -np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    -np.finfo(np.float32).tiny,
                    np.finfo(np.float32).tiny,
                    -np.nextafter(np.float32(0), np.float32(1)),
                    np.nextafter(np.float32(0), np.float32(1)),
                ],
                dtype=np.float32,
            ),
        )
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return np.concatenate(
            (
                points,
                np.nextafter(points, -np.inf),
                np.nextafter(points, np.inf),
            )
        )


def _nonlinear_stress(case, arrays):
    if case["family"] == "gelu":
        arrays["X"][...] = np.resize(gelu_boundary_inputs(), arrays["X"].shape)
    elif case["family"] == "softmax":
        length = case["constexprs"]["LENGTH"]
        rows = arrays["X"].reshape(-1, length)
        limits = np.finfo(np.float32)
        boundary = np.array(
            [
                -103.97208404541016,
                -103.2789306640625,
                -87.3365478515625,
                -64 * math.log(2),
                -math.log(2),
                -0.5 * math.log(2),
                0,
            ],
            dtype=np.float32,
        )
        for index, row in enumerate(rows):
            kind = index % 16
            if kind < 2:
                row.fill(-0.0 if kind else 0.0)
            elif kind == 2:
                row[:] = np.linspace(-200, 0, length, dtype=np.float32)
            elif kind in (3, 4):
                row[:] = np.linspace(-100, 0, length, dtype=np.float32)
                row += np.float32(1000 if kind == 3 else -1000)
            elif kind == 5:
                row.fill(-120)
                row[-1] = 0
            elif kind == 6:
                row[:] = np.resize(np.array([-np.inf, 0], np.float32), length)
            elif kind in (7, 8):
                row.fill(0)
                row[index % length] = np.nan if kind == 7 else np.inf
            elif kind in (9, 10):
                row.fill(np.inf if kind == 9 else -np.inf)
            elif kind == 11:
                row[:] = np.resize(
                    np.array([-limits.max, limits.max], np.float32), length
                )
            elif kind == 12:
                row[:] = np.resize(
                    np.array([-limits.tiny, 0, limits.tiny], np.float32), length
                )
            elif kind in (13, 14):
                row[:] = np.resize(
                    np.nextafter(boundary, -np.inf if kind == 13 else np.inf),
                    length,
                )
                row[-1] = 0
            else:
                row[:] = np.resize(
                    np.array([-0.0, 0.0, -np.inf], np.float32), length
                )


def output_names(case):
    """Writable entry arguments; a matmul's C is also an input."""
    family = case["family"]
    if family in (
        "linear",
        "matmul_i8",
        "matmul_i8_overwrite",
    ) or family.startswith("attention_"):
        return ("C",)
    if family == "per_token_quantization":
        return ("Q", "Scale")
    return ("Out",)


def make_inputs(case, shapes, stress=False):
    """Deterministic inputs, including zero rows and signed integer extremes."""
    rng = np.random.default_rng(107 if stress else 71)
    outputs = output_names(case)
    arrays = {}
    for argument, shape in zip(case["arguments"], shapes):
        name, dtype = argument["name"], DTYPES[argument["dtype"]]
        if dtype == DTYPES["i8"]:
            if stress:
                choices = np.array(
                    [-128, -127, -1, 0, 1, 126, 127], dtype=dtype
                )
                value = rng.choice(choices, size=shape)
            else:
                value = rng.integers(-15, 16, size=shape, dtype=dtype)
        elif dtype == DTYPES["i32"]:
            if stress:
                choices = np.array(
                    [-2147483648, -2147483600, -1, 1, 2147483600, 2147483647],
                    dtype=dtype,
                )
                value = rng.choice(choices, size=shape)
            else:
                value = rng.integers(-100, 101, size=shape, dtype=dtype)
        elif name in ("Row", "Column", "Scale"):
            value = rng.uniform(0.002, 0.02, size=shape).astype(dtype)
        elif name == "Weight":
            value = rng.uniform(-1, 2, size=shape).astype(dtype)
        elif name == "Bias":
            value = rng.uniform(-0.5, 0.5, size=shape).astype(dtype)
        else:
            value = rng.normal(0.2, 1.1, size=shape).astype(dtype)
        if name in outputs and name != "C":
            # A visible sentinel proves every required element was written.
            value = np.full(
                shape, 0x55 if dtype.kind == "i" else np.nan, dtype=dtype
            )
        arrays[name] = np.ascontiguousarray(value, dtype=dtype)
    if case["family"] == "per_token_quantization":
        arrays["X"][0] = 0.0
        if arrays["X"].shape[0] > 1:
            width = arrays["X"].shape[-1]
            ties = np.resize(
                np.array(
                    [-127, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 127],
                    dtype=np.float32,
                ),
                width,
            )
            arrays["X"][1] = ties
    if stress and case["family"] == "attention_pv":
        # Distinguish block-local accumulation from reassociation across K
        # blocks. The unit term is rounded away inside the second block.
        a, b = arrays["A"], arrays["B"]
        block = case["constexprs"]["BK"]
        a.fill(0)
        signs = np.where(
            (np.indices(a.shape[:-1]).sum(axis=0) % 2) == 0, 1, -1
        ).astype(np.float32)
        if a.shape[-1] > block + 1:
            a[..., 0] = signs * np.float32(2**25)
            a[..., block] = -signs * np.float32(2**25)
            a[..., block + 1] = signs
        if a.shape[-1] > 2 * block:
            a[..., 2 * block] = signs * np.float32(0.125)
        tail = a.shape[-1] % block
        if tail:
            a[..., -tail:] = signs[..., None] * np.resize(
                np.array([2, -4, 0.5, 1.25, -0.25], dtype=np.float32), tail
            )
        columns = np.arange(b.shape[-1])
        scales = np.ldexp(
            np.where(columns % 2, np.float32(-1), np.float32(1)),
            columns % 5 - 2,
        )
        b[...] = scales
        arrays["C"].fill(np.nan)
    if stress:
        _nonlinear_stress(case, arrays)
    return arrays


def nonlinear_reference64(case, arrays):
    """Independent double-precision mathematical reference, not target recipes.

    GELU uses erfc to retain its small negative tail. Nonfinite behavior follows
    the original erf expression, including GELU(-Inf)=NaN. Softmax uses a stable
    shift and math.fsum, without the emitted FP32 reduction's rounding.
    """
    x = arrays["X"]
    if case["family"] == "gelu":

        def gelu(value):
            value = float(value)
            if value == -math.inf:
                return math.nan
            return 0.5 * value * math.erfc(-value / math.sqrt(2))

        return {
            "Out": np.fromiter(
                (gelu(v) for v in x.flat), np.float64, x.size
            ).reshape(x.shape)
        }
    if case["family"] != "softmax":
        raise ValueError("double nonlinear reference requires GELU or Softmax")
    length = case["constexprs"]["LENGTH"]
    result = np.empty(x.shape, np.float64)
    for row, out in zip(x.reshape(-1, length), result.reshape(-1, length)):
        values = [float(v) for v in row]
        maximum = max(values)
        if (
            any(math.isnan(v) or v == math.inf for v in values)
            or maximum == -math.inf
        ):
            out.fill(np.nan)
            continue
        exponential = [math.exp(v - maximum) for v in values]
        total = math.fsum(exponential)
        out[:] = [v / total for v in exponential]
    return {"Out": result}


def nonlinear_reference_mp(case, arrays, dps=80):
    """Arbitrary-precision sampling oracle; requires mpmath only when invoked."""
    import mpmath as mp

    x = arrays["X"]
    result = np.empty(x.shape, np.float64)
    with mp.workdps(dps):
        if case["family"] == "gelu":
            for index, raw in enumerate(x.flat):
                value = float(raw)
                if not math.isfinite(value):
                    answer = math.inf if value == math.inf else math.nan
                elif value == 0:
                    answer = value
                elif value < -40:
                    # erfc(t) < exp(-t*t)/(t*sqrt(pi)); this entire tail
                    # rounds to zero even in the returned binary64 format.
                    answer = -0.0
                else:
                    number = mp.mpf(value)
                    answer = float(number * mp.erfc(-number / mp.sqrt(2)) / 2)
                result.flat[index] = answer
        elif case["family"] == "softmax":
            length = case["constexprs"]["LENGTH"]
            for row, out in zip(
                x.reshape(-1, length), result.reshape(-1, length)
            ):
                values = [float(v) for v in row]
                maximum = max(values)
                if (
                    any(math.isnan(v) or v == math.inf for v in values)
                    or maximum == -math.inf
                ):
                    out.fill(np.nan)
                    continue
                terms = [mp.exp(mp.mpf(v) - maximum) for v in values]
                total = mp.fsum(terms)
                out[:] = [float(v / total) for v in terms]
        else:
            raise ValueError("high-precision oracle requires GELU or Softmax")
    return {"Out": result}


def _matmul_sequential(a, b):
    """One f32 multiply and one f32 add per K, with no reassociation/FMA."""
    result = np.zeros(a.shape[:-1] + (b.shape[-1],), dtype=np.float32)
    for k in range(a.shape[-1]):
        product = np.multiply(
            a[..., :, k, None], b[..., k, None, :], dtype=np.float32
        )
        np.add(result, product, out=result)
    return result


def _matmul_blocked_sequential(a, b, block):
    """Original floating dot expansion: zero product, then add per K block.

    Retain the entire declared block, including positive-zero padding in the
    last one. Each multiply and addition rounds independently to f32; neither
    the block reductions nor the additions between blocks are reassociated.
    """
    if block <= 0:
        raise ValueError("matmul reduction block must be positive")
    result = np.zeros(a.shape[:-1] + (b.shape[-1],), dtype=np.float32)
    for start in range(0, a.shape[-1], block):
        valid = min(block, a.shape[-1] - start)
        left = np.zeros(a.shape[:-1] + (block,), dtype=np.float32)
        right = np.zeros(b.shape[:-2] + (block, b.shape[-1]), dtype=np.float32)
        left[..., :valid] = a[..., start : start + valid]
        right[..., :valid, :] = b[..., start : start + valid, :]
        product = _matmul_sequential(left, right)
        np.add(result, product, out=result)
    return result


def _conv2d_nchw(case, arrays):
    """NCHW convolution in the kernel's ic/kh/kw reduction order."""
    spec = case["constexprs"]
    x = arrays["X"][0]
    weight = arrays["Weight"]
    oh, ow = spec["OH"], spec["OW"]
    acc = np.zeros((spec["OC"], oh, ow), dtype=np.float32)
    for ic in range(spec["C"]):
        for kh in range(spec["KH"]):
            ih = np.arange(oh, dtype=np.int64) * spec["SH"] - spec["PH"] + kh
            for kw in range(spec["KW"]):
                iw = np.arange(ow, dtype=np.int64) * spec["SW"] - spec["PW"] + kw
                valid_h = (ih >= 0) & (ih < spec["H"])
                valid_w = (iw >= 0) & (iw < spec["W"])
                gathered = np.zeros((oh, ow), dtype=np.float32)
                if np.any(valid_h) and np.any(valid_w):
                    rows = np.clip(ih, 0, spec["H"] - 1)
                    cols = np.clip(iw, 0, spec["W"] - 1)
                    gathered = x[ic][rows[:, None], cols[None, :]].astype(np.float32)
                    gathered *= valid_h[:, None] & valid_w[None, :]
                acc = np.add(
                    acc,
                    gathered[None, :, :] * weight[:, ic, kh, kw, None, None],
                    dtype=np.float32,
                )
    acc = np.add(acc, arrays["Bias"][:, None, None], dtype=np.float32)
    return acc.reshape(1, spec["OC"], oh, ow)


def _dwconv2d_nchw(case, arrays):
    """Depthwise NCHW convolution, one output channel per input channel."""
    spec = case["constexprs"]
    x = arrays["X"][0]
    weight = arrays["Weight"][:, 0]
    oh, ow = spec["OH"], spec["OW"]
    acc = np.zeros((spec["C"], oh, ow), dtype=np.float32)
    for kh in range(spec["KH"]):
        ih = np.arange(oh, dtype=np.int64) * spec["SH"] - spec["PH"] + kh
        valid_h = (ih >= 0) & (ih < spec["H"])
        rows = np.clip(ih, 0, spec["H"] - 1)
        for kw in range(spec["KW"]):
            iw = np.arange(ow, dtype=np.int64) * spec["SW"] - spec["PW"] + kw
            valid_w = (iw >= 0) & (iw < spec["W"])
            cols = np.clip(iw, 0, spec["W"] - 1)
            gathered = x[:, rows[:, None], cols[None, :]].astype(np.float32)
            gathered *= valid_h[None, :, None] & valid_w[None, None, :]
            acc = np.add(
                acc,
                gathered * weight[:, kh, kw, None, None],
                dtype=np.float32,
            )
    acc = np.add(acc, arrays["Bias"][:, None, None], dtype=np.float32)
    return acc.reshape(1, spec["C"], oh, ow)


def reference(case, arrays):
    """Return references only for the writable arguments of this case."""
    family = case["family"]
    if family in ("matmul_i8", "matmul_i8_overwrite"):
        a, b, c = (arrays[name] for name in ("A", "B", "C"))
        return {
            "C": (
                (c.astype(np.int64) if family == "matmul_i8" else 0)
                + a.astype(np.int64) @ b.astype(np.int64).T
            ).astype("<i4")
        }
    if family == "linear":
        product = _matmul_sequential(arrays["A"], arrays["B"].T)
        return {"C": np.add(product, arrays["C"], dtype=np.float32)}
    if family == "linear_f32":
        # Same BK tiles as tl.dot. Each tile is still a sequential f32 dot;
        # the kernel may contract a tile with FMA, so compare() keeps a tolerance.
        product = _matmul_blocked_sequential(
            arrays["A"], arrays["W"].T, case["constexprs"]["BK"]
        )
        return {
            "Out": np.add(product, arrays["Bias"][None, :], dtype=np.float32)
        }
    if family == "conv2d_f32":
        return {"Out": _conv2d_nchw(case, arrays)}
    if family == "dwconv2d_f32":
        return {"Out": _dwconv2d_nchw(case, arrays)}
    if family == "gap":
        spatial = case["constexprs"]["SPATIAL"]
        channels = arrays["X"].reshape(arrays["X"].shape[1], spatial)
        acc = np.zeros(channels.shape[0], dtype=np.float32)
        for pixel in range(spatial):
            acc = np.add(acc, channels[:, pixel], dtype=np.float32)
        return {
            "Out": np.divide(acc, np.float32(spatial), dtype=np.float32).reshape(
                1, -1
            )
        }
    if family == "silu":
        x = arrays["X"]
        denom = np.add(
            np.float32(1), np.exp(np.negative(x), dtype=np.float32), dtype=np.float32
        )
        return {"Out": np.divide(x, denom, dtype=np.float32)}
    if family.startswith("attention_"):
        product = (
            _matmul_blocked_sequential(
                arrays["A"], arrays["B"], case["constexprs"]["BK"]
            )
            if family == "attention_pv"
            else _matmul_sequential(arrays["A"], arrays["B"])
        )
        return {
            "C": np.multiply(
                product,
                np.float32(case["constexprs"]["SCALE"]),
                dtype=np.float32,
            )
        }
    if family == "add":
        return {"Out": np.add(arrays["X"], arrays["Y"], dtype=np.float32)}
    if family == "bias_add":
        return {"Out": np.add(arrays["X"], arrays["Bias"], dtype=np.float32)}
    if family == "layernorm":
        rows, width = case["shape"]
        block = case["constexprs"]["BLOCK"]
        padded = np.pad(arrays["X"], ((0, 0), (0, block - width)))
        mean = np.divide(
            np.cumsum(padded, axis=1, dtype=np.float32)[:, -1],
            np.float32(width),
            dtype=np.float32,
        )
        centered = np.subtract(padded, mean[:, None], dtype=np.float32)
        centered[:, width:] = np.float32(0)
        squared = np.multiply(centered, centered, dtype=np.float32)
        variance = np.divide(
            np.cumsum(squared, axis=1, dtype=np.float32)[:, -1],
            np.float32(width),
            dtype=np.float32,
        )
        shifted = np.add(
            variance, np.float32(case["constexprs"]["EPS"]), dtype=np.float32
        )
        inverse = np.divide(
            np.float32(1), np.sqrt(shifted, dtype=np.float32), dtype=np.float32
        )
        normalized = np.multiply(
            centered[:, :width], inverse[:, None], dtype=np.float32
        )
        scaled = np.multiply(normalized, arrays["Weight"], dtype=np.float32)
        return {"Out": np.add(scaled, arrays["Bias"], dtype=np.float32)}
    if family == "per_token_quantization":
        # The exported EfficientNet kernel adds ±0.5 and truncates toward zero.
        x = arrays["X"]
        maximum = np.max(np.abs(x), axis=-1).astype(np.float32)
        scale = np.where(
            maximum == 0,
            np.float32(1),
            np.divide(maximum, np.float32(127), dtype=np.float32),
        ).astype(np.float32)
        value = np.divide(x, scale[:, None], dtype=np.float32)
        half = np.where(value >= 0, np.float32(0.5), np.float32(-0.5))
        rounded = np.add(value, half, dtype=np.float32)
        saturated = np.clip(rounded, np.float32(-127), np.float32(127)).astype(
            np.float32
        )
        return {
            "Q": saturated.astype(np.int32).astype(np.int8),
            "Scale": scale,
        }
    if family == "dequantization":
        first = np.multiply(
            arrays["X"].astype(np.float32),
            arrays["Row"][:, None],
            dtype=np.float32,
        )
        return {
            "Out": np.multiply(
                first, arrays["Column"][None, :], dtype=np.float32
            )
        }
    if family == "gelu":
        x = arrays["X"]
        erf_input = np.multiply(
            x, np.float32(0.7071067811865475), dtype=np.float32
        )
        erf = np.fromiter(
            (math.erf(float(v)) for v in erf_input.flat),
            dtype=np.float32,
            count=x.size,
        ).reshape(x.shape)
        factor = np.add(np.float32(1), erf, dtype=np.float32)
        half_x = np.multiply(np.float32(0.5), x, dtype=np.float32)
        return {"Out": np.multiply(half_x, factor, dtype=np.float32)}
    if family == "softmax":
        length, block = (
            case["constexprs"]["LENGTH"],
            case["constexprs"]["BLOCK"],
        )
        x = arrays["X"].reshape(-1, length)
        padded = np.pad(
            x, ((0, 0), (0, block - length)), constant_values=-np.inf
        )
        centered = np.subtract(
            padded, np.max(padded, axis=1)[:, None], dtype=np.float32
        )
        exponential = np.exp(centered, dtype=np.float32)
        total = np.cumsum(exponential, axis=1, dtype=np.float32)[:, -1]
        result = np.divide(
            exponential[:, :length], total[:, None], dtype=np.float32
        )
        return {"Out": result.reshape(arrays["Out"].shape)}
    raise ValueError("no complete independent oracle for " + family)


def compare(case, actual, expected):
    """Exact arithmetic cases compare bits; nonlinear f32 keeps tolerances."""
    max_abs = 0.0
    tolerant = case["family"] in (
        "layernorm",
        "gelu",
        "softmax",
        "silu",
        "linear_f32",
        "conv2d_f32",
        "dwconv2d_f32",
        "gap",
    )
    for name, want in expected.items():
        got = actual[name]
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(f"{name}: output shape/dtype differs")
        finite = np.isfinite(got) & np.isfinite(want)
        delta = np.abs(
            got[finite].astype(np.float64) - want[finite].astype(np.float64)
        )
        max_abs = max(max_abs, float(np.max(delta, initial=0)))
        if tolerant:
            np.testing.assert_allclose(
                got,
                want,
                rtol=NONLINEAR_TOLERANCE,
                atol=NONLINEAR_TOLERANCE,
                err_msg=name,
            )
            if case["family"] in ("gelu", "softmax"):
                zeros = (want == 0) & (got == 0)
                np.testing.assert_array_equal(
                    np.signbit(got[zeros]),
                    np.signbit(want[zeros]),
                    err_msg=name + ": signed zero differs",
                )
        else:
            np.testing.assert_array_equal(
                got.view(np.uint8),
                want.view(np.uint8),
                err_msg=f"{name}: bitwise comparison",
            )
    return max_abs


def nonlinear_error_metrics(actual, expected):
    """Apply the existing 2e-5 tolerance also to an independent real reference."""
    got, want = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    np.testing.assert_allclose(
        got,
        want,
        rtol=NONLINEAR_TOLERANCE,
        atol=NONLINEAR_TOLERANCE,
        err_msg="independent mathematical reference",
    )
    finite = np.isfinite(got) & np.isfinite(want)
    absolute = np.abs(got[finite] - want[finite])
    budget = NONLINEAR_TOLERANCE * (1 + np.abs(want[finite]))
    return {
        "reference": "independent binary64 math.erfc/exp and math.fsum; sampled against 80-digit mpmath separately",
        "rtol": NONLINEAR_TOLERANCE,
        "atol": NONLINEAR_TOLERANCE,
        "max_abs": float(np.max(absolute, initial=0)),
        "max_tolerance_fraction": float(np.max(absolute / budget, initial=0)),
        "finite_outputs": int(np.count_nonzero(finite)),
        "nan_outputs": int(np.count_nonzero(np.isnan(got))),
        "positive_infinity_outputs": int(np.count_nonzero(np.isposinf(got))),
        "negative_infinity_outputs": int(np.count_nonzero(np.isneginf(got))),
    }

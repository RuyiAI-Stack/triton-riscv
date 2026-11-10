"""Independent finite-F32 W8A8 classifier reference."""

import numpy as np


def quantize_rows(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("W8A8 quantization requires finite matrix inputs")
    maximum = np.max(np.abs(value), axis=1)
    scale = np.where(
        maximum == 0,
        np.float32(1),
        np.divide(maximum, np.float32(127), dtype=np.float32),
    ).astype(np.float32)
    if np.any((maximum != 0) & (scale == 0)):
        raise ValueError("quantization scale underflow")
    v = np.clip(np.divide(value, scale[:, None], dtype=np.float32), -127, 127)
    base = v.astype(np.int32)
    fraction = np.subtract(v, base.astype(np.float32), dtype=np.float32)
    delta = np.where(np.abs(fraction) >= np.float32(0.5), np.where(v >= 0, 1, -1), 0)
    return (base + delta).astype(np.int8), scale


def stages(arrays):
    qa, rs = quantize_rows(arrays["A"])
    qw, ws = quantize_rows(arrays["W"])
    acc = (qa.astype(np.int64) @ qw.astype(np.int64).T).astype(np.int32)
    y = np.multiply(acc.astype(np.float32), ws[None, :], dtype=np.float32)
    y = np.multiply(y, rs[:, None], dtype=np.float32)
    y = np.add(y, arrays["Bias"][None, :], dtype=np.float32)
    return {"QA": qa, "QW": qw, "RowScale": rs, "WeightScale": ws, "Acc": acc, "Out": y}


def reference(arrays):
    return {"Out": stages(arrays)["Out"]}


def fp32_error(arrays, actual, quantized):
    a = arrays["A"].astype(np.float64)
    w = arrays["W"].astype(np.float64)
    aq = quantized["QA"].astype(np.float64) * quantized["RowScale"][:, None]
    wq = quantized["QW"].astype(np.float64) * quantized["WeightScale"][:, None]
    exact = a @ w.T + arrays["Bias"].astype(np.float64)[None, :]
    bound = np.abs(a - aq) @ np.abs(w).T + np.abs(aq) @ np.abs(w - wq).T
    bound += (
        np.finfo(np.float32).eps * 8 * (np.abs(aq) @ np.abs(wq).T + np.abs(exact) + 1)
    )
    error = np.abs(actual.astype(np.float64) - exact)
    if np.any(error > bound):
        raise AssertionError(
            "W8A8 output exceeds independently derived quantization bound"
        )
    return {
        "max_abs": float(error.max()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "maximum_bound": float(bound.max()),
        "within_quantization_bound": True,
    }

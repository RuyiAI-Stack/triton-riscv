#!/usr/bin/env python3
"""Execute the classifier W8A8 chain using only current emitted kernels.

NumPy supplies independent oracles and prepares inputs; every candidate
quantize, dot, dequantize, bias and residual operation executes emitted IR in
the strict DDR/SPM/REG simulator. This is a Linear-chain check, not a complete
ViT accuracy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from simcheck import SimulationBlocked, check_artifacts, simulate_case

from build import ROOT, describe, digest

CASES = (
    "quantize_1x768",
    "quantize_1000x768",
    "matmul_i8_overwrite_1x1000x768",
    "dequantize_1x1000",
    "bias_add_1x1000",
    "add_1x1000",
)


def quantize_oracle(value):
    """Scalar f32 oracle, with Python integer saturation after truncation."""
    quantized = np.empty(value.shape, dtype=np.int8)
    scales = np.empty(value.shape[0], dtype=np.float32)
    for row, values in enumerate(value):
        maximum = np.float32(max(abs(float(x)) for x in values))
        scale = np.float32(1) if maximum == 0 else maximum / np.float32(127)
        scales[row] = scale
        for column, x in enumerate(values):
            ratio = np.float32(x / scale)
            base = int(ratio)
            fraction = float(ratio) - base
            adjustment = 1 if fraction >= 0.5 else -1 if fraction <= -0.5 else 0
            quantized[row, column] = min(127, max(-127, base + adjustment))
    return quantized, scales


def inputs(multiplier=1):
    ties = np.arange(-126.5, 127, dtype=np.float32)
    pattern = np.concatenate(
        (
            np.stack(
                (
                    np.nextafter(ties, np.float32(-np.inf)),
                    ties,
                    np.nextafter(ties, np.float32(np.inf)),
                ),
                axis=1,
            ).reshape(-1),
            np.array([-127, 0, 127], dtype=np.float32),
        )
    )
    activation = np.resize(pattern, (1, 768)) * np.float32(multiplier)
    weight = np.resize(pattern[::-1], (1000, 768)).copy()
    channel_scale = np.ldexp(
        np.ones(1000, dtype=np.float32), np.arange(1000) % 13 - 6
    )
    weight *= channel_scale[:, None]
    weight[0] = 0
    return activation, weight


def run_chain(kernels, activation, weight, old_c, order, machines):
    stages = []

    def execute(name, arrays):
        actual, evidence = simulate_case(
            describe(name),
            arrays,
            kernels,
            order=order,
            progress=False,
            machines=machines,
        )
        stages.append({"case": name, **evidence})
        return actual

    qa = execute(
        CASES[0],
        {
            "X": activation,
            "Q": np.full((1, 768), -91, np.int8),
            "Scale": np.full(1, -31, np.float32),
        },
    )
    qw = execute(
        CASES[1],
        {
            "X": weight,
            "Q": np.full((1000, 768), 83, np.int8),
            "Scale": np.full(1000, -73, np.float32),
        },
    )
    for source, output in ((activation, qa), (weight, qw)):
        q, scale = quantize_oracle(source)
        np.testing.assert_array_equal(output["Q"], q)
        np.testing.assert_array_equal(
            output["Scale"].view(np.uint32), scale.view(np.uint32)
        )
    product = execute(CASES[2], {"A": qa["Q"], "B": qw["Q"], "C": old_c})
    # Integer accumulation is independent of old C, before any float conversion.
    oracle_acc = (qa["Q"].astype(np.int64) @ qw["Q"].astype(np.int64).T).astype(
        np.int32
    )
    np.testing.assert_array_equal(product["C"], oracle_acc)
    restored = execute(
        CASES[3],
        {
            "X": product["C"],
            "Row": qa["Scale"],
            "Column": qw["Scale"],
            "Out": np.full((1, 1000), np.nan, np.float32),
        },
    )
    first = np.multiply(
        oracle_acc.astype(np.float32), qa["Scale"][:, None], dtype=np.float32
    )
    oracle_out = np.multiply(first, qw["Scale"][None, :], dtype=np.float32)
    np.testing.assert_array_equal(
        restored["Out"].view(np.uint32), oracle_out.view(np.uint32)
    )
    bias = np.linspace(-0.75, 0.5, 1000, dtype=np.float32)
    residual = np.linspace(0.125, -0.25, 1000, dtype=np.float32).reshape(
        1, 1000
    )
    biased = execute(
        CASES[4],
        {
            "X": restored["Out"],
            "Bias": bias,
            "Out": np.full((1, 1000), np.nan, np.float32),
        },
    )
    final = execute(
        CASES[5],
        {
            "X": biased["Out"],
            "Y": residual,
            "Out": np.full((1, 1000), np.nan, np.float32),
        },
    )
    oracle_final = np.add(
        np.add(oracle_out, bias, dtype=np.float32), residual, dtype=np.float32
    )
    np.testing.assert_array_equal(
        final["Out"].view(np.uint32), oracle_final.view(np.uint32)
    )
    fp32_product = np.zeros((1, 1000), dtype=np.float32)
    for k in range(768):
        np.add(
            fp32_product,
            np.multiply(
                activation[:, k, None], weight[None, :, k], dtype=np.float32
            ),
            out=fp32_product,
        )
    fp32_final = np.add(
        np.add(fp32_product, bias, dtype=np.float32), residual, dtype=np.float32
    )
    return (
        final["Out"],
        product["C"],
        {
            "order": order,
            "stages": stages,
            "output_sha256": hashlib.sha256(final["Out"].tobytes()).hexdigest(),
            "max_abs_vs_ordered_fp32": float(
                np.max(np.abs(final["Out"].astype(np.float64) - fp32_final))
            ),
            "exact_integer_and_quantized_float_oracles": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "build/w8a8-chain-check.json"
    )
    args = parser.parse_args()
    report = {
        "schema": 1,
        "status": "BLOCKED",
        "scope": "emitted classifier Linear chain",
        "full_model_executed": False,
        "accuracy_accepted": False,
        "runs": [],
    }
    try:
        modules = {
            name: check_artifacts(describe(name), args.kernels)
            for name in CASES
        }
        module_hashes = {name: digest(path) for name, path in modules.items()}
        matmul = modules[CASES[2]].read_text()
        if (
            "bosc_ame.tile.zero_acc" not in matmul
            or "bosc_ame.tile.mma" not in matmul
        ):
            raise SimulationBlocked(
                "overwrite matmul must contain real tile.zero_acc and i8 MMA"
            )
        machines = {}
        activation, weight = inputs()
        c0 = np.full((1, 1000), 0x317F5A33, dtype=np.int32)
        first, acc, evidence = run_chain(
            args.kernels, activation, weight, c0, "forward", machines
        )
        report["runs"].append({"name": "ties-and-channel-scales", **evidence})
        # Reuse the prior output as C, retaining each Machine's polluted SPM.
        second, _, evidence = run_chain(
            args.kernels, activation, weight, acc, "reverse", machines
        )
        np.testing.assert_array_equal(first, second)
        report["runs"].append({"name": "repeated-workspace-reuse", **evidence})
        # Different old DDR bits and a different token scale must not reuse state.
        activation, weight = inputs(0.125)
        _, _, evidence = run_chain(
            args.kernels, activation, weight, -c0, "tail-first", machines
        )
        report["runs"].append({"name": "changed-token-scale", **evidence})
        _, _, evidence = run_chain(
            args.kernels,
            np.zeros_like(activation),
            weight,
            c0,
            "reverse",
            machines,
        )
        report["runs"].append(
            {"name": "zero-token-and-zero-channel", **evidence}
        )
        for name in CASES:
            current = check_artifacts(describe(name), args.kernels)
            if digest(current) != module_hashes[name]:
                raise SimulationBlocked(
                    f"emitted module changed during chain execution: {name}"
                )
        report.update(
            status="PASS",
            simulator="strict emitted DDR/SPM/REG",
            old_c_reads=0,
            bias_and_residual="after ordered dequantization",
        )
    except SimulationBlocked as error:
        report["reason"] = str(error)
    except Exception as error:
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return (
        0
        if report["status"] == "PASS"
        else 2
        if report["status"] == "BLOCKED"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

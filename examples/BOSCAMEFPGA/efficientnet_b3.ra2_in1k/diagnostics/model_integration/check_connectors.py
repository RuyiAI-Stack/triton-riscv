#!/usr/bin/env python3
"""Execute actual generated BOSCAME IR for independent connector prototypes."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [
    str(REPO / "examples/BOSCAMEFPGA/tools"),
    str(REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools"),
    str(ROOT / "tools"),
]
from contract_frontend import export_kernel  # noqa: E402
from boscame_buffer_sim import Machine  # noqa: E402
from quantized_conv_oracle import quantize_rows  # noqa: E402


def run(args):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)


def compile_stage(module, name, signature, constants, grid, out, opt):
    out.mkdir(parents=True, exist_ok=True)
    symbol = "model_connector_" + name
    accuracy = (
        {
            "exp": {
                "domain": "all-f32",
                "special_values": "ieee",
                "max_abs_error": 1.1210387714598537e-44,
                "max_rel_error": 2e-6,
            }
        }
        if name in ("sigmoid", "silu")
        else None
    )
    export_kernel(
        getattr(module, name),
        signature,
        constants,
        grid,
        symbol,
        out,
        math_accuracy=accuracy,
    )
    source = out / "kernel.frontend.mlir"
    for target, flags in (
        ("kernel.bufferized.mlir", ["--kernel-contract-verify"]),
        ("kernel.planned.mlir", ["--bosc-ame-plan", "--bosc-ame-plan-verify"]),
        ("kernel.boscame.mlir", ["--bosc-ame-emit"]),
        (
            "kernel.loops.mlir",
            ["--convert-linalg-to-loops", "--lower-affine", "--canonicalize"],
        ),
    ):
        dest = out / target
        run([opt, source, *flags, "-o", dest])
        source = dest
    machine = Machine(source, symbol)
    for bank in machine.spm:
        bank[:] = bytes([0xD3]) * len(bank)
    return machine


def execute(machine, arrays, grid):
    raw = [bytearray(np.ascontiguousarray(a).tobytes()) for a in arrays]
    pointers = {"%arg" + str(i): memoryview(a) for i, a in enumerate(raw)}
    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                scalars = {
                    "%arg" + str(len(raw) + i): v
                    for i, v in enumerate([*grid, x, y, z])
                }
                machine.launch(pointers, scalars)
    return [
        np.frombuffer(b, dtype=a.dtype).reshape(a.shape).copy()
        for a, b in zip(arrays, raw)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = ROOT / "build/optimization-baseline/tools/buddy-opt"
    parser.add_argument(
        "--buddy-opt",
        type=Path,
        default=baseline
        if baseline.exists()
        else REPO / "buddy-mlir/build/bin/buddy-opt",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/model-integration/connector-checks"
    )
    args = parser.parse_args()
    opt = args.buddy_opt.resolve()
    os.environ["BUDDY_OPT"] = str(opt)
    spec = importlib.util.spec_from_file_location(
        "efficientnet_model_connectors", HERE / "connector_kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = []

    def stage(name, signature, constants, grid, arrays, check, label=None):
        out = args.output / (label or name)
        machine = compile_stage(module, name, signature, constants, grid, out, opt)
        outputs = execute(machine, arrays, grid)
        metrics = check(outputs) or {}
        result = {
            "stage": name,
            "case": label or name,
            "status": "PASS",
            "constants": constants,
            "grid": grid,
            "programs": int(np.prod(grid)),
            **metrics,
        }
        results.append(result)
        print("PASS", label or name, flush=True)

    for channels, height, width, block in ((3, 3, 5, 4), (5, 1, 1, 8)):
        x = (
            (np.arange(channels * height * width, dtype=np.int32) % 29 - 14).astype(
                np.float32
            )
            / 16
        ).reshape(1, channels, height, width)
        x[:, :, 0, 0] = 0
        q = np.full((height * width, channels), -97, np.int8)
        scale = np.full(height * width, -17, np.float32)
        eq, es = quantize_rows(x[0].reshape(channels, -1).T.copy(), channels)

        def check_quant(o, eq=eq, es=es):
            np.testing.assert_array_equal(o[1], eq)
            np.testing.assert_array_equal(o[2], es)
            return {
                "q_bit_exact": True,
                "scale_bit_exact": True,
                "zero_row_covered": True,
                "channel_tail_covered": True,
            }

        stage(
            "quantize_pack_nchw",
            {"X": "*fp32", "QPacked": "*i8", "ActivationScale": "*fp32"},
            {"C": channels, "H": height, "W": width},
            [height * width, 1, 1],
            [x, q, scale],
            check_quant,
            "quantize_pack_c%d_h%d_w%d" % (channels, height, width),
        )
    oc, h, w = 3, 3, 5
    block = 16
    acc = (np.arange(h * w * oc, dtype=np.int32) % 1025 - 512).reshape(h * w, oc)
    act = (np.arange(h * w, dtype=np.float32) + 1) / 128
    ws = (np.arange(oc, dtype=np.float32) + 1) / 64
    bias = np.array([0.25, -0.5, 0.75], np.float32)
    expected = (
        (acc.astype(np.float32) * ws[None, :]) * act[:, None] + bias[None, :]
    ).T.reshape(1, oc, h, w)

    def check_dequant(o):
        np.testing.assert_array_equal(o[-1], expected)
        return {
            "bit_exact": True,
            "spatial_channel_layout_checked": True,
            "tail_covered": True,
        }

    stage(
        "dequantize_unpack_bias",
        {
            "Acc": "*i32",
            "ActivationScale": "*fp32",
            "WeightScale": "*fp32",
            "Bias": "*fp32",
            "Out": "*fp32",
        },
        {"OC": oc, "H": h, "W": w},
        [h * w, oc, 1],
        [acc, act, ws, bias, np.full_like(expected, -99)],
        check_dequant,
    )
    x = np.linspace(-4, 4, 19, dtype=np.float32)
    block = 16
    for name, expected in (
        ("sigmoid", 1 / (1 + np.exp(-x))),
        ("silu", x / (1 + np.exp(-x))),
    ):

        def check_nonlinear(o, expected=expected):
            err = float(np.max(np.abs(o[-1] - expected)))
            np.testing.assert_allclose(o[-1], expected, atol=2e-6, rtol=2e-6)
            return {"max_abs_error": err, "tail_covered": True}

        stage(
            name,
            {"X": "*fp32", "Out": "*fp32"},
            {"COUNT": 19, "BLOCK": block},
            [2, 1, 1],
            [x, np.full_like(x, -99)],
            check_nonlinear,
        )
    x = (np.arange(45, dtype=np.float32) - 17).reshape(1, 3, 3, 5) / 16
    gate = np.array([0.25, 0.5, 0.75], np.float32)

    def check_mul(o):
        np.testing.assert_array_equal(o[-1], x * gate.reshape(1, 3, 1, 1))
        return {"bit_exact": True, "broadcast_axis_checked": True, "tail_covered": True}

    stage(
        "se_broadcast_multiply",
        {"X": "*fp32", "Gate": "*fp32", "Out": "*fp32"},
        {"C": 3, "SPATIAL": 15, "BLOCK": 16},
        [1, 3, 1],
        [x, gate, np.full_like(x, -99)],
        check_mul,
    )

    def check_zero(o):
        np.testing.assert_array_equal(o[0], np.zeros(19, np.int32))
        return {"bit_exact": True, "nonzero_initial_acc": True}

    stage(
        "zero_accumulator",
        {"Acc": "*i32"},
        {"COUNT": 19, "BLOCK": 16},
        [2, 1, 1],
        [np.arange(19, dtype=np.int32) - 7],
        check_zero,
    )
    bias = np.arange(19, dtype=np.float32) / 8

    def check_bias(o):
        np.testing.assert_array_equal(o[-1], x_bias + bias)
        return {"bit_exact": True, "tail_covered": True}

    x_bias = np.linspace(-3, 3, 19, dtype=np.float32)
    stage(
        "classifier_bias_add",
        {"X": "*fp32", "Bias": "*fp32", "Out": "*fp32"},
        {"COUNT": 19, "BLOCK": 16},
        [2, 1, 1],
        [x_bias, bias, np.full_like(bias, -99)],
        check_bias,
    )
    report = {
        "status": "PASS",
        "scope": "small-shape actual generated BOSCAME IR connector validation; no FPGA evidence",
        "source_sha256": hashlib.sha256(
            (HERE / "connector_kernels.py").read_bytes()
        ).hexdigest(),
        "buddy_opt": str(opt),
        "buddy_opt_sha256": hashlib.sha256(opt.read_bytes()).hexdigest(),
        "checks": results,
    }
    (args.output / "connector-checks.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(results),
                "programs": sum(r["programs"] for r in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

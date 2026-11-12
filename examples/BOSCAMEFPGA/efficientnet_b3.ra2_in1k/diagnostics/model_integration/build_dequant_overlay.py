#!/usr/bin/env python3
"""Prepare independently named RVV connectors without replacing defaults."""

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
from check_dequant_candidate import BUDDY, HERE, LLVM, MODEL, run, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = MODEL / "build/model-integration"
    parser.add_argument(
        "--plan", type=Path, default=base / "physical-plan/physical-plan.json"
    )
    parser.add_argument(
        "--source-connectors", type=Path, default=base / "connectors-ddr-policy"
    )
    parser.add_argument("--output", type=Path, default=base / "dequant-rvv-overlay")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [
        spec
        for spec in json.loads(args.plan.read_text())["connectors"]
        if spec["kernel"] == "dequantize_unpack_bias"
    ]
    source = HERE / "dequant_rvv_candidate.c"
    host = base / "dequant-rvv/host.so"
    fn = ctypes.CDLL(str(host.resolve())).efficientnet_dequant_rvv_candidate
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t] * 2
    rng = np.random.default_rng(124)
    tools = BUDDY / "examples/FPGA-BOSCAME/tools"
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
    ]
    results = []
    for spec in specs:
        channels, spatial = spec["constexprs"]["OC"], spec["grid"][0]
        out = args.output / spec["name"]
        out.mkdir(exist_ok=True)
        symbol = "efficientnet_rvv_" + spec["name"]
        original = args.source_connectors / spec["name"] / "adapter.c"
        original_text = original.read_text()
        marker = "  for (int32_t x"
        if original_text.count(marker) != 1:
            raise ValueError("unexpected connector ABI " + spec["name"])
        wrapper = (
            f"#include <stdint.h>\nextern void {symbol}(const int32_t *,const float *,"
            "const float *,const float *,float *,__SIZE_TYPE__,__SIZE_TYPE__);\n"
            + original_text.split(marker)[0]
            + f"  {symbol}((const int32_t *)p0,(const float *)p1,(const float *)p2,"
            f"(const float *)p3,(float *)p4,{channels}u,{spatial}u);\n}}\n"
        )
        (out / "adapter.c").write_text(wrapper)
        run(
            [
                LLVM / "clang",
                *flags,
                "-Defficientnet_dequant_rvv_candidate=" + symbol,
                "-S",
                source,
                "-o",
                out / "kernel.s",
            ],
            out / "compile.log",
        )
        restricted = run(
            [
                sys.executable,
                tools / "restrict_fpga_assembly.py",
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
            out / "restrict.log",
            (out / "kernel.s").read_text(),
        )
        (out / "kernel.nr.s").write_text(restricted)
        run(
            [LLVM / "clang", *flags, "-c", out / "kernel.nr.s", "-o", out / "kernel.o"],
            out / "object.log",
        )
        # The model builder recompiles this adapter with the final arena size.
        run(
            [
                LLVM / "clang",
                *flags,
                "-I",
                MODEL,
                "-c",
                out / "adapter.c",
                "-o",
                out / "adapter.o",
            ],
            out / "adapter.log",
        )
        count = channels * spatial
        acc = rng.integers(-(2**31), 2**31, (spatial, channels), dtype=np.int32)
        act = rng.uniform(-1, 1, spatial).astype(np.float32)
        ws = rng.uniform(-1, 1, channels).astype(np.float32)
        bias = rng.uniform(-1, 1, channels).astype(np.float32)
        storage = np.full(count + 32, 0x513A7E29, dtype=np.uint32)
        actual = storage[16:-16].view(np.float32).reshape(channels, spatial)
        expected = (
            (acc.astype(np.float32) * ws[None, :]) * act[:, None] + bias[None, :]
        ).T.copy()
        fn(
            acc.ctypes.data,
            act.ctypes.data,
            ws.ctypes.data,
            bias.ctypes.data,
            actual.ctypes.data,
            channels,
            spatial,
        )
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert np.all(storage[:16] == 0x513A7E29) and np.all(
            storage[-16:] == 0x513A7E29
        )
        record = {
            "name": spec["name"],
            "channels": channels,
            "spatial": spatial,
            "native_host_outputs": count,
            "bit_exact_host": True,
            "guard_pass_host": True,
            "original_adapter_sha256": sha(original),
            "adapter_sha256": sha(out / "adapter.c"),
            "object_sha256": sha(out / "kernel.o"),
            "source_sha256": sha(source),
            "isa_restriction_pass": True,
        }
        (out / "candidate.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
    report = {
        "status": "BUILT_LOCAL_PASS_BOARD_ACCEPTANCE_PENDING",
        "source_sha256": sha(source),
        "plan_sha256": sha(args.plan),
        "cases": len(results),
        "host_outputs": sum(r["native_host_outputs"] for r in results),
        "results": results,
        "production_replaced": False,
    }
    (args.output / "overlay-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

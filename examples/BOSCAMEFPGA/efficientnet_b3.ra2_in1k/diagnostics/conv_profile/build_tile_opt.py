#!/usr/bin/env python3
"""Build isolated one-tile preparation candidates with the baseline compiler."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from build_profile import (  # noqa: E402
    digest,
    profile_adapter_source,
    run,
    stage_cases,
    pipeline_layout,
    call_counts,
    tile_selection,
)
from export import describe  # noqa: E402
from quantized_conv_pipeline import _merge_contracts  # noqa: E402

sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
from contract_frontend import export_kernel  # noqa: E402
import tile_prototype  # noqa: E402


def candidate_stages(case, variant):
    stages = stage_cases(case)
    base = stages[1]
    if variant in ("batched", "vector_padding", "cached_scalar", "cached_precleared"):
        function = {
            "batched": "batched_scalar_im2col_quantize",
            "vector_padding": "batched_vector_padding_im2col_quantize",
            "cached_scalar": "cached_scalar_tile_im2col_quantize",
            "cached_precleared": "cached_precleared_tile_im2col_quantize",
        }[variant]
        stages[1] = {
            **base,
            "kernel": function,
            "grid": [
                pipeline_layout(case)["blocks"],
                1,
                pipeline_layout(case)["groups"],
            ],
        }
        return stages
    gather = {
        **base,
        "stage": "pure_im2col",
        "kernel": "tile_gather_im2col",
        "symbol": "triton_" + case["name"] + "__pure_im2col",
        "arguments": [
            {"name": "X", "rank": 4, "dtype": "f32"},
            {"name": "FPatch", "rank": 2, "dtype": "f32"},
        ],
        "signature": {"X": "*fp32", "FPatch": "*fp32"},
        "argument_shapes": [base["argument_shapes"][0], [64, 64]],
        "grid": [pipeline_layout(case)["blocks"], 1, pipeline_layout(case)["groups"]],
    }
    quant = {
        **base,
        "stage": "activation_quantize",
        "kernel": "tile_vector_quantize",
        "symbol": "triton_" + case["name"] + "__activation_quantize",
        "arguments": [
            {"name": "FPatch", "rank": 2, "dtype": "f32"},
            {"name": "QPatch", "rank": 2, "dtype": "i8"},
            {"name": "PatchScale", "rank": 1, "dtype": "f32"},
        ],
        "signature": {"FPatch": "*fp32", "QPatch": "*i8", "PatchScale": "*fp32"},
        "argument_shapes": [[64, 64], [64, 64], [64]],
        "constexprs": {"KPAD": 64, "TILE": 64},
        "grid": gather["grid"],
    }
    if variant == "vector_narrow":
        width = pipeline_layout(case)["width"]
        quant = {
            **quant,
            "kernel": "tile_vector_quantize_narrow",
            "constexprs": {
                "WIDTH": width,
                "ACTIVE": max(16, 1 << (width - 1).bit_length()),
                "KPAD": 64,
                "TILE": 64,
            },
        }
    return [stages[0], gather, quant, *stages[2:]]


def compile_case(case, output, variant):
    output.mkdir(parents=True, exist_ok=True)
    stages = candidate_stages(case, variant)
    opt = Path(os.environ["BUDDY_OPT"])
    buddy = REPO / "buddy-mlir"
    llvm = Path(os.environ.get("LLVM_BINARY_DIR", buddy / "llvm/build/bin"))
    prod = ROOT / "build/kernels" / case["name"]
    contracts = []
    for item in stages:
        d = output / "stages" / item["stage"]
        d.mkdir(parents=True, exist_ok=True)
        if item["stage"] in ("im2col_quantize", "pure_im2col", "activation_quantize"):
            export_kernel(
                getattr(tile_prototype, item["kernel"]),
                item["signature"],
                item["constexprs"],
                item["grid"],
                item["symbol"],
                d,
            )
        else:
            for name in (
                "kernel.ttir",
                "kernel.frontend.mlir",
                "contract-manifest.json",
            ):
                shutil.copy2(prod / "stages" / item["stage"] / name, d / name)
        contracts.append((d / "kernel.frontend.mlir", item["symbol"]))
    (output / "kernel.frontend.mlir").write_text(_merge_contracts(contracts, opt))
    for src, dest, flags in [
        (
            "kernel.frontend.mlir",
            "kernel.bufferized.mlir",
            ["--kernel-contract-verify"],
        ),
        (
            "kernel.bufferized.mlir",
            "kernel.planned.mlir",
            ["--bosc-ame-plan", "--bosc-ame-plan-verify"],
        ),
        ("kernel.planned.mlir", "kernel.boscame.mlir", ["--bosc-ame-emit"]),
        (
            "kernel.boscame.mlir",
            "kernel.llvm.mlir",
            [
                "--lower-bosc-ame=scalar-f32-ordered-reductions=true scalar-f32-max-reductions=true"
            ],
        ),
    ]:
        run([opt, output / src, *flags, "-o", output / dest], output, dest)
    run(
        [
            buddy / "build/bin/buddy-translate",
            "--buddy-to-llvmir",
            output / "kernel.llvm.mlir",
            "-o",
            output / "kernel.ll",
        ],
        output,
        "llvm-ir",
    )
    run(
        [
            llvm / "llc",
            output / "kernel.ll",
            "-O2",
            "-filetype=asm",
            "-mtriple=riscv64",
            "-target-abi=lp64d",
            "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
            "-code-model=medium",
            "-o",
            output / "kernel.s",
        ],
        output,
        "asm",
    )
    import subprocess

    helpers = buddy / "examples/FPGA-BOSCAME/tools"
    for name, src, dest, flags in [
        (
            "ame_to_word.py",
            "kernel.s",
            "kernel.encoded.s",
            ["--gpr-mode=fixed", "--preserve-matrix-registers"],
        ),
        (
            "restrict_fpga_assembly.py",
            "kernel.encoded.s",
            "kernel.nr.s",
            [
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
        ),
    ]:
        p = subprocess.run(
            [sys.executable, str(helpers / name), *flags],
            input=(output / src).read_text(),
            text=True,
            capture_output=True,
        )
        (output / (name + ".log")).write_text(p.stderr)
        if p.returncode:
            raise RuntimeError(p.stderr)
        (output / dest).write_text(p.stdout)
    run(
        [
            llvm / "clang",
            "--target=riscv64-unknown-elf",
            "-march=rv64imafdcv_zicsr_zifencei",
            "-mabi=lp64d",
            "-mno-relax",
            "-c",
            output / "kernel.nr.s",
            "-o",
            output / "kernel.o",
        ],
        output,
        "object",
    )
    return stages


def build(case, root, variant, max_tiles):
    mode = (
        "fused"
        if variant
        in ("batched", "vector_padding", "cached_scalar", "cached_precleared")
        else "split"
    )
    output = root / mode / "all/kernels" / case["name"]
    stages = compile_case(case, output, variant)
    adapter = profile_adapter_source(case, mode, "all", max_tiles)
    block = pipeline_layout(case)["blocks"]
    old = re.compile(
        r"      for [(]int32_t lane = 0; lane < 64; [+][+]lane[)]\n(.*?)(?=      (?:for [(]int32_t m_tile|PROFILE_CALL\(2, _mlir_ciface_.*__gemm))",
        re.S,
    )
    match = old.search(adapter)
    if match is not None:
        replacement = (
            match.group(1)
            .replace("block * 64 + lane", "block")
            .replace(str(block * 64) + ", 1,", str(block) + ", 1,")
        )
        adapter = adapter[: match.start()] + replacement + adapter[match.end() :]
    elif "__im2col_quantize(&u0, &uqp, &ups, " + str(
        block
    ) + ", 1," not in adapter and variant in (
        "batched",
        "vector_padding",
        "cached_scalar",
        "cached_precleared",
    ):
        raise ValueError("neither old lane loop nor current tile call found")
    adapter = adapter.replace("block * 64 + lane", "block").replace(
        str(block * 64) + ", 1,", str(block) + ", 1,"
    )
    if variant == "cached_precleared":
        offset = pipeline_layout(case)["qpatch"]["offset"]
        marker = "  uint64_t profile_adapter_begin = profile_clock();"
        assert adapter.count(marker) == 1
        adapter = adapter.replace(
            marker,
            marker
            + "\n  /* One-time QPatch padding initialization, charged inside adapter timer. */\n  for (int i=0; i<4096; ++i) ((int8_t *)(scratch + "
            + str(offset)
            + 'u))[i]=0;\n  __asm__ volatile("fence rw,rw" ::: "memory");',
        )
    (output / "adapter.c").write_text(adapter)
    layout = pipeline_layout(case)
    count = len(tile_selection(case, max_tiles))
    counts = call_counts(case, mode, count)
    for name in ("im2col_quantize", "pure_im2col", "activation_quantize"):
        if name in counts:
            counts[name] = count
    manifest = {
        "case": case["name"],
        "mode": mode,
        "selected_stage": "all",
        "variant": variant,
        "fixture": {"kind": "deterministic_nonzero_v1"},
        "zero_initialized_launcher": False,
        "coverage": "full original grid"
        if max_tiles is None
        else "bounded uniformly spaced original group/block pairs",
        "selected_tiles": count,
        "full_grid_tiles": layout["groups"] * layout["blocks"],
        "selected_tile_pairs": tile_selection(case, max_tiles),
        "expected_stage_calls": counts,
        "stage_metadata": stages,
        "timing": "same raw-entry return profiling wrapper as baseline; fixture outside timers; candidate one-time QPatch clear inside adapter timer; no added per-entry drain",
        "production_kernel_object_sha256": digest(
            ROOT / "build/kernels" / case["name"] / "kernel.o"
        ),
        "kernel_object_sha256": digest(output / "kernel.o"),
        "profile_adapter_sha256": digest(output / "adapter.c"),
        "profile_generator_sha256": digest(__file__),
        "compiler_path": str(Path(os.environ["BUDDY_OPT"]).resolve()),
        "compiler_sha256": digest(os.environ["BUDDY_OPT"]),
        "prototype_source_sha256": digest(HERE / "tile_prototype.py"),
    }
    run(
        [
            "make",
            "-s",
            "-C",
            case["directory"],
            "-f",
            ROOT / "board.mk",
            "bin",
            "KERNELS=" + str(output.parent),
        ],
        output,
        "board",
    )
    manifest["bin_sha256"] = digest(output / (case["name"] + ".bin"))
    (output / "profile-manifest.json").write_text(
        json.dumps(manifest, indent=2) + chr(10)
    )
    print("PASS", variant, case["name"], output, flush=True)
    return {
        "case": case["name"],
        "mode": mode,
        "stage": "all",
        "output": str(output),
        "status": "PASS",
        "bin_sha256": manifest["bin_sha256"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", default=[])
    p.add_argument(
        "--variant",
        choices=(
            "batched",
            "vector_split",
            "vector_narrow",
            "vector_padding",
            "cached_scalar",
            "cached_precleared",
        ),
        required=True,
    )
    p.add_argument("--max-tiles", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(
            pool.map(
                lambda name: build(
                    describe(name), a.output.resolve(), a.variant, a.max_tiles
                ),
                a.case,
            )
        )
    (a.output / "build-report.json").write_text(json.dumps(results, indent=2) + chr(10))


if __name__ == "__main__":
    main()

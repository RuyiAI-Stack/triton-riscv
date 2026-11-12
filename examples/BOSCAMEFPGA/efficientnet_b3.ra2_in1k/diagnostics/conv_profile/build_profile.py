#!/usr/bin/env python3
"""Build isolated stage-cycle diagnostics without modifying production sources/BINs.

Fused mode uses byte-identical production kernel.o and wraps the existing adapter
raw calls. Split mode replaces fused im2col+quant with two diagnostic Triton
entries and adds a 16 KiB F32 patch tile. --stage selects a standalone raw stage;
this preserves the full adapter loop/count but skips other raw entries. Such
stage-only images are performance probes, not convolution correctness tests.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from export import describe, inventory  # noqa: E402
from quantized_conv_pipeline import _merge_contracts, pipeline_layout, stage_cases  # noqa: E402

SPLIT_SOURCE = HERE / "split_kernels.py"
STAGES = (
    "weight_quantize",
    "im2col_quantize",
    "pure_im2col",
    "activation_quantize",
    "gemm_i8",
    "dequantize_bias",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_stage_cases(case):
    original = {item["stage"]: item for item in stage_cases(case)}
    base = original["im2col_quantize"]
    common = dict(base["constexprs"])
    pure = {
        **base,
        "stage": "pure_im2col",
        "name": case["name"] + "__pure_im2col",
        "symbol": "triton_" + case["name"] + "__pure_im2col",
        "kernel": "pure_im2col",
        "kernel_source": str(SPLIT_SOURCE),
        "kernel_module": "split_kernels",
        "arguments": [
            {"name": "X", "rank": 4, "dtype": "f32"},
            {"name": "FPatch", "rank": 2, "dtype": "f32"},
        ],
        "argument_shapes": [base["argument_shapes"][0], [64, 64]],
        "signature": {"X": "*fp32", "FPatch": "*fp32"},
        "constexprs": common,
    }
    quant = {
        **base,
        "stage": "activation_quantize",
        "name": case["name"] + "__activation_quantize",
        "symbol": "triton_" + case["name"] + "__activation_quantize",
        "kernel": "activation_quantize",
        "kernel_source": str(SPLIT_SOURCE),
        "kernel_module": "split_kernels",
        "arguments": [
            {"name": "FPatch", "rank": 2, "dtype": "f32"},
            {"name": "QPatch", "rank": 2, "dtype": "i8"},
            {"name": "PatchScale", "rank": 1, "dtype": "f32"},
        ],
        "argument_shapes": [[64, 64], [64, 64], [64]],
        "signature": {"FPatch": "*fp32", "QPatch": "*i8", "PatchScale": "*fp32"},
        "constexprs": {"WIDTH": pipeline_layout(case)["width"], "KPAD": 64, "TILE": 64},
    }
    return [
        original["weight_quantize"],
        pure,
        quant,
        original["gemm_i8"],
        original["dequantize_bias"],
    ]


def tile_selection(case, max_tiles=None):
    layout = pipeline_layout(case)
    total = layout["groups"] * layout["blocks"]
    count = total if max_tiles is None else min(max_tiles, total)
    indices = (
        list(range(total))
        if count == total
        else (
            [0]
            if count == 1
            else [i * (total - 1) // (count - 1) for i in range(count)]
        )
    )
    return [
        {"group": index // layout["blocks"], "block": index % layout["blocks"]}
        for index in indices
    ]


def call_counts(case, mode, selected):
    layout = pipeline_layout(case)
    mtiles = (layout["matrix_rows"] + layout["bm"] - 1) // layout["bm"]
    counts = {
        "weight_quantize": layout["channels"],
        "gemm_i8": selected * mtiles,
        "dequantize_bias": selected * layout["matrix_rows"],
    }
    if mode == "fused":
        counts["im2col_quantize"] = selected
    else:
        counts["pure_im2col"] = selected
        counts["activation_quantize"] = selected
    return counts


def fixture_source(case, mode):
    """Initialize public tensors and stage-only dependencies before timers."""
    layout = pipeline_layout(case)
    xlen = math.prod(stage_cases(case)[1]["argument_shapes"][0])
    wlen = layout["channels"] * layout["width"]
    qwo, wso = layout["qweight"]["offset"], layout["weight_scale"]["offset"]
    qpo, pso = layout["qpatch"]["offset"], layout["patch_scale"]["offset"]
    aco = layout["acc"]["offset"]
    text = [
        "  /* Deterministic nonzero diagnostic fixture; initialization outside timers. */",
        f"  for (int i = 0; i < {xlen}; ++i) ((float *)p0)[i] = (float)(i % 29 - 14) * 0.0625f;",
        f"  for (int i = 0; i < {wlen}; ++i) ((float *)p1)[i] = (float)(i % 23 - 11) * 0.03125f;",
        f"  for (int i = 0; i < {layout['channels']}; ++i) ((float *)p2)[i] = (float)(i % 7 - 3) * 0.0625f;",
        f"  for (int i = 0; i < {layout['qweight']['elements']}; ++i) ((int8_t *)(scratch + {qwo}u))[i] = i % 64 < {layout['width']} ? (int8_t)(i % 15 - 7) : 0;",
        f"  for (int i = 0; i < {layout['channels']}; ++i) ((float *)(scratch + {wso}u))[i] = 0.0078125f;",
        f"  for (int i = 0; i < 4096; ++i) ((int8_t *)(scratch + {qpo}u))[i] = i % 64 < {layout['width']} ? (int8_t)(i % 13 - 6) : 0;",
        f"  for (int i = 0; i < 64; ++i) ((float *)(scratch + {pso}u))[i] = 0.015625f;",
        f"  for (int i = 0; i < {layout['acc']['elements']}; ++i) ((int32_t *)(scratch + {aco}u))[i] = i % 1025 - 512;",
    ]
    if mode == "split":
        text.append(
            f"  for (int i = 0; i < 4096; ++i) ((float *)(scratch + {layout['total_bytes']}u))[i] = i % 64 < {layout['width']} ? (float)(i % 29 - 14) * 0.0625f : 0.0f;"
        )
    text.extend(
        [
            '  __asm__ volatile("fence rw, rw" ::: "memory");',
            "  uint64_t profile_adapter_begin = profile_clock();",
        ]
    )
    return "\n".join(text) + "\n"


def profile_adapter_source(case, mode="fused", only_stage="all", max_tiles=None):
    """Retain the exact production raw call text/order and add cycle wrappers."""
    production = ROOT / "build/kernels" / case["name"] / "adapter.c"
    text = production.read_text()
    layout = pipeline_layout(case)
    stage_list = stage_cases(case) if mode == "fused" else split_stage_cases(case)
    enabled = [item["stage"] for item in stage_list]
    if only_stage != "all" and only_stage not in enabled:
        raise ValueError(f"{only_stage} is unavailable in {mode} mode")
    if mode == "split":
        old_symbol = "triton_" + case["name"] + "__im2col_quantize"
        text = re.sub(
            r"^extern void _mlir_ciface_" + old_symbol + r"\([^\n]+\);\n",
            "",
            text,
            flags=re.MULTILINE,
        )
        extra = []
        for item in stage_list:
            if item["stage"] in ("pure_im2col", "activation_quantize"):
                extra.append(
                    f"extern void _mlir_ciface_{item['symbol']}("
                    + ", ".join(
                        ["Unranked *"] * len(item["arguments"]) + ["int32_t"] * 6
                    )
                    + ");"
                )
        entry = "void _mlir_ciface_kernel_" + case["name"]
        text = text.replace(entry, "\n".join(extra) + "\n" + entry, 1)
        old = f"const size_t scratch_bytes = {layout['total_bytes']}u;"
        text = text.replace(
            old, f"const size_t scratch_bytes = {layout['total_bytes'] + 16384}u;", 1
        )
        insertion = (
            f"  Descriptor1D dfp = {{(void *)(scratch + {layout['total_bytes']}u), (void *)(scratch + {layout['total_bytes']}u), 0, 4096, 1}};\n"
            "  Unranked ufp = {1, &dfp};\n"
        )
        text = text.replace(
            "  for (int32_t row = 0;", insertion + "  for (int32_t row = 0;", 1
        )
        needle = re.compile(
            r"^(\s*)_mlir_ciface_" + old_symbol + r"\(&u0, &uqp, &ups, (.*)\);$",
            re.MULTILINE,
        )
        match = needle.search(text)
        if match is None:
            raise ValueError("production fused raw call not found")
        indent, coordinates = match.groups()
        replacement = (
            indent
            + "{\n"
            + indent
            + "  _mlir_ciface_triton_"
            + case["name"]
            + "__pure_im2col(&u0, &ufp, "
            + coordinates
            + ");\n"
            + indent
            + "  _mlir_ciface_triton_"
            + case["name"]
            + "__activation_quantize(&ufp, &uqp, &ups, "
            + coordinates
            + ");\n"
            + indent
            + "}"
        )
        text = text[: match.start()] + replacement + text[match.end() :]
    selected = tile_selection(case, max_tiles)
    if len(selected) < layout["groups"] * layout["blocks"]:
        choices = " || ".join(
            f"(group == {item['group']} && block == {item['block']})"
            for item in selected
        )
        marker = f"    for (int32_t block = 0; block < {layout['blocks']}; ++block) {{"
        if text.count(marker) != 1:
            raise ValueError("production block loop not found")
        text = text.replace(
            marker, marker + "\n      if (!(" + choices + ")) continue;", 1
        )
    text = text.replace(
        "  for (int32_t row = 0;",
        fixture_source(case, mode) + "  for (int32_t row = 0;",
        1,
    )
    stage_ids = {item["stage"]: index for index, item in enumerate(stage_list)}
    only = -1 if only_stage == "all" else stage_ids[only_stage]
    helpers = [
        '#include "nh_e6_platform.h"',
        "/* Diagnostic timers bracket synchronous raw calls; UART occurs only at boundaries. */",
        'static inline uint64_t profile_clock(void) { uint64_t x; __asm__ volatile("rdcycle %0" : "=r"(x) :: "memory"); return x; }',
        f"enum {{ PROFILE_ONLY = {only}, PROFILE_STAGES = {len(stage_list)} }};",
        "#define PROFILE_CALL(ID, CALL) do { \\",
        "  if (PROFILE_ONLY < 0 || PROFILE_ONLY == (ID)) { \\",
        "    if (!profile_calls[ID]) { nh_e6_progress(0xa100u + (ID)); nh_e6_cache_writes(); } \\",
        "    uint64_t profile_begin = profile_clock(); \\",
        "    CALL; \\",
        "    uint64_t profile_end = profile_clock(); \\",
        "    profile_cycles[ID] += profile_end - profile_begin; \\",
        "    ++profile_calls[ID]; \\",
        "  } \\",
        "} while (0)",
    ]
    text = text.replace(
        '#include "support.h"', '#include "support.h"\n' + "\n".join(helpers), 1
    )
    start = re.search(r"^(void _mlir_ciface_kernel_[^\n]+\{)\n", text, re.MULTILINE)
    if start is None:
        raise ValueError("production public adapter not found")
    declarations = (
        "  uint64_t profile_cycles[PROFILE_STAGES] = {0};\n"
        "  uint64_t profile_calls[PROFILE_STAGES] = {0};\n"
        "  uint64_t profile_timer_overhead = 0;\n"
        "  for (unsigned probe = 0; probe < 128; ++probe) {\n"
        "    uint64_t begin = profile_clock(); uint64_t end = profile_clock();\n"
        "    profile_timer_overhead += end - begin;\n"
        "  }\n"
    )
    text = text[: start.end()] + declarations + text[start.end() :]
    # Match call statements only, never raw declarations. Macro has do/while scope,
    # so the original unbraced for loops retain exactly their one-call body.
    for name, index in stage_ids.items():
        symbol = "_mlir_ciface_triton_" + case["name"] + "__" + name
        expression = re.compile(
            r"^(\s*)(" + re.escape(symbol) + r"\([^\n]+\));$", re.MULTILINE
        )
        text, count = expression.subn(
            lambda match: (
                match.group(1) + f"PROFILE_CALL({index}, " + match.group(2) + ");"
            ),
            text,
        )
        if count != 1:
            raise ValueError(f"expected one raw call site for {name}, found {count}")
    end = text.rfind("\n}")
    reports = [
        "  uint64_t profile_adapter_cycles = profile_clock() - profile_adapter_begin;",
        "  nh_e6_progress(0xa200u + (PROFILE_ONLY < 0 ? PROFILE_STAGES : PROFILE_ONLY));",
        "  nh_e6_cache_writes();",
        f'  nr_puts("PROFILE mode={mode} selected={only_stage} case={case["name"]}\\n");',
    ]
    for name, index in stage_ids.items():
        reports.extend(
            [
                f'  nr_puts("PROFILE stage={name} calls=0x"); nr_hex64(profile_calls[{index}]);',
                f'  nr_puts(" cycles=0x"); nr_hex64(profile_cycles[{index}]); nr_puts("\\n");',
            ]
        )
    reports.extend(
        [
            '  nr_puts("PROFILE adapter_total_cycles=0x"); nr_hex64(profile_adapter_cycles); nr_puts("\\n");',
            '  nr_puts("PROFILE timer_pairs=0x0000000000000080 total_empty_pair_cycles=0x"); nr_hex64(profile_timer_overhead); nr_puts("\\n");',
        ]
    )
    return text[:end] + "\n" + "\n".join(reports) + text[end:]


def run(command, output, label, stdin=None):
    log = output / (label + ".log")
    with log.open("w") as stream:
        stream.write("$ " + shlex.join(map(str, command)) + "\n")
        stream.flush()
        result = subprocess.run(
            list(map(str, command)),
            input=stdin,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{label} failed: {log}\n" + log.read_text()[-3000:])


def compile_split(case, output):
    """Export only the two new stages; reuse the other three exact Contracts."""
    buddy = Path(os.environ.get("BUDDY_ROOT", REPO / "buddy-mlir"))
    opt = Path(os.environ.get("BUDDY_OPT", buddy / "build/bin/buddy-opt"))
    llvm = Path(os.environ.get("LLVM_BINARY_DIR", buddy / "llvm/build/bin"))
    sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
    from contract_frontend import export_kernel

    module_spec = importlib.util.spec_from_file_location(
        "efficientnet_diag_split", SPLIT_SOURCE
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    contracts = []
    stages = split_stage_cases(case)
    production = ROOT / "build/kernels" / case["name"]
    for stage in stages:
        directory = output / "stages" / stage["stage"]
        directory.mkdir(parents=True, exist_ok=True)
        if stage["stage"] in ("pure_im2col", "activation_quantize"):
            export_kernel(
                getattr(module, stage["kernel"]),
                stage["signature"],
                stage["constexprs"],
                stage["grid"],
                stage["symbol"],
                directory,
            )
        else:
            for name in (
                "kernel.ttir",
                "kernel.frontend.mlir",
                "contract-manifest.json",
            ):
                shutil.copy2(
                    production / "stages" / stage["stage"] / name, directory / name
                )
        contracts.append((directory / "kernel.frontend.mlir", stage["symbol"]))
    (output / "kernel.frontend.mlir").write_text(_merge_contracts(contracts, opt))
    run(
        [
            opt,
            output / "kernel.frontend.mlir",
            "--kernel-contract-verify",
            "-o",
            output / "kernel.bufferized.mlir",
        ],
        output,
        "bufferized",
    )
    run(
        [
            opt,
            output / "kernel.bufferized.mlir",
            "--bosc-ame-plan",
            "--bosc-ame-plan-verify",
            "-o",
            output / "kernel.planned.mlir",
        ],
        output,
        "planned",
    )
    run(
        [
            opt,
            output / "kernel.planned.mlir",
            "--bosc-ame-emit",
            "-o",
            output / "kernel.boscame.mlir",
        ],
        output,
        "boscame",
    )
    run(
        [
            opt,
            output / "kernel.boscame.mlir",
            "--lower-bosc-ame=scalar-f32-ordered-reductions=true scalar-f32-max-reductions=true",
            "-o",
            output / "kernel.llvm.mlir",
        ],
        output,
        "llvm",
    )
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
    helpers = buddy / "examples/FPGA-BOSCAME/tools"
    for name, source, dest, options in (
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
    ):
        result = subprocess.run(
            [sys.executable, str(helpers / name), *options],
            input=(output / source).read_text(),
            text=True,
            capture_output=True,
            check=False,
        )
        (output / (name + ".log")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(name + ": " + result.stderr[-2000:])
        (output / dest).write_text(result.stdout)
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
    (output / "diagnostic-stages.json").write_text(json.dumps(stages, indent=2) + "\n")


def build_case(case, output_root, mode, only_stage, build, max_tiles=None):
    output = output_root / mode / only_stage / "kernels" / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    production = ROOT / "build/kernels" / case["name"]
    if mode == "fused":
        for path in production.glob("kernel.*"):
            if path.is_file():
                shutil.copy2(path, output / path.name)
    else:
        cache = output_root / "split" / "_raw" / case["name"]
        cache.mkdir(parents=True, exist_ok=True)
        stamp = {
            "split_source_sha256": digest(SPLIT_SOURCE),
            "production_frontend_sha256": digest(production / "frontend.json"),
        }
        stamp_path = cache / ".split-inputs.json"
        if (
            not (cache / "kernel.o").is_file()
            or not stamp_path.is_file()
            or json.loads(stamp_path.read_text()) != stamp
        ):
            compile_split(case, cache)
            stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
        for path in cache.iterdir():
            if path.is_file() and path.name != ".split-inputs.json":
                shutil.copy2(path, output / path.name)
        shutil.copytree(cache / "stages", output / "stages", dirs_exist_ok=True)
    adapter = profile_adapter_source(case, mode, only_stage, max_tiles)
    (output / "adapter.c").write_text(adapter)
    manifest = {
        "case": case["name"],
        "mode": mode,
        "selected_stage": only_stage,
        "coverage": "full original grid"
        if max_tiles is None
        else "bounded uniformly selected group/block pairs with original raw grid/pids",
        "selected_tile_pairs": tile_selection(case, max_tiles),
        "selected_tiles": len(tile_selection(case, max_tiles)),
        "full_grid_tiles": pipeline_layout(case)["groups"]
        * pipeline_layout(case)["blocks"],
        "expected_stage_calls": call_counts(
            case, mode, len(tile_selection(case, max_tiles))
        ),
        "original_full_grid_stage_calls": call_counts(
            case,
            mode,
            pipeline_layout(case)["groups"] * pipeline_layout(case)["blocks"],
        ),
        "zero_initialized_launcher": False,
        "fixture": {
            "kind": "deterministic_nonzero_v1",
            "public_X": "(i%29-14)/16",
            "public_Weight": "(i%23-11)/32",
            "public_Bias": "(i%7-3)/16",
            "standalone_QWeight": "(i%15-7), padding zero",
            "standalone_QPatch": "(i%13-6), padding zero",
            "standalone_WeightScale": "1/128",
            "standalone_PatchScale": "1/64",
            "standalone_Acc": "i%1025-512",
            "standalone_FPatch": "(i%29-14)/16, padding zero",
            "timing": "all fixture initialization and fence outside adapter_total/raw-stage timers",
        },
        "split_caveat": "stage-isolation diagnostics only: each raw entry processes all 64 patch rows per spatial tile; split pure im2col stores and activation_quantize reads an extra 16 KiB F32 patch tile, so stage readings isolate diagnostic entries and the split total is not production end-to-end performance or a strict decomposition of fused im2col_quantize",
        "split_variant": "scalar_logical_width_tile_v2" if mode == "split" else None,
        "numeric_claim": "all-stage preserves the operation on deterministic fixtures; stage-only measures independent deterministic inputs and is not a convolution numeric test",
        "production_kernel_object_sha256": digest(production / "kernel.o"),
        "kernel_object_sha256": digest(output / "kernel.o"),
        "production_adapter_sha256": digest(production / "adapter.c"),
        "profile_adapter_sha256": digest(output / "adapter.c"),
        "profile_generator_sha256": digest(__file__),
        "stage_metadata": stage_cases(case)
        if mode == "fused"
        else split_stage_cases(case),
        "progress_markers": "first enabled call 0xa100+stage-index; end 0xa200+selected-index (or stage-count for all)",
        "timing": "RV64 rdcycle with compiler memory clobber around raw entry return; no per-call hardware drain, scalar stores may still be buffered; MMA has existing synchronization; includes read-pair cost, counter/loop overhead only in adapter total; empty-pair calibration is not total instrumentation overhead",
    }
    if only_stage != "all":
        manifest["expected_stage_calls"] = {
            name: count if name == only_stage else 0
            for name, count in manifest["expected_stage_calls"].items()
        }
    if (
        mode == "fused"
        and manifest["production_kernel_object_sha256"]
        != manifest["kernel_object_sha256"]
    ):
        raise AssertionError("fused profile changed the production raw object")
    (output / "profile-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if build:
        kernels = output.parent
        run(
            [
                "make",
                "-s",
                "-C",
                case["directory"],
                "-f",
                ROOT / "board.mk",
                "bin",
                "KERNELS=" + str(kernels),
            ],
            output,
            "board",
        )
        manifest["bin_sha256"] = digest(output / (case["name"] + ".bin"))
        manifest["elf_audit_sha256"] = digest(output / "elf-audit.json")
        (output / "profile-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(f"PASS {mode}/{only_stage} {case['name']} {output}", flush=True)
    return {
        "case": case["name"],
        "mode": mode,
        "stage": only_stage,
        "output": str(output),
        "status": "PASS",
        **({"bin_sha256": manifest["bin_sha256"]} if build else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("fused", "split"), default="fused")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument(
        "--all-stage-modes",
        action="store_true",
        help="build all-stage plus each standalone stage",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/w8a8-profile")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="limit uniformly selected group/block pairs; full weight rows retained",
    )
    args = parser.parse_args()
    if args.max_tiles is not None and args.max_tiles < 1:
        parser.error("--max-tiles must be positive")
    cases = (
        [
            item
            for item in inventory()
            if item["family"] in ("conv2d_f32", "dwconv2d_f32")
        ]
        if args.all
        else [describe(name) for name in args.case]
    )
    if not cases:
        parser.error("provide --all or --case")
    output = args.output.resolve()

    # Compile each case serially across its modes so split cache writes never race.
    def one(case):
        names = (
            [
                "all",
                *[
                    item["stage"]
                    for item in (
                        stage_cases(case)
                        if args.mode == "fused"
                        else split_stage_cases(case)
                    )
                ],
            ]
            if args.all_stage_modes
            else [args.stage]
        )
        return [
            build_case(case, output, args.mode, name, args.build, args.max_tiles)
            for name in names
        ]

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(one, case) for case in cases]):
            records.extend(future.result())
    report = output / ("build-report-" + args.mode + "-" + args.stage + ".json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

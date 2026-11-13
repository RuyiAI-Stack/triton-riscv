#!/usr/bin/env python3
"""Full-grid nonzero FPGA probes with downloaded DDR W8A8 validation."""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from build_profile import (  # noqa: E402
    call_counts,
    digest,
    pipeline_layout,
    profile_adapter_source,
    run,
    stage_cases,
    tile_selection,
)
from export import argument_shapes, describe  # noqa: E402
from quantized_conv_oracle import quantization_error_metrics, reference  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(name, output, kernel_root=None, preclear=False):
    case = describe(name)
    destination = output / "fused/all/kernels" / name
    destination.mkdir(parents=True, exist_ok=True)
    integrated = ROOT / "build/w8a8-opt/integrated" / name
    frozen = (kernel_root / name) if kernel_root else integrated
    stages = stage_cases(case)
    shapes = argument_shapes(case)
    counts = [math.prod(shape) for shape in shapes]
    tail = ((64 + counts[3] * 4 + 63) // 64) * 64
    offsets = [tail + 64]
    for count in counts[:2]:
        offsets.append(((offsets[-1] + count * 4 + 63) // 64) * 64)
    offsets.append(64)
    assert (
        offsets[2] + counts[2] * 4
        < 16 * 1024 * 1024 - pipeline_layout(case)["total_bytes"]
    )
    launch = [
        '#include "support.h"',
        f"extern void _mlir_ciface_kernel_{name}(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);",
        "int launch(void) {",
    ]
    for index, (offset, shape) in enumerate(zip(offsets, shapes)):
        launch.append(f"  float *p{index} = workspace({offset});")
        launch.append(
            f"  MemRef{len(shape)} a{index} = make_{len(shape)}(p{index}, "
            + ", ".join(map(str, shape))
            + ");"
        )
    launch += [
        f"  for (int i = 0; i < {counts[3]}; ++i) ((uint32_t *)p3)[i] = 0x7fc12345u;",
        f"  for (int i = 0; i < 16; ++i) ((uint32_t *)workspace(0))[i] = ((uint32_t *)workspace({tail}))[i] = 0x513a7e29u;",
        f"  _mlir_ciface_kernel_{name}(&a0, &a1, &a2, &a3);",
        '  __asm__ volatile("fence iorw,iorw" ::: "memory");',
        "  return 0;",
        "}",
    ]
    (destination / "launch.c").write_text("\n".join(launch) + "\n")
    adapter = profile_adapter_source(case, "fused", "all", None)
    if preclear:
        marker = "  uint64_t profile_adapter_begin = profile_clock();"
        offset = pipeline_layout(case)["qpatch"]["offset"]
        assert adapter.count(marker) == 1
        adapter = adapter.replace(
            marker,
            marker
            + "\n  for (int i=0; i<4096; ++i) ((int8_t *)(scratch + "
            + str(offset)
            + 'u))[i]=0;\n  __asm__ volatile("fence rw,rw" ::: "memory");',
        )
    (destination / "adapter.c").write_text(adapter)
    support = (ROOT / "board_support.c").read_text()
    declarations = []
    for artifact, offset, dtype, shape in [
        ("Out", 64, "<f4", shapes[3]),
        ("GuardBefore", 0, "<u4", [16]),
        ("GuardAfter", tail, "<u4", [16]),
    ]:
        record = {
            "version": 1,
            "name": artifact,
            "address": hex(0x80100000 + offset),
            "nbytes": math.prod(shape) * 4,
            "dtype": dtype,
            "shape": shape,
        }
        text = "\r\n@@FPGA_OUTPUT " + json.dumps(record, separators=(",", ":")) + "\r\n"
        declarations.append("    if (ok) print_uart(" + json.dumps(text) + ");")
    marker = (
        "    int ok = done == NH_E6_DONE_PASS && launch_returned && launch_status == 0;"
    )
    assert support.count(marker) == 1
    support = support.replace(marker, marker + "\n" + "\n".join(declarations))
    (destination / "board_support.c").write_text(support)
    buddy = REPO / "buddy-mlir"
    llvm = Path(os.environ.get("LLVM_BINARY_DIR", buddy / "llvm/build/bin"))
    runtime = buddy / "runtime/boscame"
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdc_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-nostdlib",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-pie",
        "-fno-stack-protector",
        "-ffunction-sections",
        "-fdata-sections",
        "-ffp-contract=off",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-DVIT_WORKSPACE_BYTES=16777216",
        "-DUART_FREQ_HZ=5529600",
        "-DBOSC_AME_DMA_ASSERT(x)=",
    ]
    flags += [
        "-I" + str(path)
        for path in [
            ROOT,
            runtime / "runtime",
            runtime / "uart",
            runtime / "nh_e6",
            runtime / "nr",
        ]
    ]
    for unit in ("launch", "adapter", "board_support"):
        run(
            [
                llvm / "clang",
                *flags,
                "-c",
                destination / (unit + ".c"),
                "-o",
                destination / (unit + ".o"),
            ],
            destination,
            unit,
        )
    objects = [
        destination / (unit + ".o") for unit in ("launch", "adapter", "board_support")
    ]
    objects += [
        integrated / "obj" / (unit + ".o")
        for unit in (
            "nh_e6_boot",
            "nh_e6_platform",
            "uart",
            "bosc_ame_dma",
            "nr_math",
            "ame_sync",
        )
    ]
    objects.append(frozen / "kernel.o")
    image = destination / (case["name"] + ".elf")
    run(
        [
            shutil.which("ld.lld"),
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            ROOT / "link.ld",
            "-Map=" + str(destination / "image.map"),
            "-o",
            image,
            *objects,
        ],
        destination,
        "link",
    )
    run(
        [
            sys.executable,
            buddy / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            image,
            "--objdump",
            llvm / "llvm-objdump",
            "--output",
            destination / "elf-audit.json",
        ],
        destination,
        "audit",
    )
    binary = image.with_suffix(".bin")
    run([llvm / "llvm-objcopy", "-O", "binary", image, binary], destination, "bin")
    layout = pipeline_layout(case)
    tiles = layout["groups"] * layout["blocks"]
    expected = call_counts(case, "fused", tiles)
    expected["im2col_quantize"] = tiles
    manifest = {
        "case": case["name"],
        "mode": "fused",
        "selected_stage": "all",
        "fixture": {"kind": "deterministic_nonzero_v1"},
        "zero_initialized_launcher": False,
        "coverage": "full original group/spatial/output grid, all DDR output elements",
        "selected_tiles": tiles,
        "full_grid_tiles": tiles,
        "selected_tile_pairs": tile_selection(case),
        "expected_stage_calls": expected,
        "stage_metadata": stages,
        "timing": "raw entry-return profile; fixture outside adapter timer",
        "production_kernel_object_sha256": digest(integrated / "kernel.o"),
        "kernel_object_sha256": digest(frozen / "kernel.o"),
        "profile_adapter_sha256": digest(destination / "adapter.c"),
        "profile_generator_sha256": digest(__file__),
        "bin_sha256": digest(binary),
        "output_probe": {
            "output_elements": counts[3],
            "output_offset": 64,
            "guard_after_offset": tail,
            "output_shape": shapes[3],
            "candidate_preclear": preclear,
            "kernel_directory": str(frozen),
        },
    }
    save(destination / "profile-manifest.json", manifest)
    print("PASS build output probe", case["name"], digest(binary), flush=True)


def verify(run_root, output, expect_all18=False):
    results = []
    # Reused exact-BIN runs may be symlinked at the case-directory level.
    # Path.rglob does not descend into those directories on Python 3.11.
    candidates = list(run_root.rglob("summary.json"))
    candidates += list(run_root.glob("fused/all/*/summary.json"))
    paths = {path.resolve(): path for path in candidates}
    for path in sorted(paths.values()):
        if path.parent.name.startswith("attempt-"):
            continue
        report = json.loads(path.read_text())
        if (
            report.get("execution_status") != "exited"
            or report.get("app_exit_code") != 0
        ):
            raise RuntimeError(f"FPGA probe did not return successfully: {path}")
        case = describe(report["case"])
        manifest = json.loads(Path(report["manifest_path"]).read_text())
        if "output_probe" not in manifest:
            continue
        arrays = {}
        for index, name in enumerate(["X", "Weight", "Bias"]):
            shape = argument_shapes(case)[index]
            length = math.prod(shape)
            modulus, center, scale = [
                (29, 14, 0.0625),
                (23, 11, 0.03125),
                (7, 3, 0.0625),
            ][index]
            arrays[name] = (
                (np.arange(length) % modulus - center).astype(np.float32)
                * np.float32(scale)
            ).reshape(shape)
        files = {item["name"]: Path(item["path"]) for item in report["downloads"]}
        actual = np.fromfile(files["Out"], dtype="<f4").reshape(
            argument_shapes(case)[3]
        )
        expected = reference(case, arrays)["Out"]
        quantization = quantization_error_metrics(case, arrays, actual)
        bit_errors = actual.view(np.uint32) != expected.view(np.uint32)
        max_error = float(np.max(np.abs(actual - expected)))
        guards = all(
            np.all(np.fromfile(files[name], dtype="<u4") == 0x513A7E29)
            for name in ["GuardBefore", "GuardAfter"]
        )
        passed = (
            not np.any(bit_errors)
            and guards
            and report.get("profile_counter_check_passed")
        )
        entry = {
            "case": case["name"],
            "status": "PASS" if passed else "FAIL",
            "task_id": report["task_id"],
            "fpga": report["fpga"],
            "elements": actual.size,
            "bit_mismatches": int(np.sum(bit_errors)),
            "max_abs_error": max_error,
            "guard_check": guards,
            "original_f32_quantization": quantization,
            "full_grid": True,
            "fixture": manifest["fixture"],
            "output_sha256": digest(files["Out"]),
            "oracle_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            "launch_cycles": report["profile"]["launch_cycles"],
            "adapter_cycles": report["profile"]["adapter_total_cycles"],
            "profile_counter_check_passed": report["profile_counter_check_passed"],
            "production_kernel_object_sha256": manifest["kernel_object_sha256"],
        }
        if np.any(bit_errors):
            entry["mismatch_samples"] = [
                {
                    "index": row.tolist(),
                    "actual": float(actual[tuple(row)]),
                    "expected": float(expected[tuple(row)]),
                }
                for row in np.argwhere(bit_errors)[:16]
            ]
        results.append(entry)
        print(
            entry["status"],
            case["name"],
            "elements=",
            actual.size,
            "bit_mismatches=",
            entry["bit_mismatches"],
            "max_abs_error=",
            max_error,
            flush=True,
        )
    save(output, results)
    if not results or any(item["status"] != "PASS" for item in results):
        raise RuntimeError("DDR numerical verification incomplete or failed")
    if expect_all18:
        from cases import inventory

        expected = {
            case["name"]
            for case in inventory()
            if case["family"] in ("conv2d_f32", "dwconv2d_f32")
        }
        actual = [item["case"] for item in results]
        if len(expected) != 18 or len(actual) != 18 or set(actual) != expected:
            raise RuntimeError(
                "expected exactly the 4 ordinary and 14 depthwise convolution cases"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-runs", type=Path)
    parser.add_argument(
        "--expect-all18",
        action="store_true",
        help="require one full-output result for every replaced convolution",
    )
    parser.add_argument(
        "--kernel-root",
        type=Path,
        help="isolated candidate kernel.o root; runtime remains frozen integrated",
    )
    parser.add_argument(
        "--preclear-qpatch",
        action="store_true",
        help="candidate raw writes logical K only; initialize QPatch once inside timer",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_runs:
        verify(
            args.verify_runs, args.output / "numerical-report.json", args.expect_all18
        )
    else:
        for name in args.case:
            build(name, args.output.resolve(), args.kernel_root, args.preclear_qpatch)


if __name__ == "__main__":
    main()

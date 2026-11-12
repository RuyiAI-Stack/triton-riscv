#!/usr/bin/env python3
"""Check native launchers and generated descriptor/grid adapters with mock entries.

The raw mock checks descriptors and every grid coordinate, then the wrapper
leaves launch() before any numerical oracle. Malformed descriptors must trap.
This checks ABI behavior only; it executes neither kernels nor the simulator.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import (
    ROOT,
    argument_shapes,
    describe,
    inventory,
    is_quantized_conv_case,
)


def composite_mock_source(case):
    """Mock all four raw convolution stages and verify their adapter order."""
    from quantized_conv_pipeline import case_is_depthwise, pipeline_layout, stage_cases

    stages = {stage["stage"]: stage for stage in stage_cases(case)}
    spec = case["constexprs"]
    width = pipeline_layout(case)["width"]
    depthwise = case_is_depthwise(case)
    rows = spec["C"] if depthwise else spec["OC"]
    positions = spec["OH"] * spec["OW"]
    tile = 64
    blocks = (positions + tile - 1) // tile
    groups = rows if depthwise else 1
    bm = 1 if depthwise else 16
    m_tiles = 1 if depthwise else (rows + bm - 1) // bm
    total_tiles = groups * blocks
    stage_order = ("weight_quantize", "im2col_quantize", "gemm_i8", "dequantize_bias")
    stage_ids = {name: index for index, name in enumerate(stage_order)}
    item_bytes = {"f32": 4, "i8": 1, "i32": 4}

    stage_arg_shapes = {}
    stage_arg_bytes = {}
    for name, stage in stages.items():
        stage_arg_shapes[name] = [
            math.prod(shape) for shape in stage["argument_shapes"]
        ]
        stage_arg_bytes[name] = [
            elements * item_bytes[argument["dtype"]]
            for elements, argument in zip(stage_arg_shapes[name], stage["arguments"])
        ]
    if depthwise:
        # The adapter offsets QWeight to one group's row before each depthwise GEMM.
        stage_arg_shapes["gemm_i8"][0] = 64
        stage_arg_bytes["gemm_i8"][0] = 64

    lines = [
        '#include "support.h"',
        "#include <setjmp.h>",
        "#include <signal.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <sys/resource.h>",
        "#include <sys/wait.h>",
        "#include <unistd.h>",
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
        "static jmp_buf completed;",
        "static size_t adapters;",
        "static size_t write_begin_calls;",
        "void bosc_ame_ddr_write_begin(void) { ++write_begin_calls; }",
        "static int phase;",
        "static size_t weight_calls, im2col_calls, gemm_calls, dequant_calls;",
        "static size_t gemm_total, dequant_total, tile_index;",
        "static uintptr_t first_pointers[4][5];",
        "static int pointer_seen[4][5];",
        "static uintptr_t current_pointers[4][5];",
        "static size_t current_bytes[4][5];",
        'static void require(int ok, const char *what) { if (!ok) { fprintf(stderr, "ABI smoke failure: %s\\n", what); abort(); } }',
        "static uintptr_t check_argument(int stage, int index, Unranked *u, size_t elements, size_t bytes, size_t variant_offset) {",
        '  require(u && u->rank == 1 && u->descriptor, "unranked rank-one descriptor");',
        "  Descriptor1D *d = (Descriptor1D *)u->descriptor;",
        '  require(d->allocated == d->aligned && d->offset == 0 && d->size == (int64_t)elements && d->stride == 1, "contiguous physical extent");',
        "  uintptr_t p = (uintptr_t)d->aligned;",
        '  require(p && !(p & 63u) && p <= UINTPTR_MAX - bytes, "address and alignment");',
        "  uintptr_t arena = (uintptr_t)workspace(0);",
        '  require(p >= arena && p + bytes <= arena + VIT_WORKSPACE_BYTES, "arena extent");',
        "  if (!pointer_seen[stage][index]) { first_pointers[stage][index] = p; pointer_seen[stage][index] = 1; }",
        '  require(p == first_pointers[stage][index] + variant_offset, "stable stage buffer address");',
        "  current_pointers[stage][index] = p; current_bytes[stage][index] = bytes;",
        "  for (int j = 0; j < index; ++j)",
        '    require(p + bytes <= current_pointers[stage][j] || current_pointers[stage][j] + current_bytes[stage][j] <= p, "stage arguments do not alias");',
        "  return p;",
        "}",
    ]

    for stage_name in stage_order:
        stage = stages[stage_name]
        stage_id = stage_ids[stage_name]
        arguments = stage["arguments"]
        shapes = stage_arg_shapes[stage_name]
        byte_lengths = stage_arg_bytes[stage_name]
        parameters = [f"Unranked *u{i}" for i in range(len(arguments))]
        parameters += [
            "int32_t gx",
            "int32_t gy",
            "int32_t gz",
            "int32_t px",
            "int32_t py",
            "int32_t pz",
        ]
        grid = stage["grid"]
        lines += [
            f"void _mlir_ciface_{stage['symbol']}({', '.join(parameters)}) {{",
            f'  require(gx == {grid[0]} && gy == {grid[1]} && gz == {grid[2]}, "{stage_name}: launch grid");',
        ]
        for index, (elements, length) in enumerate(zip(shapes, byte_lengths)):
            offset = "0"
            if depthwise and stage_name == "gemm_i8" and index == 0:
                offset = f"(uintptr_t)(tile_index / {blocks}u) * 64u"
            lines.append(
                f"  (void)check_argument({stage_id}, {index}, u{index}, {elements}u, {length}u, {offset});"
            )

        if stage_name == "weight_quantize":
            lines += [
                '  require(phase == 0, "weight quantization call order");',
                '  require(px == (int32_t)weight_calls && py == 0 && pz == 0, "weight quantization coordinates");',
                "  ++weight_calls;",
                f"  if (weight_calls == {rows}u) phase = 1;",
            ]
        elif stage_name == "im2col_quantize":
            lines += [
                '  require(phase == 1, "im2col call order");',
                f"  size_t group = tile_index / {blocks}u;",
                f"  size_t block = tile_index % {blocks}u;",
                '  require(px == (int32_t)block && py == 0 && pz == (int32_t)group, "im2col tile coordinates and order");',
                '  require(write_begin_calls == 1, "one write policy boundary per public adapter");',
                "  Descriptor1D *patch = (Descriptor1D *)u1->descriptor;",
                "  unsigned char *patch_bytes = patch->aligned;",
                "  for (int i = 0; i < 4096; ++i) {",
                f"    unsigned char expected = im2col_calls && i % 64 < {width} ? 0x5a : 0;",
                '    require(patch_bytes[i] == expected, "QPatch initialized once and padding reused");',
                f"    if (i % 64 < {width}) patch_bytes[i] = 0x5a;",
                "  }",
                "  ++im2col_calls;",
                "  gemm_calls = 0; phase = 2;",
            ]
        elif stage_name == "gemm_i8":
            lines += [
                '  require(phase == 2, "GEMM call order");',
                '  require(px == (int32_t)gemm_calls && py == 0 && pz == 0, "GEMM coordinates and tile order");',
                "  ++gemm_calls; ++gemm_total;",
                "  dequant_calls = 0; phase = 3;",
            ]
        else:
            if depthwise:
                lines += [
                    '  require(phase == 3, "dequantization call order");',
                    f"  size_t group = tile_index / {blocks}u;",
                    f"  size_t block = tile_index % {blocks}u;",
                    '  require(dequant_calls == 0 && px == (int32_t)group && py == (int32_t)block && pz == 0, "dequantization coordinates");',
                ]
            else:
                lines += [
                    '  require(phase == 3, "dequantization call order");',
                    f"  size_t block = tile_index % {blocks}u;",
                    "  size_t m_tile = gemm_calls - 1u;",
                    f"  size_t expected_oc = m_tile * {bm}u + dequant_calls;",
                    '  require(px == (int32_t)expected_oc && py == (int32_t)block && pz == (int32_t)m_tile, "dequantization coordinates and channel order");',
                ]
            if depthwise:
                lines += [
                    "  ++dequant_calls; ++dequant_total;",
                    '  require(dequant_calls == 1u, "one depthwise output channel per GEMM");',
                    "  ++tile_index;",
                    f"  if (tile_index == {total_tiles}u) phase = 4; else phase = 1;",
                ]
            else:
                lines += [
                    "  ++dequant_calls; ++dequant_total;",
                    f"  size_t channels_this_tile = {rows}u - (gemm_calls - 1u) * {bm}u;",
                    f"  if (channels_this_tile > {bm}u) channels_this_tile = {bm}u;",
                    "  if (dequant_calls == channels_this_tile) {",
                    f"    if (gemm_calls < {m_tiles}u) phase = 2;",
                    "    else {",
                    "      ++tile_index;",
                    f"      if (tile_index == {total_tiles}u) phase = 4; else phase = 1;",
                    "    }",
                    "  }",
                ]
        lines += ["}"]

    typed_parameters = ", ".join(
        f"MemRef{argument['rank']} *a{index}"
        for index, argument in enumerate(case["arguments"])
    )
    pointers = ", ".join(f"a{i}" for i in range(len(case["arguments"])))
    last = len(case["arguments"]) - 1
    rank = case["arguments"][0]["rank"]
    bad_pointers = ", ".join(
        ["NULL", *[f"a{i}" for i in range(1, len(case["arguments"]))]]
    )
    programs = total_tiles
    expected_gemm = total_tiles * m_tiles
    expected_dequant = total_tiles * (1 if depthwise else rows)
    lines += [
        f"extern void effnet_test_adapter({typed_parameters});",
        f"static void reject_invalid({typed_parameters}) {{",
        "  for (int test = 0; test < 7; ++test) {",
        "    pid_t child = fork();",
        '    require(child >= 0, "fork invalid descriptor probe");',
        "    if (!child) {",
        "      struct rlimit limit = {0, 0};",
        "      (void)setrlimit(RLIMIT_CORE, &limit);",
        f"      MemRef{rank} bad = *a0;",
        "      a0 = &bad;",
        f"      if (test == 0) effnet_test_adapter({bad_pointers});",
        "      if (test == 1) bad.offset = -1;",
        "      if (test == 2) bad.sizes[0] = 0;",
        f"      if (test == 3) bad.strides[{rank - 1}] = 2;",
        "      if (test == 4) bad.aligned = (char *)bad.aligned + 1;",
        f"      if (test == 5) a{last}->aligned = bad.aligned;",
        "      if (test == 6) { bad.aligned = (void *)(UINTPTR_MAX - 63u); bad.offset = 64; }",
        f"      effnet_test_adapter({pointers});",
        "      _Exit(0);",
        "    }",
        "    int status;",
        '    require(waitpid(child, &status, 0) == child, "wait invalid descriptor probe");',
        '    require(WIFSIGNALED(status) && (WTERMSIG(status) == SIGILL || WTERMSIG(status) == SIGTRAP), "invalid descriptor must trap before raw entry");',
        "  }",
        "}",
        f"void _mlir_ciface_kernel_{case['name']}({typed_parameters}) {{",
        '  require(++adapters == 1, "one adapter call from launch");',
        f"  reject_invalid({pointers});",
        f"  effnet_test_adapter({pointers});",
        '  require(write_begin_calls == 1, "initialization policy call count");',
        "  phase = 0; weight_calls = im2col_calls = gemm_calls = dequant_calls = 0;",
        "  gemm_total = dequant_total = tile_index = write_begin_calls = 0;",
        f"  effnet_test_adapter({pointers});",
        f'  require(phase == 4 && weight_calls == {rows}u && im2col_calls == {programs}u, "complete quantization stages");',
        f'  require(gemm_total == {expected_gemm}u && dequant_total == {expected_dequant}u, "complete GEMM and dequantization stages");',
        "  longjmp(completed, 1);",
        "}",
        "int main(void) {",
        "  for (size_t i = 0; i < VIT_WORKSPACE_BYTES; ++i) ((unsigned char *)workspace(0))[i] = 0xa9;",
        "  if (!setjmp(completed)) {",
        "    (void)launch();",
        '    require(0, "launch returned without entering its adapter");',
        "  }",
        f'  require(adapters == 1 && phase == 4 && weight_calls == {rows}u, "completed composite launch entry");',
        '  printf("ABI_SMOKE_PASS weight=%u im2col=%u gemm=%u dequant=%u numeric_oracle_executed=false\\n", (unsigned)weight_calls, (unsigned)im2col_calls, (unsigned)gemm_total, (unsigned)dequant_total);',
        "  return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_mock_source(case):
    """Check one native full-array call through the same descriptor oracle."""
    reference = dict(case, implementation="mock_native_bridge", grid=[1, 1, 1])
    source = mock_source(reference)
    shapes = argument_shapes(case)
    elements = [math.prod(shape) for shape in shapes]
    symbol = case["implementation_symbol"]
    if case["implementation"] == "hybrid_ordered_c":
        spec = case["constexprs"]
        signature = "size_t spatial, size_t channels, size_t block"
        checks = (
            f"  require(spatial == {spec['SPATIAL']}u && channels == {spec['C']}u "
            f'&& block == {spec["BLOCK"]}u, "native GAP specialization");'
        )
    else:
        signature = "size_t count"
        checks = f'  require(count == {elements[0]}u, "native SiLU full count");'
    return (
        source
        + "\n"
        + "\n".join(
            [
                f"void {symbol}(const float *input, float *output, {signature}) {{",
                checks,
                f"  Descriptor1D input_desc = {{(void *)input, (void *)input, 0, {elements[0]}, 1}};",
                f"  Descriptor1D output_desc = {{output, output, 0, {elements[1]}, 1}};",
                "  Unranked input_u = {1, &input_desc}, output_u = {1, &output_desc};",
                f"  _mlir_ciface_{case['symbol']}(&input_u, &output_u, 1, 1, 1, 0, 0, 0);",
                "}",
                "",
            ]
        )
    )


def mock_source(case):
    if case.get("implementation") in ("hybrid_rvv_c", "hybrid_ordered_c"):
        return native_mock_source(case)
    if case.get("implementation") == "w8a8_gemm":
        from quantized_linear_launch_check import mock_source as linear_mock

        return linear_mock(case)
    if is_quantized_conv_case(case):
        return composite_mock_source(case)
    arguments = case["arguments"]
    shapes = argument_shapes(case)
    gx, gy, gz = case["grid"]
    programs = math.prod(case["grid"])
    raw_parameters = [f"Unranked *u{i}" for i in range(len(arguments))]
    raw_parameters += [
        "int32_t gx",
        "int32_t gy",
        "int32_t gz",
        "int32_t px",
        "int32_t py",
        "int32_t pz",
    ]
    typed_parameters = ", ".join(
        f"MemRef{argument['rank']} *a{i}" for i, argument in enumerate(arguments)
    )
    pointers = ", ".join(f"a{i}" for i in range(len(arguments)))
    lines = [
        '#include "support.h"',
        "#include <setjmp.h>",
        "#include <signal.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <sys/resource.h>",
        "#include <sys/wait.h>",
        "#include <unistd.h>",
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
        "static jmp_buf completed;",
        "static size_t calls, adapters;",
        f"static uintptr_t first_pointers[{len(arguments)}];",
        'static void require(int ok, const char *what) { if (!ok) { fprintf(stderr, "ABI smoke failure: %s\\n", what); abort(); } }',
        f"void _mlir_ciface_{case['symbol']}({', '.join(raw_parameters)}) {{",
        "  uintptr_t arena = (uintptr_t)workspace(0);",
        f'  require(gx == {gx} && gy == {gy} && gz == {gz}, "launch grid");',
        f'  require(calls < {programs}, "excess raw entries");',
        f'  require(px == (int32_t)(calls / {gy * gz}) && py == (int32_t)((calls / {gz}) % {gy}) && pz == (int32_t)(calls % {gz}), "program coordinates and order");',
    ]
    lengths = []
    for i, (argument, shape) in enumerate(zip(arguments, shapes)):
        elements = math.prod(shape)
        lengths.append(elements * (1 if argument["dtype"] == "i8" else 4))
        lines += [
            f'  require(u{i} && u{i}->rank == 1 && u{i}->descriptor, "argument {i}: unranked rank-one descriptor");',
            f"  Descriptor1D *d{i} = (Descriptor1D *)u{i}->descriptor;",
            f'  require(d{i}->allocated == d{i}->aligned && d{i}->offset == 0 && d{i}->size == {elements} && d{i}->stride == 1, "argument {i}: contiguous physical extent");',
            f"  uintptr_t p{i} = (uintptr_t)d{i}->aligned;",
            f'  require(p{i} && !(p{i} & 63u) && p{i} <= UINTPTR_MAX - {lengths[i]}, "argument {i}: address and alignment");',
            f"  if (!calls) first_pointers[{i}] = p{i};",
            f'  require(first_pointers[{i}] == p{i}, "argument {i}: stable buffer");',
            f'  require(p{i} >= arena && p{i} + {lengths[i]} <= arena + VIT_WORKSPACE_BYTES, "argument {i}: arena extent");',
        ]
        for j in range(i):
            lines.append(
                f'  require(p{i} + {lengths[i]} <= p{j} || p{j} + {lengths[j]} <= p{i}, "arguments {j}/{i}: nonalias");'
            )
    last = len(arguments) - 1
    rank = arguments[0]["rank"]
    bad_pointers = ", ".join(["NULL", *[f"a{i}" for i in range(1, len(arguments))]])
    lines += [
        "  ++calls;",
        "}",
        f"extern void effnet_test_adapter({typed_parameters});",
        f"static void reject_invalid({typed_parameters}) {{",
        "  for (int test = 0; test < 7; ++test) {",
        "    pid_t child = fork();",
        '    require(child >= 0, "fork invalid descriptor probe");',
        "    if (!child) {",
        "      struct rlimit limit = {0, 0};",
        "      (void)setrlimit(RLIMIT_CORE, &limit);",
        f"      MemRef{rank} bad = *a0;",
        "      a0 = &bad;",
        f"      if (test == 0) effnet_test_adapter({bad_pointers});",
        "      if (test == 1) bad.offset = -1;",
        "      if (test == 2) bad.sizes[0] = 0;",
        f"      if (test == 3) bad.strides[{rank - 1}] = 2;",
        "      if (test == 4) bad.aligned = (char *)bad.aligned + 1;",
        f"      if (test == 5) a{last}->aligned = bad.aligned;",
        "      if (test == 6) { bad.aligned = (void *)(UINTPTR_MAX - 63u); bad.offset = 64; }",
        f"      effnet_test_adapter({pointers});",
        "      _Exit(0);",
        "    }",
        "    int status;",
        '    require(waitpid(child, &status, 0) == child, "wait invalid descriptor probe");',
        '    require(WIFSIGNALED(status) && (WTERMSIG(status) == SIGILL || WTERMSIG(status) == SIGTRAP), "invalid descriptor must trap before raw entry");',
        "  }",
        "}",
        f"void _mlir_ciface_kernel_{case['name']}({typed_parameters}) {{",
        '  require(++adapters == 1, "one adapter call from launch");',
        f"  reject_invalid({pointers});",
        f"  effnet_test_adapter({pointers});",
        f'  require(calls == {programs}, "complete grid executed");',
        "  longjmp(completed, 1);",
        "}",
        "int main(void) {",
        "  if (!setjmp(completed)) {",
        "    (void)launch();",
        '    require(0, "launch returned without entering its adapter");',
        "  }",
        f'  require(adapters == 1 && calls == {programs}, "completed launch entry");',
        '  printf("ABI_SMOKE_PASS programs=%zu invalid_descriptors=7 numeric_oracle_executed=false\\n", calls);',
        "  return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def check_case(case, args):
    destination = args.output / case["name"]
    destination.mkdir(parents=True, exist_ok=True)
    adapter = args.build_dir / case["name"] / "adapter.c"
    expected_programs = (
        1
        if case.get("implementation") in ("hybrid_rvv_c", "hybrid_ordered_c")
        else math.prod(case["grid"])
    )
    if case.get("implementation") == "w8a8_gemm":
        from quantized_linear_pipeline import stage_cases

        expected_programs = sum(math.prod(stage["grid"]) for stage in stage_cases(case))
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "validation": "native launch/descriptor/grid ABI smoke only",
        "numeric_oracle_executed": False,
        "simulator_executed": False,
        "workspace_bytes": 16 * 1024 * 1024,
        "expected_programs": expected_programs,
    }
    logs = []
    try:
        if not adapter.is_file():
            raise RuntimeError("missing generated adapter.c; export case first")
        mock = destination / "mock.c"
        mock.write_text(mock_source(case))
        adapter_object = destination / "adapter.o"
        executable = destination / "check-launch"
        flags = [
            args.cc,
            "-std=c11",
            "-O2",
            "-DHOST_TEST",
            "-DVIT_WORKSPACE_BYTES=16777216",
            "-I",
            str(ROOT),
        ]
        commands = [
            [
                *flags,
                f"-D_mlir_ciface_kernel_{case['name']}=effnet_test_adapter",
                "-c",
                str(adapter),
                "-o",
                str(adapter_object),
            ],
            [
                *flags,
                str(mock),
                case["launch_source"],
                str(ROOT / "support.c"),
                str(adapter_object),
                "-lm",
                "-o",
                str(executable),
            ],
            [str(executable)],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            logs.extend(["$ " + " ".join(command), result.stdout + result.stderr])
            if result.returncode:
                raise RuntimeError(
                    f"command returned {result.returncode}: "
                    + (result.stdout + result.stderr).strip()
                )
        record.update(
            status="ABI_SMOKE_PASS",
            programs=record["expected_programs"],
            invalid_descriptor_probes=7,
            launch_sha256=digest(case["launch_source"]),
            adapter_sha256=digest(adapter),
            mock_sha256=digest(mock),
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        record["reason"] = str(error)
        logs.append("BLOCKED: " + str(error))
    (destination / "check.log").write_text("\n".join(logs) + "\n")
    (destination / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{record['status']}: {case['name']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--output", type=Path, default=ROOT / "build/launch-check")
    parser.add_argument(
        "--cc", default=shutil.which("clang") or shutil.which("cc") or "cc"
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cases = [describe(name) for name in args.cases] if args.cases else inventory()
    records = [check_case(case, args) for case in cases]
    report = {
        "schema": 1,
        "validation": "ABI smoke only; no numerical or simulator validation",
        "status": "ABI_SMOKE_PASS"
        if all(record["status"] == "ABI_SMOKE_PASS" for record in records)
        else "PARTIAL",
        "cases": records,
    }
    report_path = args.report or args.output / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "ABI_SMOKE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

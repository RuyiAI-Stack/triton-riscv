#!/usr/bin/env python3
"""Exercise standalone launch and generated descriptor/grid adapters natively.

This is an ABI smoke check, not numerical or simulator validation. A mock raw
Triton entry checks every descriptor and grid coordinate of the first adapter
invocation. After that adapter returns, a test wrapper leaves launch() with longjmp before its numerical
oracle runs. No fake output values are computed or accepted as correct.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from build import argument_shapes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_mocks():
    """Check the composed patch adapter's four typed children, without math."""
    return r"""
static size_t patch_children, patch_buffers;
static uintptr_t patch_starts[16];
static size_t patch_lengths[16];
static uintptr_t patch_iq, patch_rs, patch_wq, patch_cs, patch_c;
static uintptr_t patch_buffer(void *pointer, size_t bytes) {
  uintptr_t p = (uintptr_t)pointer;
  require(p && !(p & 63u) && p <= UINTPTR_MAX - bytes,
          "patch child buffer alignment/extent");
  for (size_t i = 0; i < patch_buffers; ++i) {
    if (patch_starts[i] == p && patch_lengths[i] == bytes) return p;
    require(p + bytes <= patch_starts[i] ||
            patch_starts[i] + patch_lengths[i] <= p,
            "patch child distinct buffers overlap");
  }
  require(patch_buffers < 16, "patch buffer registry capacity");
  patch_starts[patch_buffers] = p;
  patch_lengths[patch_buffers++] = bytes;
  return p;
}
static uintptr_t patch_rank1(MemRef1 *d, int64_t size) {
  require(d && d->allocated == d->aligned && d->offset == 0 &&
          d->sizes[0] == size && d->strides[0] == 1,
          "patch child rank-one shape");
  return patch_buffer(d->aligned, (size_t)size * sizeof(float));
}
static uintptr_t patch_rank2(MemRef2 *d, int64_t rows, int64_t cols,
                             size_t element_bytes) {
  require(d && d->allocated == d->aligned && d->offset == 0 &&
          d->sizes[0] == rows && d->sizes[1] == cols &&
          d->strides[0] == cols && d->strides[1] == 1,
          "patch child rank-two contiguous shape");
  return patch_buffer(d->aligned, (size_t)rows * cols * element_bytes);
}
void _mlir_ciface_kernel_quantize_196x768(MemRef2 *x, MemRef2 *q,
                                        MemRef1 *scale) {
  require(calls == 196 && patch_children++ == 0,
          "patch im2col precedes activation quantization");
  require(patch_rank2(x, 196, 768, 4) == first_pointers[3],
          "patch quantization reads raw im2col output");
  patch_iq = patch_rank2(q, 196, 768, 1);
  patch_rs = patch_rank1(scale, 196);
}
void _mlir_ciface_kernel_quantize_768x768(MemRef2 *x, MemRef2 *q,
                                        MemRef1 *scale) {
  require(patch_children++ == 1, "patch weight quantization order");
  require(patch_rank2(x, 768, 768, 4) == first_pointers[1],
          "patch quantization reads launch weight");
  patch_wq = patch_rank2(q, 768, 768, 1);
  patch_cs = patch_rank1(scale, 768);
}
void _mlir_ciface_kernel_matmul_i8_overwrite_196x768x768(MemRef2 *a, MemRef2 *b,
                                             MemRef2 *c) {
  require(patch_children++ == 2, "patch integer matmul order");
  require(patch_rank2(a, 196, 768, 1) == patch_iq &&
          patch_rank2(b, 768, 768, 1) == patch_wq,
          "patch matmul consumes quantized operands");
  patch_c = patch_rank2(c, 196, 768, 4);
}
void _mlir_ciface_kernel_dequantize_196x768(MemRef2 *x, MemRef1 *row,
                                          MemRef1 *column, MemRef2 *y) {
  require(patch_children++ == 3, "patch dequantization order");
  require(patch_rank2(x, 196, 768, 4) == patch_c &&
          patch_rank1(row, 196) == patch_rs &&
          patch_rank1(column, 768) == patch_cs,
          "patch dequantization consumes matmul and both scales");
  (void)patch_rank2(y, 196, 768, 4);
}
"""


def mock_source(case):
    arguments = case["arguments"]
    shapes = argument_shapes(case)
    gx, gy, gz = case["grid"]
    programs = gx * gy * gz
    raw_params = [f"Unranked *u{i}" for i in range(len(arguments))]
    raw_params += ["int32_t gx", "int32_t gy", "int32_t gz"]
    raw_params += ["int32_t px", "int32_t py", "int32_t pz"]
    typed_params = ", ".join(
        f"MemRef{argument['rank']} *a{i}"
        for i, argument in enumerate(arguments)
    )
    raw_symbol = "_mlir_ciface_" + case["symbol"]
    launch_symbol = "_mlir_ciface_kernel_" + case["name"]
    patch = case["family"] == "patch_embed"
    lines = [
        '#include "support.h"',
        "#include <setjmp.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
        "static jmp_buf completed;",
        "static size_t calls, adapters;",
        f"static uintptr_t first_pointers[{len(arguments)}];",
        'static void require(int ok, const char *what) { if (!ok) { fprintf(stderr, "ABI smoke failure: %s\\n", what); abort(); } }',
    ]
    if patch:
        lines.append(patch_mocks())
    lines += [
        f"void {raw_symbol}({', '.join(raw_params)}) {{",
        "  uintptr_t arena_begin = (uintptr_t)workspace(0);",
        f'  require(gx == {gx} && gy == {gy} && gz == {gz}, "launch grid");',
        f'  require(calls < {programs}, "excess raw entries");',
        f'  require(px == (int32_t)(calls / {gy * gz}) && py == (int32_t)((calls / {gz}) % {gy}) && pz == (int32_t)(calls % {gz}), "program coordinates and order");',
    ]
    byte_lengths = []
    for i, (argument, shape) in enumerate(zip(arguments, shapes)):
        elements = 1
        for extent in shape:
            elements *= extent
        item_bytes = 1 if argument["dtype"] == "i8" else 4
        byte_lengths.append(elements * item_bytes)
        lines += [
            f'  require(u{i} && u{i}->rank == 1 && u{i}->descriptor, "argument {i}: unranked rank-one descriptor");',
            f"  Descriptor1D *d{i} = (Descriptor1D *)u{i}->descriptor;",
            f'  require(d{i}->allocated == d{i}->aligned && d{i}->offset == 0 && d{i}->size == {elements} && d{i}->stride == 1, "argument {i}: contiguous physical shape");',
            f"  uintptr_t p{i} = (uintptr_t)d{i}->aligned;",
            f'  require(p{i} && !(p{i} & 63u) && p{i} <= UINTPTR_MAX - {byte_lengths[i]}, "argument {i}: address and alignment");',
            f"  if (!calls) first_pointers[{i}] = p{i};",
            f'  require(first_pointers[{i}] == p{i}, "argument {i}: stable buffer");',
        ]
        if not patch or i != 3:
            lines.append(
                f'  require(p{i} >= arena_begin && p{i} + {byte_lengths[i]} <= arena_begin + VIT_WORKSPACE_BYTES, "argument {i}: arena extent");'
            )
        if patch:
            lines.append(f"  patch_buffer((void *)p{i}, {byte_lengths[i]});")
        for j in range(i):
            lines.append(
                f'  require(p{i} + {byte_lengths[i]} <= p{j} || p{j} + {byte_lengths[j]} <= p{i}, "arguments {j}/{i}: nonalias");'
            )
    lines += [
        "  ++calls;",
        "}",
        f"extern void vit_test_adapter({typed_params});",
        f"void {launch_symbol}({typed_params}) {{",
        '  require(++adapters == 1, "one adapter call from launch");',
    ]
    if patch:
        lines.append(
            "  patch_buffer(a3->aligned, 196u * 768u * sizeof(float));"
        )
    lines += [
        "  vit_test_adapter("
        + ", ".join(f"a{i}" for i in range(len(arguments)))
        + ");",
        f'  require(calls == {programs}, "complete grid executed");',
    ]
    if patch:
        lines.append(
            '  require(patch_children == 4, "all patch composition children executed");'
        )
    lines += [
        "  longjmp(completed, 1);",
        "}",
        "extern int launch(void);",
        "int main(void) {",
        "  if (!setjmp(completed)) {",
        "    (void)launch();",
        '    require(0, "launch returned without entering its adapter");',
        "  }",
        f'  require(adapters == 1 && calls == {programs}, "completed launch entry");',
        '  printf("ABI_SMOKE_PASS programs=%zu numeric_oracle_executed=false\\n", calls);',
        "  return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def check_case(case, args):
    destination = args.output / case["name"]
    destination.mkdir(parents=True, exist_ok=True)
    adapter = args.build_dir / case["name"] / "adapter.c"
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "validation": "native launch/descriptor/grid ABI smoke only",
        "numeric_oracle_executed": False,
        "simulator_executed": False,
        "launch_adapter_invocations": 1,
        "scope": "first launch adapter invocation, including its entire grid",
        "expected_programs": case["grid"][0]
        * case["grid"][1]
        * case["grid"][2],
    }
    logs = []
    try:
        if not adapter.is_file():
            raise RuntimeError("missing generated adapter.c; export case first")
        mock = destination / "mock.c"
        mock.write_text(mock_source(case))
        adapter_object = destination / "adapter.o"
        executable = destination / "check-launch"
        flags = [args.cc, "-std=c11", "-O2", "-DHOST_TEST", "-I", str(ROOT)]
        commands = [
            [
                *flags,
                "-D_mlir_ciface_kernel_" + case["name"] + "=vit_test_adapter",
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
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            logs.append("$ " + " ".join(command))
            logs.append(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"command returned {result.returncode}: "
                    + (result.stdout + result.stderr).strip()
                )
        record.update(
            status="ABI_SMOKE_PASS",
            programs=record["expected_programs"],
            launch_sha256=digest(case["launch_source"]),
            adapter_sha256=digest(adapter),
            mock_sha256=digest(mock),
        )
        if case["family"] == "patch_embed":
            record["composition_children_validated"] = [
                "quantize_196x768",
                "quantize_768x768",
                "matmul_i8_196x768x768",
                "dequantize_196x768",
            ]
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        record["reason"] = str(error)
        logs.append("BLOCKED: " + str(error))
    (destination / "check.log").write_text("\n".join(logs) + "\n")
    (destination / "result.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(f"{record['status']}: {case['name']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--build-dir", type=Path, default=ROOT / "build/kernels"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/launch-check"
    )
    parser.add_argument(
        "--cc", default=shutil.which("clang") or shutil.which("cc") or "cc"
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cases = (
        [describe(name) for name in args.cases] if args.cases else inventory()
    )
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

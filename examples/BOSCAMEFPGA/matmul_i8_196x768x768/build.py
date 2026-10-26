#!/usr/bin/env python3
"""Triton INT8 matmul inputs for the NH/E6 BOSCAME image.

The makefile owns the full lowering pipeline::

    external export_contract(ASTSource, target=boscame)
      -> kernel.frontend.mlir --kernel-contract-verify
      -> kernel.bufferized.mlir --bosc-ame-plan
      -> kernel.planned.mlir --bosc-ame-emit
      -> kernel.boscame.mlir --lower-bosc-ame --buddy-to-llvmir -> kernel.ll

This file owns the real Triton frontend export, the typed unranked ABI adapter
and the generated C shape header used by the launcher.  The shape and tile
constants are ``tl.constexpr`` globals in kernels.py; this file only derives
the launch grid (:data:`SHAPE`) and emits their fixed ABI.

Logical problem (``kernels.py``)::

    A[i8, 196, 768] row-major @ B[i8, 768, 768] physical [N,K]
      -> C[i32, 196, 768]

The AME tile contract requested by this example is ``BM=32``, ``BN=BK=128``.
The AME register file caps one tile dimension at 64, so the planner lowers
``BN`` into two 64-column groups and ``BK`` into 64-wide K blocks.  M=196 is
not a multiple of BM=32. The explicit launch-grid contract lets the frontend
specialize full programs and the four-row boundary program. The kernel keeps
its `tl.load`/`tl.store` masks and ``K/BK`` reduction loop; only the 196 logical
rows of C are read or written.

Build artifacts (``build/matmul_i8_196x768x768/``) are limited to IR and the
final image:

    kernel.ttir  kernel.frontend.mlir  kernel.bufferized.mlir
    kernel.planned.mlir  kernel.boscame.mlir  kernel.llvm.mlir  kernel.ll
    matmul_i8_196x768x768.bin

Generated C, assembly, objects, the ELF and the audit report are intermediates.

Run with the boscame Python 3.12 environment:

    python build.py --ttir-only       # Triton frontend -> kernel.ttir
    python build.py --adapter experimental-boscame
    python build.py --config
    python build.py --launch-grid
    python build.py --check-abi path/to/kernel.ll
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BUILD_ROOT = Path(
    os.environ.get("MATMUL_I8_BUILD_ROOT", str(ROOT / "build"))
).resolve()
OUT = BUILD_ROOT / "matmul_i8_196x768x768"
sys.path.insert(0, str(ROOT))
import kernels  # noqa: E402

# The kernel's global constants come from kernels.py; only the derived launch
# grid, the generated ABI bridge and image-layout checks live here.
NAME = "matmul_i8_196x768x768"
SYMBOL = "triton_" + NAME
KERNEL = "matmul_i8"
SIGNATURE = {"A": "*i8", "B": "*i8", "C": "*i32"}


def shape_values():
    """Return the one source of truth for the Triton shape and tile sizes."""
    values = {
        name: int(getattr(kernels, name).value)
        for name in ("M", "N", "K", "BM", "BN", "BK")
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("matmul shape and tile sizes must be positive")
    for extent, tile, label in (
        (values["N"], values["BN"], "N/BN"),
        (values["K"], values["BK"], "K/BK"),
    ):
        if extent % tile:
            raise ValueError(f"{label} must be an integer tile count")
    values["GRID_M"] = (values["M"] + values["BM"] - 1) // values["BM"]
    values["GRID_N"] = values["N"] // values["BN"]
    return values


SHAPE = shape_values()

# The checked-in linker script reserves these fixed windows.  Keep the checks
# here so an oversized shape reports its cause before clang/ld or the board
# does; the values mirror link.ld and the AME v0.5 tile contract.
AME_MAX_TILE = 64
DDR_SOURCE_CAPACITY = 0x80200000 - 0x80100000
DDR_OUTPUT_CAPACITY = 0x802E0000 - 0x80200000
DMA_GUARD_BYTES = 2 * 256 * 4


def validate_target_shape():
    # BN is a logical program tile: the planner splits it into 64-column AME
    # groups, so only the hardware M tile has to fit one register tile.  The
    # boundary M tile is masked in the kernel (M is not a multiple of BM).
    if SHAPE["BM"] > AME_MAX_TILE:
        raise ValueError(
            f"BM={SHAPE['BM']} exceeds the AME tile capacity "
            f"{AME_MAX_TILE}; use an M tile <= 64"
        )
    source_bytes = (
        SHAPE["M"] * SHAPE["K"] + SHAPE["N"] * SHAPE["K"] + 2 * DMA_GUARD_BYTES
    )
    output_bytes = SHAPE["M"] * SHAPE["N"] * 4 + DMA_GUARD_BYTES
    if source_bytes > DDR_SOURCE_CAPACITY:
        raise ValueError(
            "A/B operands need "
            f"{source_bytes} bytes, but link.ld reserves "
            f"{DDR_SOURCE_CAPACITY} bytes before .dma_output"
        )
    if output_bytes > DDR_OUTPUT_CAPACITY:
        raise ValueError(
            "C output needs "
            f"{output_bytes} bytes, but link.ld reserves "
            f"{DDR_OUTPUT_CAPACITY} bytes (0x80200000..0x802e0000)"
        )


def config_source(validate=True):
    """Emit the C header consumed by the launcher.

    Keeping the header generated from ``kernels.py`` makes a changed Triton
    shape fail at compile time instead of reaching the adapter with stale
    descriptors.
    """
    if validate:
        validate_target_shape()
    return f"""/* Generated by build.py from kernels.py; do not edit. */
#ifndef MATMUL_I8_CONFIG_H
#define MATMUL_I8_CONFIG_H
#define MATMUL_I8_M {SHAPE["M"]}u
#define MATMUL_I8_N {SHAPE["N"]}u
#define MATMUL_I8_K {SHAPE["K"]}u
#define MATMUL_I8_BM {SHAPE["BM"]}u
#define MATMUL_I8_BN {SHAPE["BN"]}u
#define MATMUL_I8_BK {SHAPE["BK"]}u
#define MATMUL_I8_GRID_M {SHAPE["GRID_M"]}u
#define MATMUL_I8_GRID_N {SHAPE["GRID_N"]}u
#define MATMUL_I8_KERNEL _mlir_ciface_kernel_{NAME}
#endif
"""


def adapter_source_experimental():
    """Typed unranked descriptors for the experimental BOSCAME C wrapper.

    Descriptor sizes/offsets are elements, including i32 C. The kernel takes
    three buffers; the launcher's aligned, nonalias contiguous buffers satisfy
    the DMA ABI. The grid covers the padded M extent so the masked boundary
    tile is launched and the kernel itself skips the rows past M.
    """
    validate_target_shape()
    grid_m = SHAPE["GRID_M"]
    grid_n = SHAPE["GRID_N"]
    return f"""#include <stdint.h>
#if defined(MATMUL_I8_TRACE)
extern void nh_e6_progress(uint32_t marker);
#define MATMUL_I8_MARK(value) nh_e6_progress(value)
#else
#define MATMUL_I8_MARK(value) ((void)0)
#endif
typedef struct {{ void *allocated, *aligned; int64_t offset; int64_t sizes[2], strides[2]; }} MemRef2;
typedef struct {{ void *allocated, *aligned; int64_t offset, size, stride; }} Descriptor1D;
typedef struct {{ int64_t rank; void *descriptor; }} Unranked;
extern void _mlir_ciface_{SYMBOL}(Unranked *, Unranked *, Unranked *,
                     int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_{NAME}(MemRef2 *a, MemRef2 *b, MemRef2 *c) {{
  void *ap = (int8_t *)a->aligned + a->offset;
  void *bp = (int8_t *)b->aligned + b->offset;
  void *cp = (int32_t *)c->aligned + c->offset;
  if (((uintptr_t)ap | (uintptr_t)bp | (uintptr_t)cp) & 63u) __builtin_trap();
  if (a->sizes[0] < {SHAPE["M"]} || a->sizes[1] < {SHAPE["K"]} ||
      b->sizes[0] < {SHAPE["N"]} || b->sizes[1] < {SHAPE["K"]} ||
      c->sizes[0] < {SHAPE["M"]} || c->sizes[1] < {SHAPE["N"]} ||
      a->strides[0] != {SHAPE["K"]} || a->strides[1] != 1 ||
      b->strides[0] != {SHAPE["K"]} || b->strides[1] != 1 ||
      c->strides[0] != {SHAPE["N"]} || c->strides[1] != 1) __builtin_trap();
  uintptr_t starts[3] = {{(uintptr_t)ap, (uintptr_t)bp, (uintptr_t)cp}};
  uintptr_t lengths[3] = {{{SHAPE["M"]} * {SHAPE["K"]}, {SHAPE["N"]} * {SHAPE["K"]}, {SHAPE["M"]} * {SHAPE["N"]} * 4}};
  for (int i = 0; i < 3; ++i) {{
    if (lengths[i] > UINTPTR_MAX - starts[i]) __builtin_trap();
    for (int j = 0; j < i; ++j)
      if (starts[i] < starts[j] + lengths[j] && starts[j] < starts[i] + lengths[i]) __builtin_trap();
  }}
  Descriptor1D da = {{a->allocated, ap, 0, (int64_t){SHAPE["M"]} * {SHAPE["K"]}, 1}};
  Descriptor1D db = {{b->allocated, bp, 0, (int64_t){SHAPE["N"]} * {SHAPE["K"]}, 1}};
  Descriptor1D dc = {{c->allocated, cp, 0, (int64_t){SHAPE["M"]} * {SHAPE["N"]}, 1}};
  Unranked ma = {{1, &da}}, mb = {{1, &db}}, mc = {{1, &dc}};
  for (int32_t x = 0; x < {grid_m}; ++x)
    for (int32_t y = 0; y < {grid_n}; ++y)
      for (int32_t z = 0; z < 1; ++z) {{
        MATMUL_I8_MARK(0x1000u + (uint32_t)((x * {grid_n} + y) & 0xfffu));
        _mlir_ciface_{SYMBOL}(&ma, &mb, &mc, {grid_m}, {grid_n}, 1, x, y, z);
      }}
}}
"""


def export_frontend():
    """Export the real kernel through the installed triton-riscv AOT API."""
    validate_target_shape()
    sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
    from contract_frontend import export_kernel

    artifact = export_kernel(
        getattr(kernels, KERNEL),
        SIGNATURE,
        {},
        (SHAPE["GRID_M"], SHAPE["GRID_N"], 1),
        SYMBOL,
        OUT,
    )
    if "@" + SYMBOL + "(" not in artifact.ttir:
        raise RuntimeError("missing compiled entry " + SYMBOL)
    print("exported " + NAME + " with verified Contract", flush=True)


def verify_abi(path):
    contents = Path(path).read_text()
    match = re.search(
        r"define\s+void\s+@" + re.escape(SYMBOL) + r"\((.*?)\)", contents, re.S
    )
    if not match:
        raise RuntimeError("Missing Triton entry definition: " + SYMBOL)
    actual = [part.strip().split()[0] for part in match.group(1).split(",")]
    expected = [value for _name in SIGNATURE for value in ("i64", "ptr")] + [
        "i32"
    ] * 6
    if actual != expected:
        raise RuntimeError(
            f"Triton ABI changed: expected {expected}, got {actual}"
        )
    if re.search(
        r"\b(?:call|declare)\b[^\n]*@(?:malloc|aligned_alloc|realloc|free)\b",
        contents,
    ):
        raise RuntimeError(
            "Unbounded per-grid heap allocation remains; stack promotion must remove it"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, help="Build root (defaults to build/)"
    )
    parser.add_argument(
        "--export-contract",
        "--ttir-only",
        dest="ttir_only",
        action="store_true",
        help="Export TTIR, verified Contract and frontend manifest",
    )
    parser.add_argument(
        "--check-abi",
        type=Path,
        help="Verify the typed entry LLVM ABI and absence of heap allocation",
    )
    parser.add_argument(
        "--adapter",
        choices=("experimental-boscame",),
        help="Only write the launcher adapter for that path",
    )
    parser.add_argument(
        "--launch-grid",
        action="store_true",
        help="Print the frontend launch-grid contract from the kernel constants",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Only write the generated C shape header for the launcher",
    )
    args = parser.parse_args()
    if args.check_abi:
        verify_abi(args.check_abi)
        return
    global BUILD_ROOT, OUT
    if args.output is not None:
        BUILD_ROOT = args.output.resolve()
        OUT = BUILD_ROOT / NAME
    if args.adapter is not None:
        sys.stdout.write(adapter_source_experimental())
        return
    if args.config:
        sys.stdout.write(config_source())
        return
    if args.launch_grid:
        print(f"{SYMBOL}:{SHAPE['GRID_M']},{SHAPE['GRID_N']},1")
        return
    if args.ttir_only:
        export_frontend()
        return
    raise SystemExit(
        "build.py only generates frontend/ABI inputs; use make for the image"
    )


if __name__ == "__main__":
    main()

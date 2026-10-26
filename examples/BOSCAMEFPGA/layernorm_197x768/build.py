#!/usr/bin/env python3
"""Triton LayerNorm inputs for the NH/E6 BOSCAME image.

The makefile owns the full lowering pipeline::

    external export_contract(ASTSource, target=boscame)
      -> kernel.frontend.mlir --kernel-contract-verify
      -> kernel.bufferized.mlir --bosc-ame-plan
      -> kernel.planned.mlir --bosc-ame-emit
      -> kernel.boscame.mlir --lower-bosc-ame --buddy-to-llvmir -> kernel.ll

This file owns the real Triton frontend export, the fixed problem size and the
typed unranked ABI adapter used by the launcher.  Only `kernel.ttir` is
generated here; the remaining stages belong to the makefile.

    layernorm_197x768: X[197,768] f32, Weight[768] f32, Bias[768] f32
                       -> Out[197,768] f32, one program per row

Run with the boscame Python environment:

    python build.py --ttir-only       # Triton frontend -> kernel.ttir
    python build.py --adapter         # typed launcher ABI bridge to stdout
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BUILD_ROOT = Path(
    os.environ.get("LAYERNORM_197X768_BUILD_ROOT", str(ROOT / "build"))
).resolve()
OUT = BUILD_ROOT / "layernorm_197x768"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixed problem size (the only place these numbers are defined).
# ---------------------------------------------------------------------------
ROWS, WIDTH = 197, 768  # logical Out[ROWS,WIDTH] = layernorm(X[ROWS,WIDTH])
BLOCK = 1 << (WIDTH - 1).bit_length()  # 1024: next power of two >= WIDTH
EPS = 1.0e-12
GRID = (ROWS, 1, 1)  # one program per row
NAME = "layernorm_197x768"
SYMBOL = "triton_" + NAME
KERNEL = "layernorm"
SIGNATURE = {
    "X": "*fp32",
    "Weight": "*fp32",
    "Bias": "*fp32",
    "Out": "*fp32",
}
CONSTEXPRS = {"ROWS": ROWS, "WIDTH": WIDTH, "BLOCK": BLOCK, "EPS": EPS}

assert BLOCK >= WIDTH and BLOCK & (BLOCK - 1) == 0, "BLOCK must be a power of 2"


def verify_ttir(ttir):
    """Fail the export unless the TTIR still carries the operator's structure.

    Two `tt.reduce` (mean and variance over the hidden dimension) plus the
    `math.rsqrt` of the shifted variance are the whole point of the kernel;
    an export that loses one of them is not this operator.
    """
    missing = [
        token for token in ("@" + SYMBOL, "math.rsqrt") if token not in ttir
    ]
    if ttir.count('"tt.reduce"') != 2:
        missing.append("two tt.reduce")
    if ttir.count("tt.store") != 1:
        missing.append("one tt.store")
    if missing:
        raise RuntimeError(
            "exported TTIR is not " + NAME + ": missing " + ", ".join(missing)
        )


def export_frontend():
    """Export the real kernel through the installed triton-riscv AOT API."""
    import kernel

    sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
    from contract_frontend import export_kernel

    artifact = export_kernel(
        getattr(kernel, KERNEL), SIGNATURE, CONSTEXPRS, GRID, SYMBOL, OUT
    )
    verify_ttir(artifact.ttir)
    print("exported " + NAME + " with verified Contract", flush=True)


def adapter_source():
    """Emit the typed-unranked C wrapper bridge with an enforced DMA ABI."""
    return f"""#include <stdint.h>
#if defined(LAYERNORM_TRACE)
extern void nh_e6_progress(uint32_t marker);
#define LAYERNORM_MARK(value) nh_e6_progress(value)
#else
#define LAYERNORM_MARK(value) ((void)0)
#endif
typedef struct {{ void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; }} MemRef2;
typedef struct {{ void *allocated, *aligned; int64_t offset, size, stride; }} MemRef1;
typedef struct {{ int64_t rank; void *descriptor; }} Unranked;
extern void _mlir_ciface_{SYMBOL}(Unranked *, Unranked *, Unranked *, Unranked *,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_{NAME}(MemRef2 *x, MemRef1 *weight, MemRef1 *bias, MemRef2 *out) {{
  float *xp = (float *)x->aligned + x->offset;
  float *wp = (float *)weight->aligned + weight->offset;
  float *bp = (float *)bias->aligned + bias->offset;
  float *op = (float *)out->aligned + out->offset;
  if (((uintptr_t)xp | (uintptr_t)wp | (uintptr_t)bp | (uintptr_t)op) & 63u) __builtin_trap();
  if (x->sizes[0] < {ROWS} || x->sizes[1] < {WIDTH} ||
      out->sizes[0] < {ROWS} || out->sizes[1] < {WIDTH} ||
      weight->size < {WIDTH} || bias->size < {WIDTH} ||
      x->strides[0] != {WIDTH} || x->strides[1] != 1 ||
      out->strides[0] != {WIDTH} || out->strides[1] != 1 ||
      weight->stride != 1 || bias->stride != 1) __builtin_trap();
  uintptr_t starts[4] = {{(uintptr_t)xp, (uintptr_t)wp, (uintptr_t)bp, (uintptr_t)op}};
  uintptr_t lengths[4] = {{{ROWS} * {WIDTH} * 4, {WIDTH} * 4, {WIDTH} * 4, {ROWS} * {WIDTH} * 4}};
  for (int i = 0; i < 4; ++i) {{
    if (lengths[i] > UINTPTR_MAX - starts[i]) __builtin_trap();
    for (int j = 0; j < i; ++j)
      if (starts[i] < starts[j] + lengths[j] && starts[j] < starts[i] + lengths[i]) __builtin_trap();
  }}
  MemRef1 dx = {{x->allocated, xp, 0, (int64_t){ROWS} * {WIDTH}, 1}};
  MemRef1 dw = {{weight->allocated, wp, 0, {WIDTH}, 1}};
  MemRef1 db = {{bias->allocated, bp, 0, {WIDTH}, 1}};
  MemRef1 dout = {{out->allocated, op, 0, (int64_t){ROWS} * {WIDTH}, 1}};
  Unranked mx = {{1, &dx}}, mw = {{1, &dw}}, mb = {{1, &db}}, mo = {{1, &dout}};
  for (int32_t index = 0; index < {ROWS}; ++index) {{
#if defined(LAYERNORM_REVERSE)
    int32_t row = {ROWS} - 1 - index;
#else
    int32_t row = index;
#endif
    LAYERNORM_MARK(0x1000u + (uint32_t)row);
    _mlir_ciface_{SYMBOL}(&mx, &mw, &mb, &mo, {ROWS}, 1, 1, row, 0, 0);
  }}
}}
"""


def verify_abi(path):
    """Verify the typed entry LLVM ABI and absence of heap allocation."""
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
        "--adapter",
        action="store_true",
        help="Write the typed-unranked C ABI adapter to stdout",
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
        "--output", type=Path, help="Build root (defaults to build/)"
    )
    args = parser.parse_args()
    if args.check_abi:
        verify_abi(args.check_abi)
        return
    if args.adapter:
        sys.stdout.write(adapter_source())
        return
    global BUILD_ROOT, OUT
    if args.output is not None:
        BUILD_ROOT = args.output.resolve()
        OUT = BUILD_ROOT / NAME
    if args.ttir_only:
        export_frontend()
        return
    raise SystemExit(
        "build.py only generates frontend/ABI inputs; use make for the image"
    )


if __name__ == "__main__":
    main()

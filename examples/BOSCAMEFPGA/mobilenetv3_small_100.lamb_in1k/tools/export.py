#!/usr/bin/env python3
"""Export one local Triton kernel and its typed ABI adapter.

Compiler commands and their file dependencies live in ../common.mk.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]

sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool(name):
    override = os.environ.get(name.upper().replace("-", "_"))
    if override:
        return override
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"missing tool {name}; set {name.upper().replace('-', '_')}"
        )
    return found


def argument_shapes(case):
    """Physical entry shapes, including the graph-facing patch ABI."""
    family, shape = case["family"], case["shape"]
    spec = case["constexprs"]
    if family == "linear_f32":
        m, n, k = spec["M"], spec["N"], spec["K"]
        return [(m, k), (n, k), (n,), (m, n)]
    if family == "conv2d_f32":
        return [
            (1, spec["C"], spec["H"], spec["W"]),
            (spec["OC"], spec["C"], spec["KH"], spec["KW"]),
            (spec["OC"],),
            (1, spec["OC"], spec["OH"], spec["OW"]),
        ]
    if family == "dwconv2d_f32":
        return [
            (1, spec["C"], spec["H"], spec["W"]),
            (spec["C"], 1, spec["KH"], spec["KW"]),
            (spec["C"],),
            (1, spec["C"], spec["OH"], spec["OW"]),
        ]
    if family == "gap":
        return [tuple(shape), (1, shape[1])]
    if family == "linear" and len(case["arguments"]) == 4:
        return [(spec["M"], spec["K"]), (spec["N"], spec["K"]),
                (spec["N"],), (spec["M"], spec["N"])]
    if family in ("linear", "matmul_i8", "matmul_i8_overwrite"):
        m, n, k = shape
        return [(m, k), (n, k), (m, n)]
    if family.startswith("attention_"):
        h, m, n, k = shape
        return [(h, m, k), (h, k, n), (h, m, n)]
    if family == "patch_embed":
        return [(1, 3, 224, 224), (768, 3, 16, 16), (768,), (1, 768, 14, 14)]
    if family == "layernorm":
        return [tuple(shape), (shape[-1],), (shape[-1],), tuple(shape)]
    if family == "bias_add":
        return [tuple(shape), (shape[-1],), tuple(shape)]
    if family == "per_token_quantization":
        return [tuple(shape), tuple(shape), (shape[0],)]
    if family == "dequantization":
        return [tuple(shape), (shape[0],), (shape[1],), tuple(shape)]
    if family == "residual_add":
        n = case["constexprs"]["COUNT"]
        return [(n,), (n,), (n,)]
    if family in ("relu", "hardsigmoid", "hardswish"):
        n = case["constexprs"]["COUNT"]
        return [(n,), (n,)]
    if family == "se_mul":
        spec = case["constexprs"]
        channels = spec["COUNT"] // spec["SPATIAL"]
        return [(spec["COUNT"],), (channels,), (spec["COUNT"],)]
    if family == "mean_hw":
        spec = case["constexprs"]
        return [(spec["C"] * spec["SPATIAL"],), (spec["C"],)]
    if family == "linear":
        spec = case["constexprs"]
        return [(spec["M"], spec["K"]), (spec["N"], spec["K"]),
                (spec["N"],), (spec["M"], spec["N"])]
    if family == "depthwise_conv2d":
        spec = case["constexprs"]
        return [
            (1, spec["C"], spec["H"], spec["W"]),
            (spec["C"], 1, spec["KH"], spec["KW"]),
            (spec["C"],),
            (1, spec["C"], spec["OH"], spec["OW"]),
        ]
    if family == "pointwise_conv2d":
        spec = case["constexprs"]
        return [
            (1, spec["CIN"], spec["H"], spec["W"]),
            (spec["COUT"], spec["CIN"], 1, 1),
            (spec["COUT"],),
            (1, spec["COUT"], spec["H"], spec["W"]),
        ]
    if family == "conv_stem":
        spec = case["constexprs"]
        return [
            (1, spec["CIN"], spec["H"], spec["W"]),
            (spec["COUT"], spec["CIN"], spec["KH"], spec["KW"]),
            (spec["COUT"],),
            (1, spec["COUT"], spec["OH"], spec["OW"]),
        ]
    return [tuple(shape)] * len(case["arguments"])


def adapter_source(case):
    """The same typed-unranked descriptor/grid bridge as matmul128."""
    arguments = case["arguments"]
    shapes = argument_shapes(case)
    params = ", ".join(
        f"MemRef{arg['rank']} *a{i}" for i, arg in enumerate(arguments)
    )
    lines = [
        "/* Generated from this case; offsets and lengths are in elements. */",
        '#include "support.h"',
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
        f"extern void _mlir_ciface_{case['symbol']}("
        + ", ".join(["Unranked *"] * len(arguments) + ["int32_t"] * 6)
        + ");",
    ]
    patch = case["family"] == "patch_embed"
    if patch:
        lines += [
            "extern void _mlir_ciface_kernel_quantize_196x768(MemRef2 *, MemRef2 *, MemRef1 *);",
            "extern void _mlir_ciface_kernel_quantize_768x768(MemRef2 *, MemRef2 *, MemRef1 *);",
            "extern void _mlir_ciface_kernel_matmul_i8_overwrite_196x768x768(MemRef2 *, MemRef2 *, MemRef2 *);",
            "extern void _mlir_ciface_kernel_dequantize_196x768(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);",
            "static float patch_im2col[196 * 768] __attribute__((aligned(64)));",
            "static int8_t patch_im2col_q[196 * 768] __attribute__((aligned(64)));",
            "static float patch_im2col_scale[196] __attribute__((aligned(64)));",
            "static int8_t patch_weight_q[768 * 768] __attribute__((aligned(64)));",
            "static float patch_weight_scale[768] __attribute__((aligned(64)));",
            "static int32_t patch_acc[196 * 768] __attribute__((aligned(64)));",
            "static float patch_gemm[196 * 768] __attribute__((aligned(64)));",
        ]
    lines += [f"void _mlir_ciface_kernel_{case['name']}({params}) {{"]
    item_sizes = {"f32": 4, "i32": 4, "i8": 1}
    for i, (arg, shape) in enumerate(zip(arguments, shapes)):
        elements = 1
        for extent in shape:
            elements *= extent
        size = item_sizes[arg["dtype"]]
        lines += [f"  if (!a{i} || a{i}->offset < 0) __builtin_trap();"]
        stride = 1
        for axis in reversed(range(len(shape))):
            lines += [
                f"  if (a{i}->sizes[{axis}] < {shape[axis]} || a{i}->strides[{axis}] != {stride}) __builtin_trap();"
            ]
            stride *= shape[axis]
        lines += [
            f"  if (!a{i}->aligned || (uint64_t)a{i}->offset > (UINTPTR_MAX - (uintptr_t)a{i}->aligned) / {size}) __builtin_trap();",
            f"  uintptr_t p{i} = (uintptr_t)a{i}->aligned + (uint64_t)a{i}->offset * {size};",
            f"  if (!p{i} || (p{i} & 63u) || p{i} > UINTPTR_MAX - {elements * size}) __builtin_trap();",
            f"  Descriptor1D d{i} = {{(void *)p{i}, (void *)p{i}, 0, {elements}, 1}};",
            f"  Unranked u{i} = {{1, &d{i}}};",
        ]
        for j in range(i):
            other_elements = 1
            for extent in shapes[j]:
                other_elements *= extent
            other_size = item_sizes[arguments[j]["dtype"]]
            lines += [
                f"  if (!(p{i} + {elements * size} <= p{j} || p{j} + {other_elements * other_size} <= p{i})) __builtin_trap();"
            ]
    gx, gy, gz = case["grid"]
    if patch:
        lines += [
            "  /* The raw Triton body is im2col; retain the complete upstream W8A8 projection. */",
            "  d3.allocated = patch_im2col;",
            "  d3.aligned = patch_im2col;",
        ]
    lines += [
        f"  for (int32_t x = 0; x < {gx}; ++x)",
        f"    for (int32_t y = 0; y < {gy}; ++y)",
        f"      for (int32_t z = 0; z < {gz}; ++z)",
        f"        _mlir_ciface_{case['symbol']}("
        + ", ".join(f"&u{i}" for i in range(len(arguments)))
        + f", {gx}, {gy}, {gz}, x, y, z);",
    ]
    if patch:
        lines += [
            "  MemRef2 im = make_2(patch_im2col, 196, 768);",
            "  MemRef2 iq = make_2(patch_im2col_q, 196, 768);",
            "  MemRef1 rs = make_1(patch_im2col_scale, 196);",
            "  _mlir_ciface_kernel_quantize_196x768(&im, &iq, &rs);",
            "  MemRef2 w = make_2((void *)p1, 768, 768);",
            "  MemRef2 wq = make_2(patch_weight_q, 768, 768);",
            "  MemRef1 cs = make_1(patch_weight_scale, 768);",
            "  _mlir_ciface_kernel_quantize_768x768(&w, &wq, &cs);",
            "  MemRef2 c = make_2(patch_acc, 196, 768);",
            "  _mlir_ciface_kernel_matmul_i8_overwrite_196x768x768(&iq, &wq, &c);",
            "  MemRef2 y = make_2(patch_gemm, 196, 768);",
            "  _mlir_ciface_kernel_dequantize_196x768(&c, &rs, &cs, &y);",
            "  for (int p = 0; p < 196; ++p)",
            "    for (int o = 0; o < 768; ++o)",
            "      ((float *)p3)[o * 196 + p] = patch_gemm[p * 768 + o] + ((const float *)p2)[o];",
        ]
    lines += ["}", ""]
    return "\n".join(lines)


def export_case(case, output):
    import triton

    sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
    from contract_frontend import export_kernel

    module_name = "vit_operator_" + case["name"]
    spec = importlib.util.spec_from_file_location(
        module_name, case["kernel_source"]
    )
    kernel_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = kernel_module
    spec.loader.exec_module(kernel_module)
    fn = getattr(kernel_module, case["kernel"])
    # These are the same explicit budgets, domain and special-value promises
    # carried by the original example exporter.
    accuracy = {
        name: {"domain": "all-f32", "special_values": "ieee", **budget}
        for name, budget in case.get("math_accuracy", {}).items()
    } or None
    artifact = export_kernel(
        fn,
        case["signature"],
        case["constexprs"],
        case["grid"],
        case["symbol"],
        output,
        math_accuracy=accuracy,
    )
    (output / "adapter.c").write_text(adapter_source(case))
    manifest = dict(case)
    manifest["kernel_source"] = (
        Path(case["kernel_source"]).relative_to(REPO).as_posix()
    )
    manifest.update(
        frontend="triton-riscv export_contract -> verified standard MLIR",
        contract_manifest_sha256=digest(output / "contract-manifest.json"),
        contract_sha256=artifact.manifest["outputs"]["contract_sha256"],
        frontend_tool_sha256=artifact.manifest["tool"]["sha256"],
        abi="typed unranked rank-one memrefs; six i32 grid/pid values",
        triton_version=triton.__version__,
        kernel_module_sha256=digest(case["kernel_source"]),
        ttir_sha256=digest(output / "kernel.ttir"),
        adapter_sha256=digest(output / "adapter.c"),
        exporter_sha256=digest(__file__),
        cases_sha256=digest(ROOT / "cases.py"),
        dependencies=[
            "quantize_196x768",
            "quantize_768x768",
            "matmul_i8_overwrite_196x768x768",
            "dequantize_196x768",
        ]
        if case["family"] == "patch_embed"
        else [],
    )
    (output / "frontend.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summary_report_path(output_root, case_names, kind):
    """Keep independent case reports separate from complete-suite evidence."""
    if case_names is None:
        return output_root.parent / f"{kind}-report.json"
    names = list(dict.fromkeys(case_names))
    if len(names) == 1:
        return output_root / names[0] / f"{kind}-report.json"
    return output_root / f"{kind}-subset-report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="this case's output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export_case(describe(args.case), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

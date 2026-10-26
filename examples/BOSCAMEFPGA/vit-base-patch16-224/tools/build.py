#!/usr/bin/env python3
"""Export independent ViT operators and run the neutral contract pipeline.

Every numerical operation comes from its case's real Triton kernel. A failed
case is recorded and the remaining cases are still attempted. No legacy
shared-Linalg export, global store policy, or handwritten replacement IR is
used as a fallback.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
from boscame_paths import buddy_tools  # noqa: E402

sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402

STAGES = ("ttir", "boscame", "llvm", "asm")


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


def command(argv, output, input_text=None, stdout_path=None):
    process = subprocess.run(
        list(map(str, argv)),
        input=input_text,
        text=True,
        capture_output=True,
    )
    with (output / "build.log").open("a") as log:
        log.write("$ " + " ".join(map(str, argv)) + "\n" + process.stderr)
        if stdout_path is None:
            log.write(process.stdout)
    if process.returncode:
        raise RuntimeError(
            (process.stderr + process.stdout).strip()
            or f"command exited {process.returncode}"
        )
    if stdout_path is not None:
        stdout_path.write_text(process.stdout)


def build_case(case, output_root, stop="boscame"):
    output = output_root / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "build.log").write_text("")
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "stage": "ttir",
        "artifact_dir": str(output),
    }
    try:
        export_case(case, output)
        if stop != "ttir":
            opt = tool("buddy-opt")
            stages = [
                ("bufferized", "frontend.mlir", ["--kernel-contract-verify"]),
                (
                    "planned",
                    "bufferized.mlir",
                    ["--bosc-ame-plan", "--bosc-ame-plan-verify"],
                ),
                ("boscame", "planned.mlir", ["--bosc-ame-emit"]),
            ]
            if stop in ("llvm", "asm"):
                stages.append(("llvm", "boscame.mlir", ["--lower-bosc-ame"]))
            for stage, source, flags in stages:
                record["stage"] = stage
                command(
                    [
                        opt,
                        output / f"kernel.{source}",
                        *flags,
                        "-o",
                        output / f"kernel.{stage}.mlir",
                    ],
                    output,
                )
            if stop in ("llvm", "asm"):
                record["stage"] = "llvm-ir"
                command(
                    [
                        tool("buddy-translate"),
                        "--buddy-to-llvmir",
                        output / "kernel.llvm.mlir",
                        "-o",
                        output / "kernel.ll",
                    ],
                    output,
                )
            if stop == "asm":
                record["stage"] = "asm"
                command(
                    [
                        tool("llc"),
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
                )
                common_tools = buddy_tools()
                command(
                    [
                        sys.executable,
                        common_tools / "ame_to_word.py",
                        "--gpr-mode=fixed",
                        "--preserve-matrix-registers",
                    ],
                    output,
                    input_text=(output / "kernel.s").read_text(),
                    stdout_path=output / "kernel.encoded.s",
                )
                command(
                    [
                        sys.executable,
                        common_tools / "restrict_fpga_assembly.py",
                        # The FPGA whitelist omits vfsub.vf; preserve its f32
                        # semantics with the existing bit-exact expansion.
                        "--expand-f32-sub",
                        "--expand-vector-merges",
                        "--order-vector-copies",
                    ],
                    output,
                    input_text=(output / "kernel.encoded.s").read_text(),
                    stdout_path=output / "kernel.nr.s",
                )
                command(
                    [
                        tool("clang"),
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
                )
        record.update(status="PASS", stage=stop)
        record["artifacts"] = {
            p.name: digest(p) for p in output.glob("kernel.*") if p.is_file()
        }
        record["kernel_module_sha256"] = digest(case["kernel_source"])
    except Exception as error:
        record["reason"] = str(error)
        with (output / "build.log").open("a") as log:
            log.write("\nBLOCKED: " + str(error) + "\n")
    (output / "build-result.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(f"{record['status']}: {case['name']} ({record['stage']})", flush=True)
    return record


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
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="boscame")
    parser.add_argument("--output", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.all and args.cases:
        parser.error("use --all or --case, not both")
    cases = (
        [describe(name) for name in dict.fromkeys(args.cases)]
        if args.cases
        else inventory()
    )
    if args.list:
        print(json.dumps(cases, indent=2))
        return 0
    if args.report is None:
        args.report = summary_report_path(args.output, args.cases, "build")
    results = [build_case(case, args.output, args.stage) for case in cases]
    report = {
        "schema": 1,
        "status": "PASS"
        if all(r["status"] == "PASS" for r in results)
        else "PARTIAL",
        "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

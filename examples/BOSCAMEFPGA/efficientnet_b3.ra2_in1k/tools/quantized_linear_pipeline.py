"""Public F32 classifier ABI backed solely by quantized integer MMA."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE = ROOT / "quantized_linear_kernels.py"
MATMUL_SOURCE = ROOT / "matmul_1x1000x1536/kernel.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_quantized_linear(case):
    return case.get("family") == "linear_f32" and (
        case.get("kernel"),
        case.get("implementation"),
        case.get("kernel_module"),
    ) == ("linear_w8a8", "w8a8_gemm", "quantized_linear_kernels")


def pipeline_layout(case):
    if not is_quantized_linear(case):
        raise ValueError("classifier must bind directly to W8A8 stages")
    m, n, k = (case["constexprs"][x] for x in ("M", "N", "K"))
    result = {"M": m, "N": n, "K": k}
    cursor = 0
    for name, elements, dtype, itemsize in (
        ("qa", m * k, "i8", 1),
        ("qw", n * k, "i8", 1),
        ("rs", m, "f32", 4),
        ("ws", n, "f32", 4),
        ("acc", m * n, "i32", 4),
    ):
        size = elements * itemsize
        result[name] = {
            "offset": cursor,
            "elements": elements,
            "bytes": size,
            "dtype": dtype,
        }
        cursor = (cursor + size + 63) & ~63
    result["total_bytes"] = cursor
    return result


def stage_cases(case):
    g = pipeline_layout(case)
    m, n, k = (g[x] for x in ("M", "N", "K"))
    b = 1 << (k - 1).bit_length()
    specs = [
        (
            "activation_quantize",
            "quantize_rows",
            [("A", 2, "f32"), ("QA", 2, "i8"), ("RowScale", 1, "f32")],
            [[m, k], [m, k], [m]],
            {"WIDTH": k, "BLOCK": b},
            [m, 1, 1],
            SOURCE,
        ),
        (
            "weight_quantize",
            "quantize_rows",
            [("W", 2, "f32"), ("QW", 2, "i8"), ("WeightScale", 1, "f32")],
            [[n, k], [n, k], [n]],
            {"WIDTH": k, "BLOCK": b},
            [n, 1, 1],
            SOURCE,
        ),
        (
            "zero_acc",
            "zero_acc",
            [("Acc", 2, "i32")],
            [[m, n]],
            {"COUNT": m * n, "BLOCK": 128},
            [(m * n + 127) // 128, 1, 1],
            SOURCE,
        ),
        (
            "gemm_i8",
            "matmul",
            [("QA", 2, "i8"), ("QW", 2, "i8"), ("Acc", 2, "i32")],
            [[m, k], [n, k], [m, n]],
            {"M": m, "N": n, "K": k, "BM": 1, "BN": 16, "BK": 64, "INTEGER": True},
            [m, (n + 15) // 16, 1],
            MATMUL_SOURCE,
        ),
        (
            "dequantize_bias",
            "dequantize_bias",
            [
                ("Acc", 2, "i32"),
                ("RowScale", 1, "f32"),
                ("WeightScale", 1, "f32"),
                ("Bias", 1, "f32"),
                ("Out", 2, "f32"),
            ],
            [[m, n], [m], [n], [n], [m, n]],
            {"M": m, "N": n, "BLOCK": 128},
            [m, (n + 127) // 128, 1],
            SOURCE,
        ),
    ]
    types = {"f32": "*fp32", "i8": "*i8", "i32": "*i32"}
    return [
        {
            "name": case["name"] + "__" + name,
            "stage": name,
            "kernel": kernel,
            "symbol": "triton_" + case["name"] + "__" + name,
            "kernel_source": str(source),
            "arguments": [{"name": a, "rank": r, "dtype": t} for a, r, t in args],
            "argument_shapes": shapes,
            "signature": {a: types[t] for a, _, t in args},
            "constexprs": constants,
            "grid": grid,
        }
        for name, kernel, args, shapes, constants, grid, source in specs
    ]


def adapter_source(case):
    from export import argument_shapes, descriptor_checks

    g = pipeline_layout(case)
    params = ", ".join(
        f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"])
    )
    lines = [
        "/* W8A8 classifier: quantize, zero, existing MMA, dequantize+bias. */",
        '#include "support.h"',
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
    ]
    stages = stage_cases(case)
    for s in stages:
        lines.append(
            f"extern void _mlir_ciface_{s['symbol']}("
            + ", ".join(["Unranked *"] * len(s["arguments"]) + ["int32_t"] * 6)
            + ");"
        )
    lines += [
        f"void _mlir_ciface_kernel_{case['name']}({params}) {{",
        *descriptor_checks(case, argument_shapes(case)),
        f"  const size_t scratch_bytes = {g['total_bytes']}u;",
        "  if (scratch_bytes > VIT_WORKSPACE_BYTES) __builtin_trap();",
        "  uintptr_t scratch = (uintptr_t)workspace(VIT_WORKSPACE_BYTES-scratch_bytes);",
        "  if (!scratch || (scratch & 63u) || scratch > UINTPTR_MAX-scratch_bytes) __builtin_trap();",
    ]
    for i, shape in enumerate(argument_shapes(case)):
        elements = 1
        for v in shape:
            elements *= v
        lines.append(
            f"  if (!(scratch+scratch_bytes <= p{i} || p{i}+{elements * 4}u <= scratch)) __builtin_trap();"
        )
    for key in ("qa", "qw", "rs", "ws", "acc"):
        v = g[key]
        lines += [
            f"  Descriptor1D d{key} = {{(void *)(scratch+{v['offset']}u), (void *)(scratch+{v['offset']}u), 0, {v['elements']}, 1}};",
            f"  Unranked u{key} = {{1, &d{key}}};",
        ]
    pointers = {
        "A": "&u0",
        "W": "&u1",
        "Bias": "&u2",
        "Out": "&u3",
        "QA": "&uqa",
        "QW": "&uqw",
        "RowScale": "&urs",
        "WeightScale": "&uws",
        "Acc": "&uacc",
    }
    for s in stages:
        gx, gy, gz = s["grid"]
        values = (
            [pointers[a["name"]] for a in s["arguments"]]
            + list(map(str, s["grid"]))
            + ["x", "y", "z"]
        )
        lines += [
            f"  for (int32_t x=0; x<{gx}; ++x)",
            f"    for (int32_t y=0; y<{gy}; ++y)",
            f"      for (int32_t z=0; z<{gz}; ++z)",
            f"        _mlir_ciface_{s['symbol']}(" + ", ".join(values) + ");",
        ]
    return "\n".join(lines + ["}", ""])


def export_pipeline(case, output):
    import triton
    from quantized_conv_pipeline import _merge_contracts

    sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
    from contract_frontend import export_kernel

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    opt = os.environ.get("BUDDY_OPT", str(REPO / "buddy-mlir/build/bin/buddy-opt"))
    stages = []
    contracts = []
    bundle = []
    tool_hashes = set()
    for s in stage_cases(case):
        directory = output / "stages" / s["stage"]
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(s["kernel_source"])
        spec = importlib.util.spec_from_file_location(
            "efficientnet_linear_" + s["stage"], source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        fn = getattr(module, s["kernel"])
        signature = dict(zip(fn.arg_names, s["signature"].values()))
        artifact = export_kernel(
            fn, signature, s["constexprs"], s["grid"], s["symbol"], directory
        )
        contracts.append((directory / "kernel.frontend.mlir", s["symbol"]))
        bundle += ["// stage: " + s["stage"], (directory / "kernel.ttir").read_text()]
        tool_hashes.add(artifact.manifest["tool"]["sha256"])
        stages.append(
            {
                **s,
                "kernel_source_sha256": digest(source),
                "artifacts": {
                    str((directory / name).relative_to(output)): digest(
                        directory / name
                    )
                    for name in (
                        "kernel.ttir",
                        "kernel.frontend.mlir",
                        "contract-manifest.json",
                    )
                },
            }
        )
    if len(tool_hashes) != 1:
        raise ValueError("mixed frontend tools")
    merged = output / "kernel.frontend.mlir.tmp"
    merged.write_text(_merge_contracts(contracts, opt))
    subprocess.run(
        [
            opt,
            str(merged),
            "--kernel-contract-verify",
            "-o",
            str(output / "kernel.frontend.mlir"),
        ],
        check=True,
    )
    merged.unlink()
    (output / "kernel.ttir").write_text("\n".join(bundle))
    (output / "adapter.c").write_text(adapter_source(case))
    composite = {
        "schema": 1,
        "kind": "composite-w8a8-linear",
        "entry": "_mlir_ciface_kernel_" + case["name"],
        "raw_entries": [s["symbol"] for s in stages],
        "stages": stages,
        "outputs": {
            "contract_sha256": digest(output / "kernel.frontend.mlir"),
            "ttir_sha256": digest(output / "kernel.ttir"),
        },
    }
    (output / "contract-manifest.json").write_text(
        json.dumps(composite, indent=2) + "\n"
    )
    manifest = dict(case)
    manifest.update(
        kernel_source=Path(case["kernel_source"]).relative_to(REPO).as_posix(),
        frontend="independent Triton stages -> merged verified MLIR",
        frontend_tool_sha256=next(iter(tool_hashes)),
        kernel_module_sha256=digest(case["kernel_source"]),
        triton_version=triton.__version__,
        contract_manifest_sha256=digest(output / "contract-manifest.json"),
        contract_sha256=digest(output / "kernel.frontend.mlir"),
        ttir_sha256=digest(output / "kernel.ttir"),
        adapter_sha256=digest(output / "adapter.c"),
        pipeline_layout=pipeline_layout(case),
        stages=stages,
        source_sha256={
            str(p.relative_to(REPO)): digest(p)
            for p in (
                SOURCE,
                MATMUL_SOURCE,
                Path(__file__),
                Path(case["kernel_source"]),
                ROOT / "cases.py",
                ROOT / "tools/export.py",
            )
        },
    )
    (output / "frontend.json").write_text(json.dumps(manifest, indent=2) + "\n")

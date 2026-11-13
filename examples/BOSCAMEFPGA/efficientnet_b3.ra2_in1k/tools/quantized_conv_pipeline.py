"""Describe and export the bounded im2col/W8A8 convolution pipeline.

Every public convolution keeps its original tensor ABI.  Its generated C
adapter calls four independently exported Triton entries from one merged
Contract module, so the ordinary Make pipeline still produces one kernel.o.
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE = ROOT / "quantized_conv_kernels.py"
TILE = 64
KPAD = 64
POINTER_TYPES = {"f32": "*fp32", "i8": "*i8", "i32": "*i32"}
ITEM_BYTES = {"f32": 4, "i8": 1, "i32": 4}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry(case):
    spec = case["constexprs"]
    depthwise = case_is_depthwise(case)
    if case["family"] not in ("conv2d_f32", "dwconv2d_f32"):
        raise ValueError("im2col_w8a8 requires a floating convolution public ABI")
    channels = spec["C"] if depthwise else spec["OC"]
    width = (1 if depthwise else spec["C"]) * spec["KH"] * spec["KW"]
    if width > KPAD:
        raise ValueError("this bounded convolution pipeline requires K <= 64")
    pixels = spec["OH"] * spec["OW"]
    return {
        "depthwise": depthwise,
        "channels": channels,
        "groups": spec["C"] if depthwise else 1,
        "matrix_rows": 1 if depthwise else channels,
        "bm": 1 if depthwise else 16,
        "width": width,
        "pixels": pixels,
        "blocks": (pixels + TILE - 1) // TILE,
    }


def case_is_depthwise(case):
    """Require the explicit per-case binding to the shared W8A8 stages."""
    bindings = {
        "conv2d_f32": (
            "conv2d_im2col_w8a8",
            "im2col_w8a8",
            "quantized_conv_kernels",
            False,
        ),
        "dwconv2d_f32": (
            "dwconv2d_im2col_w8a8",
            "im2col_w8a8_depthwise",
            "quantized_conv_kernels",
            True,
        ),
    }
    expected = bindings.get(case["family"])
    if (
        expected is None
        or (
            case.get("kernel"),
            case.get("implementation"),
            case.get("kernel_module"),
        )
        != expected[:3]
    ):
        raise ValueError(case["name"] + ": missing direct W8A8 pipeline binding")
    return expected[3]


def pipeline_layout(case):
    """Aligned scratch offsets, all relative to the arena's reserved tail."""
    geometry = _geometry(case)
    result = {**geometry, "tile": TILE, "kpad": KPAD}
    cursor = 0
    for name, elements, dtype in (
        ("qweight", geometry["channels"] * KPAD, "i8"),
        ("weight_scale", geometry["channels"], "f32"),
        ("qpatch", TILE * KPAD, "i8"),
        ("patch_scale", TILE, "f32"),
        ("acc", geometry["bm"] * TILE, "i32"),
    ):
        size = elements * ITEM_BYTES[dtype]
        result[name] = {
            "offset": cursor,
            "bytes": size,
            "elements": elements,
            "dtype": dtype,
        }
        cursor = (cursor + size + 63) & ~63
    result["total_bytes"] = cursor
    return result


def stage_cases(case):
    """Return the exact raw signatures and global grids used by the adapter."""
    module_spec = importlib.util.spec_from_file_location(
        "efficientnet_quantized_conv_kernels", SOURCE
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel_case = dict(case)
    case_is_depthwise(kernel_case)
    result = []
    pipeline = getattr(module, kernel_case["kernel"])
    for config in pipeline(kernel_case):
        stage_name = config["name"]
        arguments = config["arguments"]
        stage = {
            "name": case["name"] + "__" + stage_name,
            "stage": stage_name,
            "family": "quantized_conv_" + stage_name,
            "kernel": config["kernel"].__name__,
            "kernel_module": "quantized_conv_kernels",
            "kernel_source": str(SOURCE),
            "symbol": "triton_" + case["name"] + "__" + stage_name,
            "arguments": arguments,
            "argument_shapes": config["shapes"],
            "signature": {
                argument["name"]: POINTER_TYPES[argument["dtype"]]
                for argument in arguments
            },
            "constexprs": config["constexprs"],
            "grid": config["grid"],
            "shape": config["shapes"][0],
            "roles": [],
        }
        result.append(stage)
    return result


def scratch_initialization(case):
    """Internal public-adapter contract required by the logical-K raw stage."""
    field = pipeline_layout(case)["qpatch"]
    return {
        "buffer": "QPatch",
        "offset": field["offset"],
        "bytes": field["bytes"],
        "value": 0,
        "frequency": "once per public adapter invocation",
        "before": "all raw stage calls",
        "visibility": "bosc_ame_ddr_write_begin then clear then fence rw,rw",
        "raw_stage": "im2col_quantize writes logical K only; padding remains zero across tiles/groups",
    }


def adapter_source(case, validate_public_lines=None):
    """Compose real raw entries using a bounded scratch tile in the arena."""
    if validate_public_lines is None:
        from export import argument_shapes, descriptor_checks

        validate_public_lines = descriptor_checks(case, argument_shapes(case))
    if len(case["arguments"]) != 4:
        raise ValueError("quantized convolution requires X, Weight, Bias, Out")
    layout = pipeline_layout(case)
    stages = {stage["stage"]: stage for stage in stage_cases(case)}
    params = ", ".join(
        f"MemRef{argument['rank']} *a{index}"
        for index, argument in enumerate(case["arguments"])
    )
    lines = [
        "/* Generated bounded im2col -> W8A8 MMA -> dequantize/bias adapter. */",
        '#include "support.h"',
        "extern void bosc_ame_ddr_write_begin(void);",
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
    ]
    for stage in stages.values():
        lines.append(
            f"extern void _mlir_ciface_{stage['symbol']}("
            + ", ".join(["Unranked *"] * len(stage["arguments"]) + ["int32_t"] * 6)
            + ");"
        )
    lines += [
        f"void _mlir_ciface_kernel_{case['name']}({params}) {{",
        *validate_public_lines,
        f"  const size_t scratch_bytes = {layout['total_bytes']}u;",
        "  if ((size_t)VIT_WORKSPACE_BYTES < scratch_bytes) __builtin_trap();",
        "  uintptr_t scratch = (uintptr_t)workspace((size_t)VIT_WORKSPACE_BYTES - scratch_bytes);",
        "  if (!scratch || (scratch & 63u) || scratch > UINTPTR_MAX - scratch_bytes) __builtin_trap();",
    ]
    spec = case["constexprs"]
    public_lengths = [
        spec["C"] * spec["H"] * spec["W"] * 4,
        layout["channels"] * layout["width"] * 4,
        layout["channels"] * 4,
        layout["channels"] * layout["pixels"] * 4,
    ]
    for index, length in enumerate(public_lengths):
        lines.append(
            f"  if (!(scratch + scratch_bytes <= p{index} || p{index} + {length}u <= scratch)) __builtin_trap();"
        )
    short_names = {
        "qweight": "qw",
        "weight_scale": "ws",
        "qpatch": "qp",
        "patch_scale": "ps",
        "acc": "acc",
    }
    for name, short in short_names.items():
        buffer = layout[name]
        lines += [
            f"  Descriptor1D d{short} = {{(void *)(scratch + {buffer['offset']}u), (void *)(scratch + {buffer['offset']}u), 0, {buffer['elements']}, 1}};",
            f"  Unranked u{short} = {{1, &d{short}}};",
        ]

    def raw_call(name, pointers, coordinates, indent):
        stage = stages[name]
        values = [*pointers, *map(str, stage["grid"]), *coordinates]
        return (
            " " * indent + f"_mlir_ciface_{stage['symbol']}(" + ", ".join(values) + ");"
        )

    lines += [
        "  /* Initialize immutable QPatch padding once for this public invocation. */",
        "  bosc_ame_ddr_write_begin();",
        f"  for (size_t byte = 0; byte < {layout['qpatch']['bytes']}u; ++byte)",
        f"    ((unsigned char *)(scratch + {layout['qpatch']['offset']}u))[byte] = 0;",
        "#ifndef HOST_TEST",
        '  __asm__ volatile("fence rw,rw" ::: "memory");',
        "#else",
        '  __asm__ volatile("" ::: "memory");',
        "#endif",
        f"  for (int32_t row = 0; row < {layout['channels']}; ++row)",
        raw_call("weight_quantize", ["&u1", "&uqw", "&uws"], ["row", "0", "0"], 4),
        f"  for (int32_t group = 0; group < {layout['groups']}; ++group) {{",
        "    Descriptor1D dgw = dqw;",
    ]
    if layout["depthwise"]:
        lines += [
            f"    dgw.allocated = dgw.aligned = (void *)(scratch + {layout['qweight']['offset']}u + (uintptr_t)group * {KPAD}u);",
            f"    dgw.size = {KPAD};",
        ]
    lines += [
        "    Unranked ugw = {1, &dgw};",
        f"    for (int32_t block = 0; block < {layout['blocks']}; ++block) {{",
        raw_call(
            "im2col_quantize", ["&u0", "&uqp", "&ups"], ["block", "0", "group"], 6
        ),
    ]
    bm = stages["gemm_i8"]["constexprs"]["BM"]
    m_tiles = stages["gemm_i8"]["grid"][0]
    if layout["depthwise"]:
        lines += [
            raw_call("gemm_i8", ["&ugw", "&uqp", "&uacc"], ["0", "0", "0"], 6),
            raw_call(
                "dequantize_bias",
                ["&uacc", "&ups", "&uws", "&u2", "&u3"],
                ["group", "block", "0"],
                6,
            ),
        ]
    else:
        lines += [
            f"      for (int32_t m_tile = 0; m_tile < {m_tiles}; ++m_tile) {{",
            raw_call("gemm_i8", ["&ugw", "&uqp", "&uacc"], ["m_tile", "0", "0"], 8),
            f"        for (int32_t oc = m_tile * {bm}; oc < (m_tile + 1) * {bm} && oc < {layout['channels']}; ++oc)",
            raw_call(
                "dequantize_bias",
                ["&uacc", "&ups", "&uws", "&u2", "&u3"],
                ["oc", "block", "m_tile"],
                10,
            ),
            "      }",
        ]
    lines += ["    }", "  }", "}", ""]
    return "\n".join(lines)


def _merge_contracts(paths, opt):
    """Print local scopes before merging so independently named aliases cannot collide."""
    header, bodies = None, []
    symbols = set()
    for path, expected_symbol in paths:
        normalized = subprocess.check_output(
            [str(opt), str(path), "--mlir-print-local-scope"], text=True
        )
        lines = normalized.strip().splitlines()
        if (
            not lines
            or not lines[0].startswith("module attributes ")
            or lines[-1] != "}"
        ):
            raise ValueError(
                "expected one directly printed Contract module: " + str(path)
            )
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise ValueError("stage Contract module attributes differ")
        if any(re.match(r"^[#!]\S+\s*=", line) for line in lines):
            raise ValueError(
                "local-scope Contract unexpectedly retains top-level aliases"
            )
        functions = re.findall(r"^  func\.func\s+@([^ (]+)\(", normalized, re.MULTILINE)
        if functions != [expected_symbol] or expected_symbol in symbols:
            raise ValueError("stage Contract must define its one unique raw entry")
        symbols.add(expected_symbol)
        bodies.append("\n".join(lines[1:-1]))
    return header + "\n" + "\n".join(bodies) + "\n}\n"


def export_pipeline(case, output):
    """Export all raw stages, merge their Contracts, and write composite evidence."""
    import triton

    sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
    from contract_frontend import export_kernel
    from export import adapter_source

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    module_name = "efficientnet_quantized_conv_kernels"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    kernel_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = kernel_module
    spec.loader.exec_module(kernel_module)
    opt = os.environ.get("BUDDY_OPT", str(REPO / "buddy-mlir/build/bin/buddy-opt"))
    stages, contracts, bundle = (
        [],
        [],
        [
            "// Composite convolution TTIR audit bundle.",
            "// Each section is an independent exported module; this bundle is not a backend input.",
        ],
    )
    tool_hashes = set()
    for stage in stage_cases(case):
        directory = output / "stages" / stage["stage"]
        directory.mkdir(parents=True, exist_ok=True)
        artifact = export_kernel(
            getattr(kernel_module, stage["kernel"]),
            stage["signature"],
            stage["constexprs"],
            stage["grid"],
            stage["symbol"],
            directory,
        )
        contracts.append((directory / "kernel.frontend.mlir", stage["symbol"]))
        bundle += [
            f"// ----- stage: {stage['stage']} symbol: {stage['symbol']}",
            (directory / "kernel.ttir").read_text().rstrip(),
        ]
        tool_hash = artifact.manifest["tool"]["sha256"]
        tool_hashes.add(tool_hash)
        record = {
            **stage,
            "kernel_source": SOURCE.relative_to(REPO).as_posix(),
            "kernel_source_sha256": digest(SOURCE),
            "frontend_tool_sha256": tool_hash,
            "artifacts": {},
        }
        for name in ("kernel.ttir", "kernel.frontend.mlir", "contract-manifest.json"):
            relative = (directory / name).relative_to(output).as_posix()
            record["artifacts"][relative] = digest(directory / name)
        stages.append(record)
    if len(tool_hashes) != 1:
        raise ValueError("composite stages were exported by different frontend tools")
    temporary = output / "kernel.frontend.mlir.tmp"
    temporary.write_text(_merge_contracts(contracts, opt))
    try:
        subprocess.run(
            [
                str(opt),
                str(temporary),
                "--kernel-contract-verify",
                "-o",
                str(output / "kernel.frontend.mlir"),
            ],
            check=True,
        )
    finally:
        temporary.unlink(missing_ok=True)
    (output / "kernel.ttir").write_text("\n".join(bundle) + "\n")
    (output / "adapter.c").write_text(adapter_source(case))
    aggregate = {
        "schema": 1,
        "kind": "composite-im2col-w8a8-convolution",
        "entry": "_mlir_ciface_kernel_" + case["name"],
        "raw_entries": [stage["symbol"] for stage in stages],
        "abi": "args_aligned64_nonalias",
        "ttir_format": "split-input-file audit bundle, not a backend input",
        "scratch_initialization": scratch_initialization(case),
        "outputs": {
            "contract_sha256": digest(output / "kernel.frontend.mlir"),
            "ttir_sha256": digest(output / "kernel.ttir"),
        },
        "stages": stages,
    }
    (output / "contract-manifest.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    manifest = dict(case)
    manifest["kernel_source"] = Path(case["kernel_source"]).relative_to(REPO).as_posix()
    manifest.update(
        frontend="independent Triton stage exports -> merged verified standard MLIR",
        contract_manifest_sha256=digest(output / "contract-manifest.json"),
        contract_sha256=digest(output / "kernel.frontend.mlir"),
        frontend_tool_sha256=next(iter(tool_hashes)),
        abi="public typed convolution ABI; private typed unranked stage entries",
        triton_version=triton.__version__,
        kernel_module_sha256=digest(case["kernel_source"]),
        ttir_sha256=digest(output / "kernel.ttir"),
        ttir_format=aggregate["ttir_format"],
        adapter_sha256=digest(output / "adapter.c"),
        exporter_sha256=digest(ROOT / "tools/export.py"),
        pipeline_exporter_sha256=digest(__file__),
        cases_sha256=digest(ROOT / "cases.py"),
        dependencies=[],
        stages=stages,
        pipeline_layout=pipeline_layout(case),
        scratch_initialization=scratch_initialization(case),
        source_sha256={
            Path(path).relative_to(REPO).as_posix(): digest(path)
            for path in (
                case["kernel_source"],
                SOURCE,
                Path(__file__),
                ROOT / "tools/export.py",
                ROOT / "cases.py",
            )
        },
    )
    (output / "frontend.json").write_text(json.dumps(manifest, indent=2) + "\n")

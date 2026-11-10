#!/usr/bin/env python3
"""Build the complete ViT-B/16 NH/E6 flat image from the current graph artifacts.

Pipeline: lower the replaced subgraph to LLVM IR with a C-interface @vit entry,
pack the 273 checkpoint/derived parameters, generate a 734-argument launcher,
compile the graph, the 1543 generated adapters, the 28 per-case kernel adapters
and the prebuilt BOSCAME kernel objects, link the NH/E6 runtime with the
board_model.ld memory map and emit the flat .bin.

This tool deliberately ignores board load/time/memory limits (they are tracked
separately): it produces the artifact, not a hardware claim.

Requirements: run with the boscame Python environment (torch/transformers/
safetensors/buddy_mlir) and the in-tree Buddy build; see the module constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

V = Path(__file__).resolve().parents[1]
REPO = V.parents[2]
BUDDY = REPO / "buddy-mlir"
BUDDY_PYTHON = BUDDY / "build" / "python_packages"
BUDDY_BIN = BUDDY / "build" / "bin"
LLVM_BIN = BUDDY / "llvm" / "build" / "bin"
BUDDY_TOOLS = BUDDY / "examples" / "FPGA-BOSCAME" / "tools"
RUNTIME = BUDDY / "runtime" / "boscame"

KERNELS_DEFAULT = V / "build" / "kernels"
REPLACEMENT_DEFAULT = V / "build" / "replacement-w8a8"
OUTPUT_DEFAULT = V / "build" / "model-bin"
MODEL_INPUT_DEFAULT = V / "build" / "model-input"
ASSETS_DEFAULT = V / "assets"

CFLAGS = [
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
    "-fno-vectorize",
    "-fno-slp-vectorize",
    "-ffp-contract=off",
    "-ffunction-sections",
    "-fdata-sections",
    "-DBOSC_AME_DMA_ASSERT(x)=",
]
LLC_FLAGS = [
    "-O2",
    "-filetype=obj",
    "-mtriple=riscv64",
    "-target-abi=lp64d",
    "-mattr=+m,+a,+f,+d,+c",
    "-code-model=medium",
]
DTYPE_ITEMSIZE = {"f32": 4, "i8": 1, "i32": 4, "i64": 8}


def log(message):
    print(message, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log_path, cwd=None):
    command = [str(value) for value in command]
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text("$ " + " ".join(command) + "\n" + result.stdout)
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-20:])
        raise RuntimeError(
            "command failed (%d): %s\n%s" % (result.returncode, " ".join(command), tail)
        )
    return result.stdout


def align64(value):
    return (value + 63) & ~63


def mlir_dtype_name(element_type):
    text = str(element_type)
    if text in ("f32", "i8", "i32", "i64"):
        return text
    raise ValueError("unsupported element type %r" % (element_type,))


def dtype_from_layout(value):
    name = str(value).rsplit(".", 1)[-1]
    mapping = {"Float32": "f32", "Int8": "i8", "Int32": "i32", "Int64": "i64"}
    if name not in mapping:
        raise ValueError("unsupported layout dtype %r" % (value,))
    return mapping[name]


# ---------------------------------------------------------------------------
# Lowering: @vit with a C interface (the wrapper's param packing is bypassed)
# ---------------------------------------------------------------------------


def add_ciface(source_mlir: Path, destination: Path):
    if str(BUDDY_PYTHON) not in sys.path:
        sys.path.insert(0, str(BUDDY_PYTHON))
    from buddy_mlir import ir

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(source_mlir.read_text(), context=context)
        function = None
        for operation in module.body.operations:
            if operation.operation.name != "func.func":
                continue
            if ir.StringAttr(operation.attributes["sym_name"]).value == "vit":
                function = operation
                break
        if function is None:
            raise ValueError("@vit is missing from the subgraph")
        function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        destination.write_text(str(module))


def stage_lower(ctx):
    output_ll = ctx.output / "vit.ciface.ll"
    reuse = ctx.inputs / "vit.ciface.ll"
    if ctx.reuse_lowered and reuse.is_file():
        shutil.copyfile(reuse, output_ll)
        log("lower: reused %s" % reuse)
        return output_ll
    source = ctx.replacement / "subgraph0.triton.mlir"
    ciface = ctx.output / "subgraph.ciface.mlir"
    add_ciface(source, ciface)
    log("lower: wrote %s" % ciface)
    tosa = ctx.output / "subgraph.tosa.mlir"
    bufferized = ctx.output / "subgraph.bufferized.mlir"
    lowered = ctx.output / "subgraph.lowered.mlir"
    tosa_pipeline = (
        "builtin.module("
        + ",".join(
            "func.func(%s)" % name
            for name in (
                "tosa-to-linalg-named",
                "tosa-to-linalg",
                "tosa-to-tensor",
                "tosa-to-arith",
            )
        )
        + ")"
    )
    run(
        [ctx.buddy_opt, ciface, "--pass-pipeline=" + tosa_pipeline, "-o", tosa],
        ctx.output / "lower.tosa.log",
    )
    run(
        [
            ctx.buddy_opt,
            tosa,
            "--canonicalize",
            "--cse",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs-from-loops=true buffer-alignment=64",
            "-o",
            bufferized,
        ],
        ctx.output / "lower.bufferize.log",
    )
    run(
        [
            ctx.buddy_opt,
            bufferized,
            "--convert-linalg-to-loops",
            "--expand-strided-metadata",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            lowered,
        ],
        ctx.output / "lower.scalar.log",
    )
    run(
        [ctx.buddy_translate, "--buddy-to-llvmir", lowered, "-o", output_ll],
        ctx.output / "lower.translate.log",
    )
    text = output_ll.read_text()
    if "define void @_mlir_ciface_vit(" not in text:
        raise RuntimeError("lowered IR does not export _mlir_ciface_vit")
    log("lower: wrote %s (%.1f MiB)" % (output_ll, output_ll.stat().st_size / 2**20))
    return output_ll


# ---------------------------------------------------------------------------
# Parameter values: 127 original checkpoint tensors + 73x2 derived W8A8 params
# ---------------------------------------------------------------------------


def quantize_rows(value):
    import numpy as np

    value = np.ascontiguousarray(value, dtype=np.float32)
    rows = value.reshape(value.shape[0], -1)
    maximum = np.max(np.abs(rows), axis=1)
    scale = np.where(maximum == 0.0, 1.0, maximum / np.float32(127.0)).astype(
        np.float32
    )
    ratio = rows / scale[:, None]
    saturated = np.clip(ratio, np.float32(-127), np.float32(127))
    base = saturated.astype(np.int32)
    fraction = saturated - base.astype(np.float32)
    adjustment = np.where(np.abs(fraction) >= 0.5, np.where(saturated >= 0, 1, -1), 0)
    return (base + adjustment).astype(np.int8).reshape(value.shape), scale


def load_parameter_values(ctx, report):
    import torch

    sys.path.insert(0, str(V / "tools"))
    import import_model
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.ops import tosa

    model, _, _ = import_model.load_model(
        ctx.assets / "config.json", ctx.assets / "model.safetensors", torch.float32
    )
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=import_model.capture_decompositions(),
        func_name="forward",
    )
    with torch.no_grad():
        graphs = compiler.importer(
            model, pixel_values=torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        )
    if len(graphs) != 1:
        raise RuntimeError("expected one captured graph for parameter recovery")
    imported = compiler.imported_params[graphs[0]]
    original = {}
    for index, tensor in enumerate(imported):
        original["arg%d_1" % (index + 1)] = tensor.detach().cpu().contiguous().numpy()
    values = dict(original)
    derived = 0
    for record in report["w8a8_linears"]:
        weight = original.get(record["weight_param"])
        if weight is None:
            raise ValueError("missing source weight %s" % record["weight_param"])
        quantized, scale = quantize_rows(weight)
        values[record["weight_param_name"]] = quantized
        values[record["scale_param_name"]] = scale
        derived += 2
    log(
        "pack: %d original tensors + %d derived W8A8 tensors" % (len(original), derived)
    )
    return values


def stage_pack(ctx):
    import numpy as np

    report = json.loads((ctx.replacement / "triton-call-replacement.json").read_text())
    layout = report["parameter_layout"]
    values = load_parameter_values(ctx, report)
    blob = bytearray()
    records = []
    missing = []
    for entry in layout:
        name = entry["name"]
        array = values.get(name)
        if array is None:
            missing.append(name)
            continue
        dtype = dtype_from_layout(entry["dtype"])
        shape = [int(x) for x in entry["shape"]]
        if list(array.shape) != shape:
            raise ValueError(
                "shape mismatch for %s: %s != %s" % (name, list(array.shape), shape)
            )
        if dtype == "i8":
            if array.dtype != np.int8:
                raise ValueError("%s is not int8" % name)
        elif dtype == "f32":
            if array.dtype != np.float32:
                raise ValueError("%s is not float32" % name)
        else:
            raise ValueError("unsupported dtype %s for %s" % (dtype, name))
        offset = align64(len(blob))
        blob.extend(b"\x00" * (offset - len(blob)))
        payload = array.tobytes()
        blob.extend(payload)
        records.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "elements": int(array.size),
                "nbytes": len(payload),
                "offset": offset,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if missing:
        raise ValueError("unresolved parameters: %s" % missing[:5])
    blob.extend(b"\x00" * (align64(len(blob)) - len(blob)))
    (ctx.output / "params.bin").write_bytes(bytes(blob))
    (ctx.output / "params-layout.json").write_text(
        json.dumps(
            {
                "bytes": len(blob),
                "sha256": sha256_file(ctx.output / "params.bin"),
                "entries": records,
            },
            indent=2,
        )
        + "\n"
    )
    log("pack: params.bin %.1f MiB, %d entries" % (len(blob) / 2**20, len(records)))
    return {record["name"]: record for record in records}


def read_vit_signature(subgraph: Path):
    if str(BUDDY_PYTHON) not in sys.path:
        sys.path.insert(0, str(BUDDY_PYTHON))
    from buddy_mlir import ir

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(subgraph.read_text(), context=context)
        function = None
        for operation in module.body.operations:
            if operation.operation.name == "func.func" and (
                ir.StringAttr(operation.attributes["sym_name"]).value == "vit"
            ):
                function = operation
                break
        if function is None:
            raise ValueError("@vit is missing")
        signature = []
        for argument in function.regions[0].blocks[0].arguments:
            typ = argument.type
            if isinstance(typ, ir.RankedTensorType):
                shape = [int(dim) for dim in typ.shape]
                dtype = mlir_dtype_name(typ.element_type)
            elif isinstance(typ, ir.MemRefType):
                shape = [int(dim) for dim in typ.shape]
                dtype = mlir_dtype_name(typ.element_type)
            else:
                raise ValueError("unsupported @vit argument type %r" % (typ,))
            signature.append({"shape": shape, "dtype": dtype})
        return signature


# ---------------------------------------------------------------------------
# Launcher generation
# ---------------------------------------------------------------------------


def stage_generate(ctx, params):
    report = json.loads((ctx.replacement / "triton-call-replacement.json").read_text())
    buffer_contract = json.loads(
        (ctx.replacement / "workspace-buffers.json").read_text()
    )
    roots = buffer_contract["argument_roots"]
    signature = read_vit_signature(ctx.replacement / "subgraph0.triton.mlir")
    if len(roots) != len(signature):
        raise ValueError(
            "argument_roots %d != @vit args %d" % (len(roots), len(signature))
        )
    param_layout = {entry["name"]: entry for entry in report["parameter_layout"]}
    input_layout = {entry["name"]: entry for entry in report["input_layout"]}

    args = []
    workspace_index = 0
    for index, (root, sig) in enumerate(zip(roots, signature)):
        dtype = sig["dtype"]
        shape = sig["shape"]
        itemsize = DTYPE_ITEMSIZE[dtype]
        elements = 1
        for dim in shape:
            elements *= dim
        nbytes = elements * itemsize
        if root in param_layout:
            record = params[root]
            if record["dtype"] != dtype or list(record["shape"]) != shape:
                raise ValueError("parameter layout mismatch for %s" % root)
            source = "(void *)(uintptr_t)(model_params_blob + %d)" % record["offset"]
            args.append(
                {
                    "index": index,
                    "name": root,
                    "kind": "param",
                    "dtype": dtype,
                    "shape": shape,
                    "nbytes": nbytes,
                    "source": source,
                }
            )
        elif root == "arg0_1":
            if shape != [1, 3, 224, 224]:
                raise ValueError("unexpected image shape %s" % shape)
            args.append(
                {
                    "index": index,
                    "name": root,
                    "kind": "input",
                    "dtype": dtype,
                    "shape": shape,
                    "nbytes": nbytes,
                    "source": "(void *)(uintptr_t)model_input_blob",
                }
            )
        else:
            if root not in input_layout:
                raise ValueError("unknown @vit argument %s" % root)
            args.append(
                {
                    "index": index,
                    "name": root,
                    "kind": "workspace",
                    "dtype": dtype,
                    "shape": shape,
                    "nbytes": nbytes,
                    "workspace": workspace_index,
                }
            )
            workspace_index += 1

    def strides_of(shape):
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return strides

    lines = []
    lines.append("/* Generated by tools/build_model_bin.py. Do not edit. */")
    lines.append("#include <stdint.h>")
    lines.append('#include "support.h"')
    lines.append('#include "model_runtime.h"')
    lines.append('#include "bosc_ame_dma.h"')
    lines.append('#include "nh_e6_platform.h"')
    lines.append("")
    lines.append("static float model_logits[1000] __attribute__((aligned(64)));")
    lines.append("")
    workspace_total = 0
    for arg in args:
        if arg["kind"] == "workspace":
            arg["workspace_offset"] = align64(workspace_total)
            workspace_total = arg["workspace_offset"] + arg["nbytes"]
    workspace_total = align64(workspace_total)
    lines.append("")
    prototype = ["MemRef1 *"]
    for arg in args:
        prototype.append("MemRef%d *" % len(arg["shape"]))
    lines.append("extern void _mlir_ciface_vit(" + ", ".join(prototype) + ");")
    lines.append("void vit_install_trap_handler(void);")
    lines.append("")
    lines.append("int e6_main(void) {")
    lines.append("  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;")
    lines.append("  bosc_ame_private_ddr_fail_hook = vit_publish_private_ddr_failure;")
    lines.append("  vit_install_trap_handler();")
    lines.append("  nh_e6_cache_writes();")
    lines.append("  nh_e6_progress(0x1001u);")
    lines.append(
        "  MemRef1 model_result = {(void *)0, (void *)model_logits, 0, {1000}, {1}};"
    )
    for arg in args:
        rank = len(arg["shape"])
        if arg["kind"] == "workspace":
            pointer = (
                "(void *)(uintptr_t)(model_workspace_blob + %d)"
                % arg["workspace_offset"]
            )
        else:
            pointer = arg["source"]
        lines.append(
            "  MemRef%d a_%d = {(void *)0, %s, 0, {%s}, {%s}};"
            % (
                rank,
                arg["index"],
                pointer,
                ", ".join(str(x) for x in arg["shape"]),
                ", ".join(str(x) for x in strides_of(arg["shape"])),
            )
        )
    call_args = ", ".join(["&model_result"] + ["&a_%d" % arg["index"] for arg in args])
    lines.append("  uint64_t begin = nr_cycles();")
    lines.append("  _mlir_ciface_vit(" + call_args + ");")
    lines.append("  uint64_t end = nr_cycles();")
    lines.append("  nh_e6_progress(0x2001u);")
    lines.append("  model_cycles = end - begin;")
    lines.append(
        "  const float *logits = (const float *)((const char *)model_result.aligned + model_result.offset * 4);"
    )
    lines.append("  const float *reference = (const float *)model_reference_blob;")
    lines.append("  unsigned errors = 0;")
    lines.append("  uint32_t maximum_bits = 0;")
    lines.append("  uint32_t top_index = 0;")
    lines.append("  float top_value = logits[0];")
    lines.append("  for (unsigned i = 0; i < 1000u; ++i) {")
    lines.append("    float actual = logits[i], expected = reference[i];")
    lines.append("    if (actual > top_value) { top_value = actual; top_index = i; }")
    lines.append("    union { float f; uint32_t u; } a = {actual}, b = {expected}, d;")
    lines.append("    d.f = actual - expected;")
    lines.append("    if (d.f < 0.0f) d.f = -d.f;")
    lines.append("    if (d.u > maximum_bits) maximum_bits = d.u;")
    lines.append("    if (!(actual == actual)) { ++errors; continue; }")
    lines.append("    float magnitude = expected < 0.0f ? -expected : expected;")
    lines.append("    if (d.f > 0.5f + 0.1f * magnitude) ++errors;")
    lines.append("  }")
    lines.append("  model_logits_fnv64 = model_fnv1a64(logits, 1000u * sizeof(float));")
    lines.append("  model_top1 = top_index;")
    lines.append("  model_max_abs_bits = maximum_bits;")
    lines.append("  model_returned = 1u;")
    lines.append("  model_validation_executed = 1u;")
    lines.append("  model_checked = 1000u;")
    lines.append("  model_errors = errors;")
    lines.append("  nh_e6_cache_writes();")
    lines.append('  __asm__ volatile("fence rw, rw" ::: "memory");')
    lines.append("  return errors == 0u ? 0 : 1;")
    lines.append("}")
    (ctx.output / "model_launcher.c").write_text("\n".join(lines) + "\n")

    params_bin = (ctx.output / "params.bin").resolve()
    input_bin = (
        ctx.assets.parent
        / "build"
        / "model-bin"
        / "inputs"
        / "model-input"
        / "pixel-values.f32"
    ).resolve()
    if ctx.model_input is not None:
        input_bin = (ctx.model_input / "pixel-values.f32").resolve()
    reference_bin = (ctx.model_input / "reference-logits.f32").resolve()
    for path in (params_bin, input_bin, reference_bin):
        if not path.is_file():
            raise ValueError("missing embedded blob %s" % path)
    asm = []
    for section, symbol, blob in (
        (".model_params", "model_params_blob", params_bin),
        (".model_input", "model_input_blob", input_bin),
        (".model_reference", "model_reference_blob", reference_bin),
    ):
        asm.append('  .section %s,"a",@progbits' % section)
        asm.append("  .balign 64")
        asm.append("  .globl %s" % symbol)
        asm.append("%s:" % symbol)
        asm.append('  .incbin "%s"' % str(blob).replace("\\", "/"))
    (ctx.output / "model_blobs.S").write_text("\n".join(asm) + "\n")
    workspace_asm = [
        '  .section .model_workspace,"aw",@nobits',
        "  .balign 64",
        "  .globl model_workspace_blob",
        "model_workspace_blob:",
        "  .space %d" % workspace_total,
        "  .size model_workspace_blob, %d" % workspace_total,
    ]
    (ctx.output / "model_workspace.S").write_text("\n".join(workspace_asm) + "\n")

    arg_map = {
        "vit_args": len(args),
        "ciface_args": len(args) + 1,
        "result": {"shape": [1, 1000], "dtype": "f32", "nbytes": 4000},
        "params": len([arg for arg in args if arg["kind"] == "param"]),
        "workspaces": len([arg for arg in args if arg["kind"] == "workspace"]),
        "workspace_bytes": workspace_total,
        "args": args,
    }
    (ctx.output / "ARG-MAP.json").write_text(json.dumps(arg_map, indent=2) + "\n")
    log(
        "generate: launcher with %d @vit args (%d params, %d workspace, %.1f MiB)"
        % (
            len(args),
            arg_map["params"],
            arg_map["workspaces"],
            arg_map["workspace_bytes"] / 2**20,
        )
    )


# ---------------------------------------------------------------------------
# Compilation and link
# ---------------------------------------------------------------------------


def compile_c(ctx, source, output, extra_flags=()):
    flags = list(CFLAGS) + list(extra_flags)
    flags += [
        "-I" + str(V),
        "-I" + str(V / "tools"),
        "-I" + str(ctx.output),
        "-I" + str(RUNTIME / "nh_e6"),
        "-I" + str(RUNTIME / "runtime"),
        "-I" + str(RUNTIME / "uart"),
        "-I" + str(RUNTIME / "nr"),
    ]
    run(
        [ctx.clang, *flags, "-c", source, "-o", output],
        ctx.output / ("compile-" + Path(output).stem + ".log"),
    )


def compile_asm(ctx, source, output, extra_flags=()):
    flags = list(CFLAGS) + list(extra_flags) + ["-x", "assembler-with-cpp"]
    run(
        [ctx.clang, *flags, "-c", source, "-o", output],
        ctx.output / ("compile-" + Path(output).stem + ".log"),
    )


def keep_object(ctx, objects, path):
    if path not in objects:
        objects.append(path)


KERNEL_ENTRY = re.compile(
    r"^extern void (_mlir_ciface_kernel_[A-Za-z0-9_]+)\(([^)]*)\);",
    re.MULTILINE,
)


def kernel_entries(source):
    """Parse the typed kernel entries the generated model adapters call."""
    entries = {}
    for symbol, params in KERNEL_ENTRY.findall(source):
        ranks = [int(rank) for rank in re.findall(r"MemRef(\d+)", params)]
        if not ranks:
            raise ValueError("kernel entry without typed memrefs: " + symbol)
        entries.setdefault(symbol, ranks)
    return entries


def operator_trace_adapters(ctx, source):
    """Replace adapter trace strings with numeric MMIO events for this image.

    NH uses a read-only message table; no cross-core DDR pointer or writable
    string is part of the logging handshake. Leave canonical adapters intact.
    """
    text = source.read_text()
    messages = []
    phases = {}

    def replace(match):
        message = json.loads(match.group(1))
        marker = re.fullmatch(
            r"\[vit-op\] (\d+)/(\d+) symbol=\S+ case=\S+ (BEGIN|END)\r\n",
            message,
        )
        if marker is None:
            raise ValueError("unrecognized operator trace message: " + repr(message))
        ordinal, total, phase = marker.groups()
        key = (int(ordinal), phase)
        if key in phases:
            raise ValueError("duplicate operator trace phase: " + repr(key))
        phases[key] = int(total)
        event = len(messages)
        messages.append(message)
        return "  vit_model_trace_event(%du);" % event

    instrumented = re.sub(
        r'^  VIT_TRACE_MARK\(("(?:\\.|[^"\\])*")\);$',
        replace,
        text,
        flags=re.MULTILINE,
    )
    totals = set(phases.values())
    if len(totals) != 1:
        raise ValueError("operator trace has no unique graph size")
    total = totals.pop()
    expected = {(i, phase) for i in range(1, total + 1) for phase in ("BEGIN", "END")}
    if set(phases) != expected or len(messages) > 4096:
        raise ValueError(
            "operator trace must cover every BEGIN/END within 12 event bits"
        )
    header = [
        "/* Generated by tools/build_model_bin.py: immutable NH trace messages. */",
        "static const char *const vit_op_trace_messages[] = {",
        *("  " + json.dumps(message) + "," for message in messages),
        "};",
        "",
    ]
    (ctx.output / "model_op_trace_messages.h").write_text("\n".join(header))
    path = ctx.output / "model_op_trace_adapters.c"
    path.write_text(
        "#include <stdint.h>\nvoid vit_model_trace_event(uint32_t event);\n"
        + instrumented
    )
    log("compile: operator trace covers %d BEGIN/END events" % len(messages))
    return path


def cache_bracket_wrappers(ctx, adapters_source):
    """Mirror the per-operator board wrappers around every typed kernel entry.

    The BOSCAME runtime switches CSR 0xbc0 between write-through-store and
    uncached-load policies inside each kernel (DMA phases, private-DDR
    bookkeeping). The graph glue builds CPU memref descriptors between calls,
    so restore the write policy at both boundaries. Allocator controls and CPU
    stacks must also stay outside the PMP window whose load policy changes.
    """
    entries = kernel_entries(adapters_source)
    lines = [
        "/* Generated by tools/build_model_bin.py: cache-policy brackets around",
        " * every typed kernel entry, mirroring the per-operator board wrappers. */",
        '#include "support.h"',
        '#include "nh_e6_platform.h"',
        "",
        "extern volatile uint64_t vit_kernel_case;",
        "extern volatile uint64_t vit_kernel_calls;",
        "",
    ]
    for ordinal, symbol in enumerate(sorted(entries), start=1):
        ranks = entries[symbol]
        params = ", ".join(
            "MemRef%d *a%d" % (rank, index) for index, rank in enumerate(ranks)
        )
        args = ", ".join("a%d" % index for index in range(len(ranks)))
        lines.append("extern void __real_%s(%s);" % (symbol, params))
        lines.append("void __wrap_%s(%s) {" % (symbol, params))
        lines.append("  vit_kernel_case = %du;" % ordinal)
        lines.append("  ++vit_kernel_calls;")
        lines.append("  nh_e6_cache_writes();")
        lines.append("  __real_%s(%s);" % (symbol, args))
        lines.append('  __asm__ volatile("fence iorw, iorw" ::: "memory");')
        # Phase-boundary policy: the graph glue between kernel calls builds
        # memref descriptors on the stack, so leave the coherent write-through
        # store / cached-load policy active.  The kernels switch to the
        # uncached-load policy internally when they read DMA-written data.
        lines.append("  nh_e6_cache_writes();")
        lines.append("}")
        lines.append("")
    path = ctx.output / "model_kernel_cache_wrappers.c"
    path.write_text("\n".join(lines))
    return path, entries


def stage_compile(ctx):
    objects = []
    adapters_path = ctx.replacement / "vit_triton_adapters.c"
    if os.environ.get("MODEL_OP_TRACE") == "1":
        adapters_path = operator_trace_adapters(ctx, adapters_path)
    graph_object = ctx.output / "vit_graph.o"
    run(
        [ctx.llc, *LLC_FLAGS, "-o", graph_object, ctx.output / "vit.ciface.ll"],
        ctx.output / "compile-graph.log",
    )
    keep_object(ctx, objects, graph_object)

    compile_asm(ctx, ctx.output / "model_blobs.S", ctx.output / "model_blobs.o")
    keep_object(ctx, objects, ctx.output / "model_blobs.o")
    compile_asm(ctx, ctx.output / "model_workspace.S", ctx.output / "model_workspace.o")
    keep_object(ctx, objects, ctx.output / "model_workspace.o")
    compile_asm(ctx, V / "tools" / "model_trap.S", ctx.output / "model_trap.o")
    keep_object(ctx, objects, ctx.output / "model_trap.o")

    compile_c(ctx, ctx.output / "model_launcher.c", ctx.output / "model_launcher.o")
    keep_object(ctx, objects, ctx.output / "model_launcher.o")
    support_flags = []
    if os.environ.get("MODEL_DDR_TRACE") == "1":
        support_flags = ["-DMODEL_DDR_TRACE", "-DBOSC_AME_PRIVATE_DDR_TRACE=1"]
    if os.environ.get("MODEL_OP_TRACE") == "1":
        support_flags.append("-DMODEL_OP_TRACE=1")
    compile_c(
        ctx,
        V / "tools" / "model_support.c",
        ctx.output / "model_support.o",
        support_flags,
    )
    keep_object(ctx, objects, ctx.output / "model_support.o")

    compile_c(ctx, adapters_path, ctx.output / "model_adapters.o")
    keep_object(ctx, objects, ctx.output / "model_adapters.o")
    adapters_source = (ctx.replacement / "vit_triton_adapters.c").read_text()
    wrappers = None
    if os.environ.get("MODEL_NO_KERNEL_CACHE_BRACKETS") != "1":
        wrappers, _kernel_cache_entries = cache_bracket_wrappers(ctx, adapters_source)
        compile_c(ctx, wrappers, ctx.output / "model_kernel_cache_wrappers.o")
        keep_object(ctx, objects, ctx.output / "model_kernel_cache_wrappers.o")

    report = json.loads((ctx.replacement / "triton-call-replacement.json").read_text())
    if str(V) not in sys.path:
        sys.path.insert(0, str(V))
    if str(V / "tools") not in sys.path:
        sys.path.insert(0, str(V / "tools"))
    import board_config
    import cases as cases_module

    required = set(report["distinct_cases"])
    # patch_embed's generated adapter executes its own child kernels, so link
    # the declared dependencies of every case as well.
    for case in sorted(required):
        try:
            description = cases_module.describe(case)
        except Exception:
            continue
        required.update(board_config.dependencies(description))
    available = sorted(
        directory.name
        for directory in ctx.kernels.iterdir()
        if directory.is_dir()
        and (directory / "adapter.c").is_file()
        and (directory / "kernel.o").is_file()
    )
    missing_required = sorted(required.difference(available))
    if missing_required:
        raise ValueError("missing built cases: %s" % missing_required)
    log("compile: linking %d cases (closure)" % len(required))
    for case in sorted(required):
        case_dir = ctx.kernels / case
        adapter_o = ctx.output / ("adapter_" + case + ".o")
        compile_c(ctx, case_dir / "adapter.c", adapter_o)
        keep_object(ctx, objects, adapter_o)
        keep_object(ctx, objects, case_dir / "kernel.o")

    rflags = [
        "-Dinit_uart=board_init_uart",
        "-DNH_E6_WAIT_ITERATIONS=0xffffffffu",
        "-DNH_E6_MMIO_READ32=vit_board_mmio_read32",
        "-include",
        str(V / "tools" / "board_support.h"),
    ]
    compile_asm(ctx, RUNTIME / "nh_e6" / "nh_e6_boot.S", ctx.output / "boot.o")
    keep_object(ctx, objects, ctx.output / "boot.o")
    compile_c(
        ctx, RUNTIME / "nh_e6" / "nh_e6_platform.c", ctx.output / "platform.o", rflags
    )
    keep_object(ctx, objects, ctx.output / "platform.o")
    compile_c(ctx, RUNTIME / "uart" / "uart.c", ctx.output / "uart.o")
    keep_object(ctx, objects, ctx.output / "uart.o")
    dma_flags = ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"]
    if os.environ.get("MODEL_DDR_TRACE") == "1":
        dma_flags.append("-DBOSC_AME_PRIVATE_DDR_TRACE=1")
    compile_c(
        ctx, RUNTIME / "runtime" / "bosc_ame_dma.c", ctx.output / "dma.o", dma_flags
    )
    keep_object(ctx, objects, ctx.output / "dma.o")
    compile_c(ctx, RUNTIME / "nr" / "nr_math.c", ctx.output / "math.o")
    keep_object(ctx, objects, ctx.output / "math.o")
    (ctx.output / "objects.json").write_text(
        json.dumps([str(path) for path in objects], indent=2) + "\n"
    )
    log("compile: %d objects" % len(objects))
    return objects


def stage_link(ctx, objects):
    # The NH boot stub reaches the platform trap handler with a JAL (+-1 MiB),
    # so keep the platform/runtime text early in .text.
    early = ("boot.o", "platform.o", "uart.o", "dma.o", "math.o")
    objects = sorted(
        objects,
        key=lambda path: (
            Path(path).name not in early,
            early.index(Path(path).name) if Path(path).name in early else 0,
        ),
    )
    elf = ctx.output / "vit_model.elf"
    linker_script = V / "board_model.ld"
    wrap_flags = []
    if os.environ.get("MODEL_NO_KERNEL_CACHE_BRACKETS") != "1":
        adapters_source = (ctx.replacement / "vit_triton_adapters.c").read_text()
        wrap_flags += [
            "--wrap=" + symbol for symbol in sorted(kernel_entries(adapters_source))
        ]
    run(
        [
            ctx.ld,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            linker_script,
            "-Map=" + str(ctx.output / "vit_model.map"),
            "-o",
            elf,
            *wrap_flags,
            *objects,
        ],
        ctx.output / "link.log",
    )
    audit = {"status": "SKIPPED"}
    audit_output = ctx.output / "elf-audit.json"
    audit_flags = []
    if os.environ.get("ALLOW_VECTOR_ORDERED_REDUCTIONS") == "1":
        audit_flags.append("--allow-vector-ordered-reductions")
    if os.environ.get("ALLOW_VECTOR_MAX_REDUCTIONS") == "1":
        audit_flags.append("--allow-vector-max-reductions")
    try:
        run(
            [
                sys.executable,
                str(BUDDY_TOOLS / "check_nr_elf.py"),
                elf,
                "--objdump",
                ctx.objdump,
                *audit_flags,
                "--output",
                audit_output,
            ],
            ctx.output / "audit.log",
        )
        audit = json.loads(audit_output.read_text())
    except Exception as error:  # audit failures are recorded, not hidden
        audit = {"status": "FAIL", "reason": str(error)}
    binary = ctx.output / "vit_model.bin"
    run([ctx.objcopy, "-O", "binary", elf, binary], ctx.output / "objcopy.log")

    undefined = []
    try:
        nm = subprocess.run(
            [str(ctx.nm), "-u", str(elf)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        undefined = [
            line.split()[-1] for line in nm.stdout.splitlines() if line.strip()
        ]
    except Exception:
        undefined = []
    sections = subprocess.run(
        [str(ctx.objdump), "-h", str(elf)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    manifest = {
        "schema": 1,
        "model": "google/vit-base-patch16-224",
        "mode": "whole-model NH/E6 flat image",
        "operator_trace": os.environ.get("MODEL_OP_TRACE") == "1",
        "operator_trace_protocol": (
            "mailbox-sequence-result-ack-v2"
            if os.environ.get("MODEL_OP_TRACE") == "1"
            else None
        ),
        "scope": "artifact generation only; no board load, no hardware run, no numerical claim",
        "vit_args": 734,
        "ciface_args": 735,
        "params_bytes": (ctx.output / "params.bin").stat().st_size,
        "bin_bytes": binary.stat().st_size,
        "bin_sha256": sha256_file(binary),
        "elf_bytes": elf.stat().st_size,
        "elf_sha256": sha256_file(elf),
        "params_sha256": sha256_file(ctx.output / "params.bin"),
        "undefined_symbols": undefined,
        "audit": audit,
        "elf_sections": sections.strip().splitlines(),
        "objects": len(objects),
    }
    (ctx.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(
        "link: %s (%.1f MiB), bin (%.1f MiB)"
        % (elf, elf.stat().st_size / 2**20, binary.stat().st_size / 2**20)
    )
    log("link: undefined symbols: %s" % (undefined if undefined else "none"))
    log(
        "link: ELF audit: %s"
        % (audit.get("status") if isinstance(audit, dict) else audit)
    )
    return elf, binary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class Context:
    pass


def resolve_tools(ctx):
    ctx.buddy_opt = os.environ.get("BUDDY_OPT", str(BUDDY_BIN / "buddy-opt"))
    ctx.buddy_translate = os.environ.get(
        "BUDDY_TRANSLATE", str(BUDDY_BIN / "buddy-translate")
    )
    ctx.clang = str(LLVM_BIN / "clang")
    ctx.llc = str(LLVM_BIN / "llc")
    ctx.objcopy = str(LLVM_BIN / "llvm-objcopy")
    ctx.objdump = str(LLVM_BIN / "llvm-objdump")
    ctx.nm = str(LLVM_BIN / "llvm-nm")
    linker = (
        os.environ.get("LD_LLD")
        or shutil.which("ld.lld")
        or shutil.which("riscv64-unknown-elf-ld")
    )
    if not linker:
        raise RuntimeError("no RISC-V linker found (set LD_LLD or put ld.lld on PATH)")
    ctx.ld = linker
    for tool in (
        ctx.buddy_opt,
        ctx.buddy_translate,
        ctx.clang,
        ctx.llc,
        ctx.objcopy,
        ctx.objdump,
        ctx.nm,
    ):
        if not Path(tool).is_file():
            raise RuntimeError("missing tool %s" % tool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--kernels", type=Path, default=KERNELS_DEFAULT)
    parser.add_argument("--replacement", type=Path, default=REPLACEMENT_DEFAULT)
    parser.add_argument("--assets", type=Path, default=ASSETS_DEFAULT)
    parser.add_argument("--model-input", type=Path, default=MODEL_INPUT_DEFAULT)
    parser.add_argument(
        "--reuse-lowered",
        action="store_true",
        help="reuse inputs/vit.ciface.ll instead of lowering again",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "lower", "pack", "generate", "compile", "link"),
    )
    args = parser.parse_args()
    ctx = Context()
    ctx.output = args.output.resolve()
    ctx.output.mkdir(parents=True, exist_ok=True)
    ctx.inputs = ctx.output / "inputs"
    ctx.kernels = args.kernels.resolve()
    ctx.replacement = args.replacement.resolve()
    ctx.assets = args.assets.resolve()
    ctx.model_input = args.model_input.resolve()
    ctx.reuse_lowered = args.reuse_lowered
    if not ctx.replacement.is_dir():
        raise SystemExit("missing replacement directory %s" % ctx.replacement)
    if str(BUDDY_PYTHON) not in sys.path:
        sys.path.insert(0, str(BUDDY_PYTHON))
    resolve_tools(ctx)
    if args.stage in ("all", "lower"):
        stage_lower(ctx)
    if args.stage in ("all", "pack"):
        params = stage_pack(ctx)
    if args.stage in ("all", "generate"):
        if "params" not in locals():
            layout = json.loads((ctx.output / "params-layout.json").read_text())
            params = {entry["name"]: entry for entry in layout["entries"]}
        stage_generate(ctx, params)
    if args.stage in ("all", "compile"):
        objects = stage_compile(ctx)
    if args.stage in ("all", "link"):
        if "objects" not in locals():
            object_list = ctx.output / "objects.json"
            if object_list.is_file():
                objects = [Path(path) for path in json.loads(object_list.read_text())]
            else:
                objects = sorted(ctx.output.glob("*.o"))
        stage_link(ctx, objects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

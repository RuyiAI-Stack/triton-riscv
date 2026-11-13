#!/usr/bin/env python3
"""Stop an isolated real-fixture convolution after tile 1 and export scratch."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute(command, log):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    log.write_text(
        "$ " + repr(list(map(str, command))) + "\n" + result.stdout + result.stderr
    )
    if result.returncode:
        raise RuntimeError(str(log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "physical-plan.json",
        "parameters.bin",
        "parameters.json",
        "input.f32",
        "w8a8-logits.f32",
        "original-logits.f32",
        "reference-provenance.json",
    ):
        shutil.copy2(a.base / name, a.output / name)
    shutil.copytree(a.base / "runner", a.output / "runner", dirs_exist_ok=True)
    plan = json.loads((a.output / "physical-plan.json").read_text())
    name = plan["calls"][0]["target"]
    current = MODEL / "build/kernels" / name
    overlay = a.output / "kernels" / name
    overlay.mkdir(parents=True, exist_ok=True)
    for filename in (
        "kernel.o",
        "kernel.boscame.mlir",
        "kernel.nr.s",
        "kernel.llvm.mlir",
    ):
        shutil.copy2(current / filename, overlay / filename)
    adapter = (current / "adapter.c").read_text()
    old = "for (int32_t block = 0; block < 324; ++block)"
    assert adapter.count(old) == 1
    adapter = adapter.replace(old, "for (int32_t block = 0; block < 2; ++block)")
    (overlay / "adapter.c").write_text(adapter)
    image = a.output / "image"
    execute(
        [
            sys.executable,
            HERE / "build_model_image.py",
            "--plan-dir",
            a.output,
            "--runner",
            a.output / "runner",
            "--output",
            image,
            "--bounded-probe",
            "--strict-probe",
            "--host-numerical-validation",
            "--conv-overlay",
            overlay.parent,
        ],
        a.output / "build-image.log",
    )
    support = (HERE / "model_board_support.c").read_text()
    marker = "  print_uart(ok?protocol_pass:protocol_fail);"
    assert support.count(marker) == 1
    scratch = plan["workspace"]["provisional_total_bytes"] - 10112
    declarations = []
    buffers = []
    copies = []
    for label, offset, length, dtype, shape in (
        ("QWeight", 0, 1536, "|i1", [24, 64]),
        ("WeightScale", 1536, 96, "<f4", [24]),
        ("QPatch", 1664, 4096, "|i1", [64, 64]),
        ("PatchScale", 5760, 256, "<f4", [64]),
        ("AccumulatorLastTile", 6016, 4096, "<i4", [16, 64]),
    ):
        buffer = "diagnostic_capture_" + label
        buffers.append(
            "static unsigned char "
            + buffer
            + "["
            + str(length)
            + '] __attribute__((section(".model_export"),aligned(64)));'
        )
        copies.append(
            "  coherent_copy("
            + buffer
            + ",model_workspace_blob+"
            + str(scratch + offset)
            + "u,"
            + str(length)
            + "u);"
        )
        declarations.append(
            "  output("
            + json.dumps(label)
            + ","
            + buffer
            + ","
            + str(length)
            + ","
            + json.dumps(dtype)
            + ","
            + json.dumps(json.dumps(shape))
            + ");"
        )
    export = "void efficientnet_export_results(const float *logits){"
    assert support.count(export) == 1
    support = support.replace(
        export, "\n".join(buffers) + "\n" + export + "\n" + "\n".join(copies)
    )
    support = support.replace(marker, "\n".join(declarations) + "\n" + marker)
    support_path = image / "diagnostic-model-support.c"
    support_path.write_text(support)
    command = ast.literal_eval(
        (image / "compile-model_support.log").read_text().splitlines()[0][2:]
    )
    command = [
        str(support_path) if value == str(HERE / "model_board_support.c") else value
        for value in command
    ]
    execute(command, image / "compile-diagnostic-model-support.log")
    command = ast.literal_eval((image / "link.log").read_text().splitlines()[0][2:])
    execute(command, image / "link-diagnostic.log")
    command = ast.literal_eval((image / "audit.log").read_text().splitlines()[0][2:])
    execute(command, image / "audit-diagnostic.log")
    command = ast.literal_eval((image / "objcopy.log").read_text().splitlines()[0][2:])
    execute(command, image / "objcopy-diagnostic.log")
    manifest_path = image / "image-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["elf_sha256"] = sha(image / "efficientnet_b3_model.elf")
    manifest["image_sha256"] = sha(image / "efficientnet_b3_model.bin")
    manifest["image_bytes"] = (image / "efficientnet_b3_model.bin").stat().st_size
    llvm = MODEL.parents[2] / "buddy-mlir/llvm/build/bin"
    symbols = subprocess.check_output(
        [str(llvm / "llvm-nm"), "-n", str(image / "efficientnet_b3_model.elf")],
        text=True,
    )
    for line in symbols.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in manifest["elf_layout_symbols"]:
            manifest["elf_layout_symbols"][parts[2]] = "0x" + parts[0]
    manifest["output_artifacts"] += [
        "QWeight",
        "WeightScale",
        "QPatch",
        "PatchScale",
        "AccumulatorLastTile",
    ]
    manifest["diagnostic_tile_capture"] = {
        "scope": "tiles 0 and 1 only; all unmodified raw weight/im2col/MMA/dequant stages",
        "stopped_after_spatial": 127,
        "QPatch_layout": "[spatial_within_tile, K]",
        "accumulator_layout": "[output_channel_within_mtile, spatial_within_tile]",
        "accumulator_channels": [16, 31],
        "preserved_kernel_o_sha256": sha(current / "kernel.o"),
        "original_adapter_source_sha256": sha(current / "adapter.c"),
        "diagnostic_adapter_source_sha256": sha(overlay / "adapter.c"),
        "diagnostic_model_support_sha256": sha(support_path),
        "builder_sha256": sha(__file__),
        "scratch_offset": scratch,
        "scratch_export": "existing volatile coherent_copy after both tiles; low .model_export buffers",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot_path = image / "input-snapshot-manifest.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["image_sha256"] = manifest["image_sha256"]
        snapshot["image_manifest_sha256"] = sha(manifest_path)
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(manifest["diagnostic_tile_capture"], indent=2))


if __name__ == "__main__":
    main()

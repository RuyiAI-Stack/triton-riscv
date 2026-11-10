#!/usr/bin/env python3
"""Build bounded probes of the existing scalar convolution; never submit jobs.

The original kernel object, inputs, descriptor checks, grid and program order
are retained. Generated diagnostic copies add phase/cycle checkpoints, a
program-prefix bound and output sentinels. Production sources are untouched.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
CASE = "conv2d_288x288_3x40_k3_s2"
PROGRAMS = 40 * 144 * 144


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("expected one diagnostic anchor: " + old)
    return source.replace(old, new)


def launch_source(source, limit):
    source = replace_once(
        source,
        '#include "support.h"',
        '#include "support.h"\n#include "probe.h"',
    )
    source = replace_once(
        source, "int launch(void) {", "int launch(void) {\n  effnet_probe_phase(1, 0);"
    )
    source = replace_once(source, "o[i] = 0;", "o[i] = -12345.0f;")
    call = f"  _mlir_ciface_kernel_{CASE}(&X, &W, &B, &O);"
    source = replace_once(
        source,
        call,
        "  effnet_probe_phase(2, 0);\n"
        + call
        + f"\n  effnet_probe_phase(4, {limit});\n"
        + "  for (int i = 0; i < 829440; ++i)\n"
        + f"    if (o[i] != (i < {limit} ? 0.0f : -12345.0f)) ++errors;\n"
        + f"  effnet_probe_phase(5, {limit});",
    )
    source = replace_once(
        source,
        '  nr_puts("numerical acceptance is simcheck\\n");\n  return 0;',
        '  return print_check("zero-fixture-output-and-sentinels", errors, 0);',
    )
    return source


def adapter_source(source, limit, interval):
    source = replace_once(
        source,
        '#include "support.h"',
        '#include "support.h"\n#include "probe.h"',
    )
    old = (
        "  for (int32_t x = 0; x < 40; ++x)\n"
        "    for (int32_t y = 0; y < 144; ++y)\n"
        "      for (int32_t z = 0; z < 144; ++z)\n"
        f"        _mlir_ciface_triton_{CASE}(&u0, &u1, &u2, &u3, 40, 144, 144, x, y, z);"
    )
    new = (
        "  uint32_t completed = 0;\n"
        "  for (int32_t x = 0; x < 40; ++x)\n"
        "    for (int32_t y = 0; y < 144; ++y)\n"
        "      for (int32_t z = 0; z < 144; ++z) {\n"
        f"        _mlir_ciface_triton_{CASE}(&u0, &u1, &u2, &u3, 40, 144, 144, x, y, z);\n"
        "        ++completed;\n"
        f"        if (completed == 1 || completed == 144 || completed % {interval} == 0 || completed == {limit})\n"
        "          effnet_probe_phase(3, completed);\n"
        f"        if (completed == {limit}) return;\n"
        "      }"
    )
    return replace_once(source, old, new)


def board_source(source, slots):
    source = replace_once(
        source,
        '#include "uart.h"',
        '#include "uart.h"\n#include "probe.h"',
    )
    declarations = f"""
typedef struct __attribute__((aligned(64))) {{
  uint64_t elapsed_cycles, adapter_cycles;
  uint32_t phase, completed;
  unsigned char padding[40];
}} ProbeSlot;
static volatile ProbeSlot probe_slots[{slots}];
static unsigned probe_written;
static unsigned probe_reported __attribute__((section(".data.probe_nh"))) = ~0u;
static uint64_t probe_start, probe_adapter_start;

void effnet_probe_phase(uint32_t phase, uint32_t completed) {{
  if (probe_written >= {slots}) __builtin_trap();
  uint64_t now = nr_cycles();
  if (phase == 2) probe_adapter_start = now;
  volatile ProbeSlot *slot = &probe_slots[probe_written++];
  slot->elapsed_cycles = now - probe_start;
  slot->adapter_cycles = probe_adapter_start ? now - probe_adapter_start : 0;
  slot->phase = phase;
  slot->completed = completed;
  __asm__ volatile("fence rw, rw" ::: "memory");
  nh_e6_progress(0xc1000000u | probe_written);
}}

static void probe_report_to(unsigned count) {{
  if (count > {slots}) return;
  while (probe_reported < count) {{
    const volatile ProbeSlot *slot = &probe_slots[probe_reported++];
    print_uart("[conv-probe] phase=");
    print_uart_dec32(slot->phase);
    print_uart(" completed=");
    print_uart_dec32(slot->completed);
    print_uart(" elapsed_cycles=0x");
    print_uart_hex64(slot->elapsed_cycles);
    print_uart(" adapter_cycles=0x");
    print_uart_hex64(slot->adapter_cycles);
    print_uart("\\r\\n");
  }}
}}

uint32_t effnet_probe_mmio_read32(uintptr_t address) {{
  uint32_t value = *(volatile uint32_t *)address;
  if (address == NH_E6_MAILBOX_REG && (value & 0xff000000u) == 0xc1000000u) {{
    __asm__ volatile("fence r, r" ::: "memory");
    probe_report_to(value & 0x00ffffffu);
  }}
  return value;
}}
"""
    source = replace_once(
        source, "int e6_main(void) {", declarations + "\nint e6_main(void) {"
    )
    source = replace_once(
        source,
        "  uint64_t begin = nr_cycles();",
        "  uint64_t begin = nr_cycles();\n  probe_start = begin;",
    )
    source = replace_once(
        source,
        "  init_uart(frequency, baud);",
        "  init_uart(frequency, baud);\n  probe_reported = 0;",
    )
    source = replace_once(
        source,
        "void nh_e6_app_report(void) {",
        "void nh_e6_app_report(void) {\n  probe_report_to(probe_written);",
    )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--programs",
        type=int,
        default=20736,
        help="run first N programs in the original grid (max 829440)",
    )
    parser.add_argument("--interval", type=int, default=20736)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--buddy-root",
        type=Path,
        default=Path(os.environ.get("BUDDY_ROOT", REPO / "buddy-mlir")),
    )
    args = parser.parse_args()
    if not 1 <= args.programs <= PROGRAMS or args.interval < 1:
        parser.error("invalid prefix or interval")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("use a fresh output directory to preserve evidence")
    out.mkdir(parents=True, exist_ok=True)
    buddy = args.buddy_root.resolve()
    llvm = Path(
        os.environ.get(
            "LLVM_BIN", os.environ.get("LLVM_BINARY_DIR", buddy / "llvm/build/bin")
        )
    )
    clang = os.environ.get("CLANG", str(llvm / "clang"))
    linker = os.environ.get("LINKER") or (
        str(llvm / "ld.lld") if (llvm / "ld.lld").exists() else shutil.which("ld.lld")
    )
    source = args.kernels.resolve() / CASE
    original = out / "original"
    original.mkdir()
    inputs = [
        ROOT / CASE / "kernel.py",
        ROOT / CASE / "launch.c",
        source / "adapter.c",
        source / "kernel.o",
        source / "kernel.ll",
        ROOT / "board_support.c",
        ROOT / "board.mk",
        ROOT / "support.h",
        ROOT / "link.ld",
        Path(__file__).resolve(),
    ]
    manifest = {
        "case": CASE,
        "programs": args.programs,
        "full_grid_programs": PROGRAMS,
        "grid": [40, 144, 144],
        "progress_interval": args.interval,
        "input_fixture": "original zero X/Weight/Bias",
        "output_fixture": "diagnostic -12345 sentinel; selected prefix must become zero",
        "kernel_object_reused": True,
        "source_sha256": {},
        "phases": {
            1: "initialization-start",
            2: "initialization-done-adapter-start",
            3: "program-prefix-progress",
            4: "adapter-returned",
            5: "output-and-sentinel-check-done",
        },
    }
    for index, path in enumerate(inputs):
        manifest["source_sha256"][str(path.relative_to(REPO))] = digest(path)
        shutil.copy2(path, original / (f"{index:02d}-" + path.name))
    shutil.copy2(source / "kernel.o", out / "kernel.o")
    shutil.copy2(ROOT / "link.ld", out / "link.ld")
    (out / "probe.h").write_text(
        "#include <stdint.h>\nvoid effnet_probe_phase(uint32_t, uint32_t);\n"
        "uint32_t effnet_probe_mmio_read32(uintptr_t);\n"
    )
    (out / "launch.c").write_text(
        launch_source((ROOT / CASE / "launch.c").read_text(), args.programs)
    )
    (out / "adapter.c").write_text(
        adapter_source((source / "adapter.c").read_text(), args.programs, args.interval)
    )
    slots = args.programs // args.interval + 10
    (out / "board_support.c").write_text(
        board_source((ROOT / "board_support.c").read_text(), slots)
    )
    commands = []

    def run(stage, command):
        result = subprocess.run(
            list(map(str, command)), text=True, capture_output=True, check=False
        )
        (out / (stage + ".log")).write_text(result.stdout + result.stderr)
        commands.append(
            {
                "stage": stage,
                "argv": list(map(str, command)),
                "exit_code": result.returncode,
            }
        )
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(stage + ": " + result.stderr[-4000:])

    runtime = buddy / "runtime/boscame"
    flags = [
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
        "-ffunction-sections",
        "-fdata-sections",
        "-ffp-contract=off",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-DBOSC_AME_DMA_ASSERT(x)=",
        "-DUART_FREQ_HZ=5529600",
        "-DVIT_WORKSPACE_BYTES=16777216",
    ]
    for directory in [
        ROOT,
        out,
        *[runtime / name for name in ("runtime", "uart", "nh_e6", "nr")],
    ]:
        flags.extend(["-I", str(directory)])
    compile_inputs = {
        "launch": (out / "launch.c", []),
        "adapter": (out / "adapter.c", []),
        "board_support": (out / "board_support.c", []),
        "boot": (runtime / "nh_e6/nh_e6_boot.S", []),
        "platform": (
            runtime / "nh_e6/nh_e6_platform.c",
            [
                "-include",
                out / "probe.h",
                "-Dinit_uart=effnet_init_uart",
                "-DNH_E6_WAIT_ITERATIONS=0xffffffffu",
                "-DNH_E6_MMIO_READ32=effnet_probe_mmio_read32",
            ],
        ),
        "uart": (runtime / "uart/uart.c", []),
        "dma": (
            runtime / "runtime/bosc_ame_dma.c",
            ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"],
        ),
        "math": (runtime / "nr/nr_math.c", []),
        "ame_sync": (runtime / "nr/ame_sync.c", []),
    }
    objects = []
    for name, (path, extra) in compile_inputs.items():
        obj = out / (name + ".o")
        run(name, [clang, *flags, *extra, "-c", path, "-o", obj])
        objects.append(obj)
        if path.is_relative_to(buddy):
            manifest["source_sha256"][str(path.relative_to(REPO))] = digest(path)
            shutil.copy2(path, original / ("runtime-" + path.name))
    elf, binary = out / "conv-timeout-probe.elf", out / "conv-timeout-probe.bin"
    run(
        "link",
        [
            linker,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            out / "link.ld",
            "-Map=" + str(out / "conv-timeout-probe.map"),
            "-o",
            elf,
            *objects,
            out / "kernel.o",
        ],
    )
    run(
        "elf-audit",
        [
            sys.executable,
            buddy / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            elf,
            "--objdump",
            llvm / "llvm-objdump",
            "--output",
            out / "elf-audit.json",
        ],
    )
    run("objcopy", [llvm / "llvm-objcopy", "-O", "binary", elf, binary])
    manifest["artifacts"] = {
        path.name: digest(path)
        for path in [
            out / "kernel.o",
            out / "launch.c",
            out / "adapter.c",
            out / "board_support.c",
            elf,
            binary,
        ]
    }
    manifest["image_bytes"] = binary.stat().st_size
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(binary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

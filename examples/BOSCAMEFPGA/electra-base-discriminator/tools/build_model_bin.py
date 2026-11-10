#!/usr/bin/env python3
"""Link the complete typed ELECTRA graph and audited real kernels with NH/E6."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument("--output", type=Path, default=ROOT / "build/model-board")
    a = p.parse_args()
    a.kernels = a.kernels.resolve()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    buddy = REPO / "buddy-mlir"
    llvm = buddy / "llvm/build/bin"
    runtime = buddy / "runtime/boscame"
    platform = runtime / "nh_e6"
    shared = ROOT.parent / "tools"
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
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
        "-DELECTRA_PREQUANTIZED_WEIGHTS",
        "-DELECTRA_MODEL_BOARD",
        "-DBOSC_AME_DMA_ASSERT(x)=",
        "-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1",
    ]
    flags += [
        f"-I{x}"
        for x in (
            ROOT,
            ROOT / "model_runtime",
            a.output,
            shared,
            platform,
            runtime / "runtime",
            runtime / "uart",
            runtime / "nr",
        )
    ]
    commands = []

    def run(command, name):
        command = [str(x) for x in command]
        commands.append(command)
        proc = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (a.output / (name + ".log")).write_text(" ".join(command) + "\n" + proc.stdout)
        if proc.returncode:
            raise RuntimeError(name + "\n" + proc.stdout[-5000:])

    objects, inputs = [], []

    def compile(source, name, extra=()):
        dest = a.output / (name + ".o")
        run([llvm / "clang", *flags, *extra, "-c", source, "-o", dest], name)
        objects.append(dest)
        inputs.append(dict(path=str(source), sha256=sha(source)))

    cases = inventory()
    wrappers = ['#include "board.h"', '#include "support.h"']
    for index, case in enumerate(cases):
        name = case["name"]
        params = ", ".join(
            f"MemRef{x['rank']} *a{i}" for i, x in enumerate(case["arguments"])
        )
        args = ", ".join(f"a{i}" for i in range(len(case["arguments"])))
        wrappers += [
            f"extern void model_impl_{name}({params});",
            f"void _mlir_ciface_kernel_{name}({params}) {{",
            f"  unsigned slot = electra_operator_begin({index}u);",
            "  uint64_t begin = nr_cycles();",
            f"  model_impl_{name}({args});",
            '  __asm__ volatile("fence iorw, iorw" ::: "memory");',
            "  uint64_t end = nr_cycles();",
            "  electra_operator_end(slot, end - begin);",
            "  nh_e6_cache_uncached_loads();",
            "}",
        ]
        source = a.kernels / name / "adapter.c"
        compile(
            source,
            "adapter_" + name,
            [f"-D_mlir_ciface_kernel_{name}=model_impl_{name}"],
        )
        kernel = a.kernels / name / "kernel.o"
        objects.append(kernel)
        inputs.append(dict(path=str(kernel), sha256=sha(kernel)))
    (a.output / "wrappers.c").write_text("\n".join(wrappers) + "\n")
    metadata = []
    for stage in range(14):
        name = (
            "embedding"
            if stage == 0
            else f"layer_{stage - 1:02d}"
            if stage < 13
            else "logits"
        )
        count = 128 * 768 if stage < 13 else 128
        shape = "{1,128,768,0}" if stage < 13 else "{128,0,0,0}"
        metadata.append(
            f'{{"{name}","<f4",{3 if stage < 13 else 1},{shape},{count * 4},{stage * 128 * 768 * 4}}}'
        )
    (a.output / "model_outputs.h").write_text(
        "#define VIT_BOARD_EXPECTED_OUTPUTS 14u\n"
        "#define VIT_BOARD_EXPORT_BYTES 5112320u\n"
        "#define VIT_BOARD_OUTPUT_TABLE {" + ",".join(metadata) + "}\n"
    )
    assembly = []
    for symbol, filename, section in [
        ("electra_params", "params.bin", ".model_params"),
        *[
            (f"electra_{name}", name + ".bin", ".model_input")
            for name in ("input_ids", "token_type_ids", "attention_mask")
        ],
    ]:
        path = a.output / filename
        assembly += [
            f'.section {section},"a",@progbits',
            ".balign 64",
            f".global {symbol}",
            f"{symbol}:",
            f'.incbin "{path}"',
            f".global {symbol}_end",
            f"{symbol}_end:",
        ]
        inputs.append(dict(path=str(path), sha256=sha(path)))
    (a.output / "blobs.S").write_text("\n".join(assembly) + "\n")
    compile(a.output / "blobs.S", "blobs")
    compile(a.output / "wrappers.c", "wrappers")
    compile(ROOT / "model_runtime/model.c", "model")
    compile(ROOT / "model_runtime/board.c", "board")
    compile(platform / "nh_e6_boot.S", "boot")
    compile(
        platform / "nh_e6_platform.c",
        "platform",
        [
            "-include",
            str(ROOT / "model_runtime/board.h"),
            "-Dinit_uart=board_init_uart",
            "-DNH_E6_WAIT_ITERATIONS=0xffffffffu",
            "-DNH_E6_MMIO_READ32=electra_board_mmio_read32",
        ],
    )
    compile(runtime / "uart/uart.c", "uart")
    compile(runtime / "runtime/bosc_ame_dma.c", "dma")
    compile(runtime / "nr/nr_math.c", "math")
    elf = a.output / "electra-model.elf"
    binary = a.output / "electra-model.bin"
    linker = llvm / "ld.lld" if (llvm / "ld.lld").exists() else shutil.which("ld.lld")
    run(
        [
            linker,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            ROOT / "board_model.ld",
            "-Map=" + str(a.output / "electra-model.map"),
            "-o",
            elf,
            *objects,
        ],
        "link",
    )
    run(
        [
            sys.executable,
            buddy / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            elf,
            "--objdump",
            llvm / "llvm-objdump",
            "--output",
            a.output / "elf-audit.json",
        ],
        "audit",
    )
    run([llvm / "llvm-objcopy", "-O", "binary", elf, binary], "objcopy")
    report = dict(
        model="google/electra-base-discriminator",
        operator_calls=1571,
        kernels=str(a.kernels),
        binary=str(binary),
        binary_bytes=binary.stat().st_size,
        binary_sha256=sha(binary),
        elf_sha256=sha(elf),
        params_manifest=sha(a.output / "params-manifest.json"),
        inputs=inputs,
        commands=commands,
        board_executed=False,
    )
    report["headers_and_linker"] = {
        str(path): sha(path)
        for path in (
            ROOT / "board_model.ld",
            ROOT / "support.h",
            ROOT / "model_runtime/model.h",
            ROOT / "model_runtime/kernels.h",
            ROOT / "model_runtime/board.h",
            a.output / "params.h",
            a.output / "model_outputs.h",
            shared / "scheduler_output_runtime.h",
            shared / "scheduler_output_runtime.inc",
        )
    }
    (a.output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("inputs", "commands")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

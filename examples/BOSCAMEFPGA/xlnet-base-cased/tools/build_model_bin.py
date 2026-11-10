#!/usr/bin/env python3
"""Link the real 12-layer XLNet NH/E6 graph, typed adapters and Triton objects.

Follows ViT's launcher/cache brackets/runtime/link workflow. This builds an
artifact only; numerical validation consumes scheduler-downloaded outputs.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
BUDDY = REPO / "buddy-mlir"
LLVM = BUDDY / "llvm/build/bin"
RUNTIME = BUDDY / "runtime/boscame"
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402 - local modules require the example path

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


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(command, logfile):
    result = subprocess.run(
        [str(x) for x in command],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logfile.write_text("$ " + " ".join(str(x) for x in command) + "\n" + result.stdout)
    if result.returncode:
        raise RuntimeError(str(logfile) + "\n" + result.stdout[-6000:])
    return result.stdout


def generate(weights, inputs, output, lm_head=False):
    records = json.loads((weights / "manifest.json").read_text())["files"]
    blob = bytearray()
    entries = {}
    for record in records:
        path = weights / (record["name"] + ".bin")
        if digest(path) != record["sha256"]:
            raise ValueError("weight SHA mismatch: " + str(path))
        blob += bytes((-len(blob)) % 64)
        entry = dict(record, offset=len(blob), nbytes=path.stat().st_size)
        entries[record["name"]] = entry
        blob += path.read_bytes()
    blob += bytes((-len(blob)) % 64)
    (output / "params.bin").write_bytes(blob)

    def pointer(name, dtype="float"):
        return f"({dtype}*)(xlnet_params+{entries[name]['offset']}u)"

    def matrix(name):
        return (
            "{" + pointer(name + ".q", "int8_t") + "," + pointer(name + ".scale") + "}"
        )

    lines = [
        "/* Generated from SHA-verified original checkpoint-derived tensors. */",
        '#include "model.h"',
        "extern unsigned char xlnet_params[],xlnet_input[];",
        "static const XLNetWeights xlnet_weights={",
        ".embedding=" + pointer("word_embedding") + ",",
        ".relative_positions=" + pointer("relative_positions") + ",",
        ".layers={",
    ]
    for i in range(12):
        base = f"layer{i}."
        lines.append("{")
        for name in ["q", "k", "v", "r", "o"]:
            lines.append("." + name + "=" + matrix(base + name) + ",")
        lines.append(
            ".segment={" + ",".join(matrix(base + f"seg{h}") for h in range(12)) + "},"
        )
        for field, name in [("ff1", "ffn1"), ("ff2", "ffn2")]:
            lines.append("." + field + "=" + matrix(base + name) + ",")
        for field, name in [
            ("rw_bias", "r_w_bias"),
            ("rr_bias", "r_r_bias"),
            ("rs_bias", "r_s_bias"),
            ("attn_norm_weight", "attn_norm.weight"),
            ("attn_norm_bias", "attn_norm.bias"),
            ("ffn_norm_weight", "ffn_norm.weight"),
            ("ffn_norm_bias", "ffn_norm.bias"),
            ("ff1_bias", "ffn1.bias"),
            ("ff2_bias", "ffn2.bias"),
        ]:
            lines.append("." + field + "=" + pointer(base + name) + ",")
        lines.append("},")
    lines += ["},"]
    if lm_head:
        lines += [
            ".lm_tiles={" + ",".join(matrix(f"lm{i}") for i in range(32)) + "},",
            ".lm_bias={" + ",".join(pointer(f"lm{i}.bias") for i in range(32)) + "},",
        ]
    lines += [
        "};",
        f"#define XLNET_MODEL_LM_HEAD {int(lm_head)}",
        f"#define VIT_BOARD_EXPECTED_OUTPUTS {1 if lm_head else 13}",
        f"#define VIT_BOARD_EXPORT_BYTES {16384000 if lm_head else 5111808}u",
    ]
    names = (
        ["logits"]
        if lm_head
        else ["embedding"]
        + [f"layer_{i:02d}" for i in range(1, 12)]
        + ["last_hidden_state"]
    )
    width = 32000 if lm_head else 768
    output_bytes = 128 * width * 4
    rows = [
        f'{{"{name}","<f4",3,{{1,128,{width},0}},{output_bytes},{i * output_bytes}}}'
        for i, name in enumerate(names)
    ]
    lines.append("#define VIT_BOARD_OUTPUT_TABLE {" + ",".join(rows) + "}")
    (output / "model_artifacts.h").write_text("\n".join(lines) + "\n")
    input_manifest = json.loads((inputs / "manifest.json").read_text())
    input_records = {r["name"]: r for r in input_manifest["files"]}
    input_blob = bytearray()
    for name in ["input_ids", "attention_mask", "token_type_ids"]:
        path = inputs / (name + ".bin")
        if digest(path) != input_records[name]["sha256"] or path.stat().st_size != 512:
            raise ValueError("input SHA/size mismatch: " + str(path))
        input_blob += path.read_bytes()
    (output / "input.bin").write_bytes(input_blob)
    assembly = []
    for section, symbol, path in [
        (".model_params", "xlnet_params", output / "params.bin"),
        (".model_input", "xlnet_input", output / "input.bin"),
    ]:
        assembly += [
            f'.section {section},"a",@progbits',
            ".balign 64",
            f".globl {symbol}",
            symbol + ":",
            '.incbin "' + str(path) + '"',
        ]
    assembly += [
        '.section .model_workspace,"aw",@nobits',
        ".balign 64",
        ".globl xlnet_workspace",
        "xlnet_workspace:",
        f".space {25559040 + (16384000 if lm_head else 0)}",
    ]
    (output / "model_blobs.S").write_text("\n".join(assembly) + "\n")
    (output / "params-layout.json").write_text(
        json.dumps(
            {
                "entries": list(entries.values()),
                "nbytes": len(blob),
                "sha256": digest(output / "params.bin"),
            },
            indent=2,
        )
        + "\n"
    )
    (output / "output-layout.json").write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "shape": [1, 128, width],
                    "dtype": "<f4",
                    "nbytes": output_bytes,
                    "offset": i * output_bytes,
                }
                for i, name in enumerate(names)
            ],
            indent=2,
        )
        + "\n"
    )


def wrappers(output):
    source = (ROOT / "model_runtime/model.c").read_text()
    entries = re.findall(r"extern void (_mlir_ciface_kernel_\w+)\(([^)]*)\);", source)
    lines = [
        '#include "support.h"',
        '#include "nh_e6_platform.h"',
        "extern volatile uint64_t vit_kernel_case,vit_kernel_calls;",
    ]
    symbols = []
    for ordinal, (symbol, signature) in enumerate(entries, 1):
        ranks = re.findall(r"MemRef(\d+)", signature)
        params = ",".join(f"MemRef{r} *a{i}" for i, r in enumerate(ranks))
        args = ",".join(f"a{i}" for i in range(len(ranks)))
        symbols.append(symbol)
        lines += [
            f"extern void __real_{symbol}({params});",
            f"void __wrap_{symbol}({params}){{",
            f"vit_kernel_case={ordinal};++vit_kernel_calls;",
            "nh_e6_cache_writes();",
            f"__real_{symbol}({args});",
            '__asm__ volatile("fence iorw, iorw" ::: "memory");',
            "nh_e6_cache_writes();}",
        ]
    (output / "model_kernel_cache_wrappers.c").write_text("\n".join(lines) + "\n")
    (output / "kernel-case-map.json").write_text(
        json.dumps({str(i): symbol for i, symbol in enumerate(symbols, 1)}, indent=2)
        + "\n"
    )
    return symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "build/weights")
    parser.add_argument("--model-input", type=Path, default=ROOT / "build/model-input")
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--output", type=Path, default=ROOT / "build/model-bin")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--lm-head",
        action="store_true",
        help="include the checkpoint's complete 32000-way LM output",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    generate(
        args.weights.resolve(), args.model_input.resolve(), args.output, args.lm_head
    )
    symbols = wrappers(args.output)
    includes = [ROOT, ROOT / "model_runtime", args.output, ROOT.parent / "tools"]
    includes += [RUNTIME / p for p in ["nh_e6", "runtime", "uart", "nr"]]
    flags = CFLAGS + ["-I" + str(p) for p in includes]
    objects = []
    jobs = []

    def add(source, name, extra=()):
        target = args.output / (name + ".o")
        objects.append(target)
        jobs.append(
            (
                [LLVM / "clang", *flags, *extra, "-c", source, "-o", target],
                args.output / ("compile-" + name + ".log"),
            )
        )

    add(ROOT / "model_runtime/model.c", "model_graph")
    add(ROOT / "model_runtime/board_support.c", "model_support")
    add(args.output / "model_kernel_cache_wrappers.c", "cache_wrappers")
    add(args.output / "model_blobs.S", "blobs")
    add(ROOT / "model_runtime/model_trap.S", "model_trap")
    add(RUNTIME / "nh_e6/nh_e6_boot.S", "boot")
    add(
        RUNTIME / "nh_e6/nh_e6_platform.c",
        "platform",
        ["-Dinit_uart=board_init_uart", "-DNH_E6_WAIT_ITERATIONS=0xffffffffu"],
    )
    add(RUNTIME / "uart/uart.c", "uart")
    add(
        RUNTIME / "runtime/bosc_ame_dma.c",
        "dma",
        ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"],
    )
    add(RUNTIME / "nr/nr_math.c", "math")
    kernel_evidence = []
    for case in inventory():
        name = case["name"]
        directory = args.kernels.resolve() / name
        frontend = json.loads((directory / "frontend.json").read_text())
        result = json.loads((directory / "build-result.json").read_text())
        if result.get("status") != "PASS" or result.get("stage") != "asm":
            raise ValueError("case has no successful current assembly build: " + name)
        if digest(ROOT / name / "kernel.py") != frontend["kernel_module_sha256"]:
            raise ValueError("kernel source changed since AOT: " + name)
        if digest(directory / "adapter.c") != frontend["adapter_sha256"]:
            raise ValueError("typed adapter changed since AOT: " + name)
        for artifact in ["kernel.o", "kernel.ll"]:
            if digest(directory / artifact) != result["artifacts"][artifact]:
                raise ValueError(
                    "AOT artifact checksum mismatch: " + name + "/" + artifact
                )
        add(directory / "adapter.c", "adapter_" + name)
        objects.append(directory / "kernel.o")
        kernel_evidence.append(
            {
                "case": name,
                "kernel_source_sha256": digest(ROOT / name / "kernel.py"),
                "scalar_llvm_sha256": digest(directory / "kernel.ll"),
                "kernel_sha256": digest(directory / "kernel.o"),
                "adapter_sha256": digest(directory / "adapter.c"),
            }
        )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda job: run(*job), jobs))
    early = ["boot.o", "platform.o", "uart.o", "dma.o", "math.o"]
    objects.sort(
        key=lambda p: (
            p.name not in early,
            early.index(p.name) if p.name in early else 0,
        )
    )
    linker = os.environ.get("LD_LLD") or shutil.which("ld.lld")
    elf = args.output / "xlnet_model.elf"
    binary = args.output / "xlnet_model.bin"
    export_base = "0x80040000" if args.lm_head else "0x80100000"
    run(
        [
            linker,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "--defsym=__model_export_base=" + export_base,
            "-T",
            ROOT / "board_model.ld",
            "-Map=" + str(args.output / "xlnet_model.map"),
            "-o",
            elf,
            *("--wrap=" + symbol for symbol in symbols),
            *objects,
        ],
        args.output / "link.log",
    )
    run(
        [LLVM / "llvm-objcopy", "-O", "binary", elf, binary],
        args.output / "objcopy.log",
    )
    undefined = run(
        [LLVM / "llvm-nm", "-u", elf], args.output / "undefined.log"
    ).strip()
    if undefined:
        raise RuntimeError("undefined symbols: " + undefined)
    run(
        [
            sys.executable,
            BUDDY / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            elf,
            "--objdump",
            LLVM / "llvm-objdump",
            "--output",
            args.output / "elf-audit.json",
        ],
        args.output / "audit.log",
    )
    manifest = {
        "schema": 1,
        "model": "xlnet/xlnet-base-cased",
        "layers": 12,
        "mode": "real whole-model NH/E6; fixed batch1/seq128; W8A8; FP32 embedding",
        "output": "XLNetLMHeadModel.logits[1,128,32000]"
        if args.lm_head
        else "official XLNetModel last_hidden_state and all 12 layer checkpoints",
        "lm_head": args.lm_head,
        "numerical_validation": "pending scheduler download and local check",
        "bin_bytes": binary.stat().st_size,
        "bin_sha256": digest(binary),
        "elf_sha256": digest(elf),
        "params_sha256": digest(args.output / "params.bin"),
        "input_sha256": digest(args.output / "input.bin"),
        "undefined_symbols": [],
        "audit": json.loads((args.output / "elf-audit.json").read_text()),
        "kernels": kernel_evidence,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / name
                for name in [
                    "board_model.ld",
                    "support.h",
                    "cases.py",
                    "model_runtime/model.c",
                    "model_runtime/model.h",
                    "model_runtime/board_support.c",
                    "model_runtime/model_trap.S",
                    "tools/build_model_bin.py",
                    "tools/build_model_host.py",
                    "tools/prepare_weights.py",
                    "tools/prepare_model_input.py",
                    "tools/model_reference.py",
                    "tools/quantized_model_reference.py",
                    "tools/check_model_weights.py",
                ]
            ]
            + [ROOT / c["name"] / "kernel.py" for c in inventory()]
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in ["kernels", "audit", "source_sha256"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

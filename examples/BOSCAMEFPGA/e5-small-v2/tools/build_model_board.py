#!/usr/bin/env python3
"""Pack the real E5 checkpoint and link every real operator into an NH/E6 image."""

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
BUDDY = (
    Path(os.environ.get("BUDDY_ROOT", str(REPO / "buddy-mlir"))).expanduser().resolve()
)
LLVM = (
    Path(os.environ.get("LLVM_BIN", str(BUDDY / "llvm/build/bin")))
    .expanduser()
    .resolve()
)
RUNTIME = BUDDY / "runtime/boscame"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402 - local modules require the example path

FLAGS = [
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(command, log):
    command = list(map(str, command))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write_text("$ " + " ".join(command) + "\n" + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{log.name}: " + (result.stdout + result.stderr)[-4000:])


def prepare(args):
    import torch
    from check_model_runtime import sequence
    from model_reference import load_model
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    metadata = json.loads((args.assets / "manifest.json").read_text())
    for name, expected in metadata["sha256"].items():
        if sha(args.assets / name) != expected:
            raise ValueError("pinned asset SHA mismatch: " + name)
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True)
    prefixed = args.kind + ": " + args.text
    batch = tokenizer(
        prefixed,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model, config = load_model(args.assets)
    state = model.state_dict()
    ids = batch["input_ids"][0]
    mask = batch["attention_mask"][0].to(torch.int32)
    types = batch.get("token_type_ids", torch.zeros_like(batch["input_ids"]))[0]
    parameters = bytearray()
    entries = []
    pointers = {}

    def add(key, shape):
        value = state[key].detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        if tuple(value.shape) != tuple(shape):
            raise ValueError(key + " shape mismatch")
        offset = (len(parameters) + 63) & ~63
        parameters.extend(b"\0" * (offset - len(parameters)))
        raw = value.tobytes()
        parameters.extend(raw)
        pointers[key] = f"(const float *)(e5_params + {offset}u)"
        entries.append(
            {
                "key": key,
                "shape": list(shape),
                "dtype": "<f4",
                "offset": offset,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    add("embeddings.word_embeddings.weight", (30522, 384))
    add("embeddings.position_embeddings.weight", (512, 384))
    add("embeddings.token_type_embeddings.weight", (2, 384))
    for field in ("weight", "bias"):
        add("embeddings.LayerNorm." + field, (384,))
    for layer in range(12):
        prefix = f"encoder.layer.{layer}."
        for name, n, k in [
            ("attention.self.query", 384, 384),
            ("attention.self.key", 384, 384),
            ("attention.self.value", 384, 384),
            ("attention.output.dense", 384, 384),
            ("intermediate.dense", 1536, 384),
            ("output.dense", 384, 1536),
        ]:
            add(prefix + name + ".weight", (n, k))
            add(prefix + name + ".bias", (n,))
        for norm in ("attention.output.LayerNorm", "output.LayerNorm"):
            for field in ("weight", "bias"):
                add(prefix + norm + "." + field, (384,))
    parameters.extend(b"\0" * ((-len(parameters)) % 64))
    (args.output / "params.bin").write_bytes(parameters)
    save(
        args.output / "params.json",
        {
            "nbytes": len(parameters),
            "sha256": sha(args.output / "params.bin"),
            "entries": entries,
        },
    )
    input_bytes = b"".join(
        v.to(torch.int32).numpy().astype("<i4").tobytes()
        for v in (ids, mask, types, torch.arange(128))
    )
    (args.output / "inputs.bin").write_bytes(input_bytes)
    catalog = {case["name"]: case for case in inventory()}
    trace = []
    for index, name in enumerate(sequence()):
        scope = (
            "embedding"
            if index < 3
            else f"encoder.layer.{(index - 3) // 137}"
            if index < 1647
            else "sentence_embedding"
        )
        trace.append({"scope": scope, "case": name, "grid": catalog[name]["grid"]})
    diagnostics = {}
    if args.legacy_math_reference:
        import torch.nn.functional as F
        from model_reference import average_pool, forward

        with torch.inference_mode():
            fp32_layers = []
            quantized_layers = []
            fp32_hidden, fp32_sentence, _ = forward(
                state, config, ids, mask, types, checkpoints=fp32_layers
            )
            quantized_hidden, quantized_sentence, legacy_trace = forward(
                state, config, ids, mask, types, w8a8=True, checkpoints=quantized_layers
            )
            hf_hidden = model(
                input_ids=ids[None],
                attention_mask=mask[None],
                token_type_ids=types[None],
            ).last_hidden_state[0]
            hf_sentence = F.normalize(average_pool(hf_hidden, mask), dim=0)
            torch.testing.assert_close(fp32_hidden, hf_hidden, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(fp32_sentence, hf_sentence, atol=2e-6, rtol=2e-5)
        if [item["case"] for item in legacy_trace] != [item["case"] for item in trace]:
            raise ValueError("diagnostic execution differs from static model schedule")
        for name, value in [
            ("fp32_last_hidden", fp32_hidden),
            ("fp32_sentence_embedding", fp32_sentence),
            ("w8a8_last_hidden", quantized_hidden),
            ("w8a8_sentence_embedding", quantized_sentence),
            ("fp32_layer_hidden", torch.stack(fp32_layers)),
            ("w8a8_layer_hidden", torch.stack(quantized_layers)),
        ]:
            array = value.detach().numpy().astype("<f4")
            (args.output / (name + ".bin")).write_bytes(array.tobytes())
        diagnostics = {
            "legacy_math_reference": True,
            "fp32_decomposition_max_abs_error_to_hf": float(
                (fp32_hidden - hf_hidden).abs().max()
            ),
            "w8a8_sentence_cosine_to_fp32": float(
                F.cosine_similarity(fp32_sentence, quantized_sentence, dim=0)
            ),
        }
    if len(trace) != 1649:
        raise ValueError("complete model trace is not 1649 calls")
    save(args.output / "operator_trace.json", trace)
    save(
        args.output / "input_reference.json",
        {
            "repository": metadata["repository"],
            "revision": metadata["revision"],
            "checkpoint_sha256": metadata["sha256"]["model.safetensors"],
            "tokenizer_sha256": metadata["sha256"]["tokenizer.json"],
            "text": args.text,
            "kind": args.kind,
            "prefixed_text": prefixed,
            "valid_tokens": int(mask.sum()),
            "input_ids": ids.tolist(),
            "attention_mask": mask.tolist(),
            "token_type_ids": types.tolist(),
            "input_sha256": sha(args.output / "inputs.bin"),
            "operator_calls": len(trace),
            "reference_scope": "checkpoint/tokenizer packing and static dispatch schedule; numerical acceptance requires a separately executed emitted reference",
            **diagnostics,
        },
    )

    def linear(prefix):
        return (
            "{" + pointers[prefix + ".weight"] + ", " + pointers[prefix + ".bias"] + "}"
        )

    lines = [
        '#include "model_board.h"',
        '#include "model_runtime/model.h"',
        "extern const unsigned char e5_params[], e5_inputs[];",
        'static unsigned char model_workspace[7479296] __attribute__((section(".model_workspace"),aligned(64),used));',
        "static const E5Weights weights = {",
        " .word_embeddings=" + pointers["embeddings.word_embeddings.weight"] + ",",
        " .position_embeddings="
        + pointers["embeddings.position_embeddings.weight"]
        + ",",
        " .type_embeddings="
        + pointers["embeddings.token_type_embeddings.weight"]
        + ",",
        " .embedding_norm=" + linear("embeddings.LayerNorm") + ",",
        " .layers={",
    ]
    for layer in range(12):
        prefix = f"encoder.layer.{layer}."
        fields = {
            name: linear(prefix + key)
            for name, key in [
                ("query", "attention.self.query"),
                ("key", "attention.self.key"),
                ("value", "attention.self.value"),
                ("attention_output", "attention.output.dense"),
                ("intermediate", "intermediate.dense"),
                ("output", "output.dense"),
                ("attention_norm", "attention.output.LayerNorm"),
                ("output_norm", "output.LayerNorm"),
            ]
        }
        lines.append(
            "  {"
            + ", ".join("." + key + "=" + value for key, value in fields.items())
            + "},"
        )
    lines += [
        " }};",
        "int e5_model_launch(void){",
        " if(e5_workspace_bytes()!=sizeof(model_workspace))return -2;",
        " return e5_infer(&weights,(const int32_t *)(e5_inputs+0),(const int32_t *)(e5_inputs+512),",
        "  (const int32_t *)(e5_inputs+1024),(const int32_t *)(e5_inputs+1536),",
        "  model_workspace,sizeof(model_workspace),e5_sentence_output,e5_hidden_output);",
        "}",
        "",
    ]
    (args.output / "model_launch.c").write_text("\n".join(lines))
    asm = []
    for section, symbol, name in [
        (".model_params", "e5_params", "params.bin"),
        (".model_input", "e5_inputs", "inputs.bin"),
    ]:
        asm += [
            f'.section {section},"a",@progbits',
            ".balign 64",
            f".globl {symbol}",
            symbol + ":",
            f'.incbin "{args.output / name}"',
        ]
    (args.output / "model_blobs.S").write_text("\n".join(asm) + "\n")
    print(
        "prepared real checkpoint:",
        len(parameters),
        "bytes;",
        len(trace),
        "kernel calls;",
        int(mask.sum()),
        "valid tokens",
        flush=True,
    )


def build(args):
    cases = inventory()
    objects = []
    includes = [
        ROOT,
        ROOT / "tools",
        args.output,
        RUNTIME / "nh_e6",
        RUNTIME / "runtime",
        RUNTIME / "uart",
        RUNTIME / "nr",
    ]
    flags = FLAGS + [item for p in includes for item in ("-I", str(p))]

    def compile(source, name, extra=()):
        dest = args.output / (name + ".o")
        run(
            [LLVM / "clang", *flags, *extra, "-c", source, "-o", dest],
            args.output / (name + ".log"),
        )
        objects.append(dest)

    names = (
        "static const char *const e5_model_case_names[]={"
        + ",".join(json.dumps(c["name"]) for c in cases)
        + "};\n"
    )
    (args.output / "model_case_names.h").write_text(names)
    compile(RUNTIME / "nh_e6/nh_e6_boot.S", "boot")
    compile(
        RUNTIME / "nh_e6/nh_e6_platform.c",
        "platform",
        [
            "-Dinit_uart=board_init_uart",
            "-DNH_E6_WAIT_ITERATIONS=0xffffffffu",
            "-DNH_E6_MMIO_READ32=e5_model_mmio_read32",
            "-include",
            str(ROOT / "tools/model_board.h"),
        ],
    )
    compile(RUNTIME / "uart/uart.c", "uart")
    compile(
        RUNTIME / "runtime/bosc_ame_dma.c",
        "dma",
        ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"],
    )
    compile(RUNTIME / "nr/nr_math.c", "math")
    compile(ROOT / "tools/model_trap.S", "trap")
    compile(ROOT / "tools/model_board.c", "model_board")
    compile(
        ROOT / "model_runtime/model.c", "model_graph", ["-DE5_TARGET_CHECKPOINTS=1"]
    )
    compile(args.output / "model_launch.c", "model_launch")
    compile(args.output / "model_blobs.S", "model_blobs")
    wrappers = ['#include "model_board.h"']
    kernel_hashes = {}
    for ordinal, case in enumerate(cases):
        directory = args.kernels / case["name"]
        raw, adapter = directory / "kernel.o", directory / "adapter.c"
        if not raw.is_file() or not adapter.is_file():
            raise ValueError("missing real built kernel " + case["name"])
        kernel_hashes[case["name"]] = sha(raw)
        arguments = case["arguments"]
        params = ", ".join(f"MemRef{a['rank']} *a{i}" for i, a in enumerate(arguments))
        call = ", ".join("a" + str(i) for i in range(len(arguments)))
        name = case["name"]
        wrappers += [
            f"extern void e5_impl_{name}({params});",
            f"void _mlir_ciface_kernel_{name}({params})" + " {",
            f" uint64_t begin=e5_model_kernel_begin({ordinal});",
            f" e5_impl_{name}({call});",
            f" e5_model_kernel_end({ordinal},begin);",
            "}",
        ]
        compile(
            adapter,
            "adapter_" + name,
            ["-D_mlir_ciface_kernel_" + name + "=e5_impl_" + name],
        )
        objects.append(raw)
    (args.output / "model_wrappers.c").write_text("\n".join(wrappers) + "\n")
    compile(args.output / "model_wrappers.c", "model_wrappers")
    elf = args.output / "e5_model.elf"
    binary = args.output / "e5_model.bin"
    linker = LLVM / "ld.lld"
    if not linker.exists():
        linker = Path(shutil.which("ld.lld"))
    run(
        [
            linker,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            ROOT / "board_model.ld",
            "-Map=" + str(args.output / "e5_model.map"),
            "-o",
            elf,
            *objects,
        ],
        args.output / "link.log",
    )
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
    run(
        [LLVM / "llvm-objcopy", "-O", "binary", elf, binary],
        args.output / "objcopy.log",
    )
    sources = {
        str(p.relative_to(REPO)): sha(p)
        for p in [
            ROOT / "board_model.ld",
            ROOT / "model_runtime/model.c",
            ROOT / "model_runtime/model.h",
            ROOT / "tools/model_board.c",
            ROOT / "tools/model_board.h",
            ROOT / "tools/model_trap.S",
            Path(__file__),
            ROOT / "tools/model_reference.py",
        ]
    }
    save(
        args.output / "build.json",
        {
            "status": "BUILT_AUDITED",
            "model": "intfloat/e5-small-v2",
            "image_sha256": sha(binary),
            "image_bytes": binary.stat().st_size,
            "elf_sha256": sha(elf),
            "real_kernel_objects": kernel_hashes,
            "sources": sources,
            "objects": [{"path": str(p), "sha256": sha(p)} for p in objects],
            "operator_calls": 1649,
            "checkpoint_layers": 13,
            "outputs": [
                {
                    "name": "sentence_embedding",
                    "dtype": "<f4",
                    "shape": [384],
                    "nbytes": 1536,
                },
                {
                    "name": "last_hidden",
                    "dtype": "<f4",
                    "shape": [128, 384],
                    "nbytes": 196608,
                },
                {
                    "name": "layer_hidden",
                    "dtype": "<f4",
                    "shape": [13, 128, 384],
                    "nbytes": 2555904,
                },
            ],
            "board_executed": False,
        },
    )
    print("BUILT_AUDITED", binary, binary.stat().st_size, "bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--output", type=Path, default=ROOT / "build/model-board")
    parser.add_argument("--text", default="how much protein should a female eat")
    parser.add_argument("--kind", choices=("query", "passage"), default="query")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument(
        "--legacy-math-reference",
        action="store_true",
        help="explicitly generate historical mathematical diagnostics; never used for formal acceptance",
    )
    args = parser.parse_args()
    args.assets = args.assets.resolve()
    args.kernels = args.kernels.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.reuse_prepared:
        prepare(args)
    build(args)


if __name__ == "__main__":
    main()

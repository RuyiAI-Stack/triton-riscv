#!/usr/bin/env python3
"""Build a real 12-layer GTE checkpoint NH/E6 image using all 30 kernels.

The C graph is model_runtime/model.c. Like ViT's build_model_bin.py this packs
real FP32 parameters/input, compiles typed cache brackets and links the actual
compiled BOSCAME objects with the NH/E6 runtime. It runs no board oracle.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from cases import inventory  # noqa: E402 - local modules require the example path
from check_model_runtime import sequence  # noqa: E402 - local modules require the example path

LLVM = REPO / "buddy-mlir/llvm/build/bin"
RUNTIME = REPO / "buddy-mlir/runtime/boscame"
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
TEXT = "The capital of France is Paris. This sentence is encoded with the official GTE tokenizer to validate a complete twelve layer sentence embedding model."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run(command, log):
    command = list(map(str, command))
    p = subprocess.run(command, capture_output=True, text=True, check=False)
    Path(log).write_text("$ " + " ".join(command) + "\n" + p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(
            "command failed: " + " ".join(command) + "\n" + p.stderr[-4000:]
        )
    return p.stdout


def prepare(output, assets, text):
    manifest = json.loads((assets / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha(assets / name) != digest:
            raise ValueError("Pinned checkpoint/tokenizer hash mismatch: " + name)
    config = json.loads((assets / "config.json").read_text())
    expected = {
        "hidden_size": 384,
        "num_attention_heads": 12,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint config differs from fixed operator specialization")
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    encoded = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="np"
    )
    ids = encoded["input_ids"][0].astype("<i4")
    mask = encoded["attention_mask"][0].astype("<i4")
    types = encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"]))[
        0
    ].astype("<i4")
    positions = np.arange(128, dtype="<i4")
    state = load_file(assets / "model.safetensors")
    offsets = {}
    blob = bytearray()
    # Preserve the original packed physical layout rather than safetensors key order.
    shapes = {
        "embeddings.word_embeddings.weight": (30522, 384),
        "embeddings.position_embeddings.weight": (512, 384),
        "embeddings.token_type_embeddings.weight": (2, 384),
        "embeddings.LayerNorm.weight": (384,),
        "embeddings.LayerNorm.bias": (384,),
    }
    for layer in range(12):
        for module, weight_shape in (
            ("attention.self.query", (384, 384)),
            ("attention.self.key", (384, 384)),
            ("attention.self.value", (384, 384)),
            ("attention.output.dense", (384, 384)),
            ("attention.output.LayerNorm", (384,)),
            ("intermediate.dense", (1536, 384)),
            ("output.dense", (384, 1536)),
            ("output.LayerNorm", (384,)),
        ):
            prefix = f"encoder.layer.{layer}.{module}"
            shapes[prefix + ".weight"] = weight_shape
            shapes[prefix + ".bias"] = weight_shape[:1]
    unused = {"pooler.dense.weight", "pooler.dense.bias", "embeddings.position_ids"}
    if set(state) - unused != set(shapes):
        raise ValueError("checkpoint parameter names differ from fixed specialization")
    for name, shape in shapes.items():
        while len(blob) % 64:
            blob.append(0)
        array = state[name].astype("<f4")
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("checkpoint tensor shape/value mismatch: " + name)
        offsets[name] = {
            "offset": len(blob),
            "shape": list(array.shape),
            "nbytes": array.nbytes,
        }
        blob.extend(array.tobytes())
    (output / "params.bin").write_bytes(blob)
    arrays = {
        "input_ids": ids,
        "attention_mask": mask,
        "token_type_ids": types,
        "position_ids": positions,
    }
    input_bytes = b"".join(a.astype("<i4").tobytes() for a in arrays.values())
    (output / "input.bin").write_bytes(input_bytes)
    # Inputs only: numerical reference BINs come from build_model_host.py.
    np.savez(output / "reference.npz", **arrays)
    trace = [{"call": index, "case": name} for index, name in enumerate(sequence())]
    save(output / "operator_trace.json", trace)
    assert len(trace) == 1649
    contract = {
        "model": "thenlper/gte-small",
        "revision": manifest["revision"],
        "text": text,
        "tokenizer": "pinned official tokenizer; no prefix; max_length/padding=128",
        "valid_tokens": int(mask.sum()),
        "inputs": {k: v.tolist() for k, v in arrays.items()},
        "checkpoint_manifest": manifest,
        "params_bytes": len(blob),
        "params_sha256": sha(output / "params.bin"),
        "input_sha256": sha(output / "input.bin"),
        "parameter_layout": offsets,
        "operator_calls": 1649,
        "semantics": "12 layers, eager bidirectional BERT attention, valid-token mean (including CLS/SEP), L2 normalization",
        "validation": "byte-for-byte comparison against local emitted-code outputs",
        "reference_sha256": sha(output / "reference.npz"),
    }
    save(output / "model-input.json", contract)
    print(
        "INPUT",
        int(mask.sum()),
        "valid tokens;",
        len(blob),
        "FP32 parameter bytes; 1649-call routing ready",
        flush=True,
    )
    return offsets, contract


def launcher(output, offsets):
    def ptr(name):
        return f"(const float *)(gte_parameters+{offsets[name]['offset']}u)"

    def affine(name):
        return "{" + ptr(name + ".weight") + "," + ptr(name + ".bias") + "}"

    lines = [
        '#include "model_runtime/model.h"',
        '#include "nh_e6_platform.h"',
        "extern const unsigned char gte_parameters[],gte_inputs[];",
        'static unsigned char model_workspace[7479296] __attribute__((section(".model_workspace"),aligned(64),used));',
        'float model_embedding[384] __attribute__((section(".model_exports"),aligned(64),used));',
        'float model_hidden[128*384] __attribute__((section(".model_exports"),aligned(64),used));',
        "int model_launch(void){",
        " GTEWeights w={0};",
        " w.word_embeddings=" + ptr("embeddings.word_embeddings.weight") + ";",
        " w.position_embeddings=" + ptr("embeddings.position_embeddings.weight") + ";",
        " w.type_embeddings=" + ptr("embeddings.token_type_embeddings.weight") + ";",
        " w.embedding_norm=(GTELayerNorm)" + affine("embeddings.LayerNorm") + ";",
    ]
    mapping = {
        "query": "attention.self.query",
        "key": "attention.self.key",
        "value": "attention.self.value",
        "attention_output": "attention.output.dense",
        "intermediate": "intermediate.dense",
        "output": "output.dense",
        "attention_norm": "attention.output.LayerNorm",
        "output_norm": "output.LayerNorm",
    }
    for layer in range(12):
        for field, suffix in mapping.items():
            typ = "GTELayerNorm" if field.endswith("_norm") else "GTELinear"
            lines.append(
                f" w.layers[{layer}].{field}=({typ})"
                + affine(f"encoder.layer.{layer}.{suffix}")
                + ";"
            )
    lines += [
        " return gte_infer(&w,(const int32_t *)gte_inputs,(const int32_t *)(gte_inputs+512),",
        "  (const int32_t *)(gte_inputs+1024),(const int32_t *)(gte_inputs+1536),",
        "  model_workspace,sizeof(model_workspace),model_embedding,model_hidden);",
        "}",
    ]
    (output / "model_launcher.c").write_text("\n".join(lines) + "\n")
    (output / "model_blobs.S").write_text(
        f'.section .model_params,"a",@progbits\n.balign 64\n.global gte_parameters\ngte_parameters:\n.incbin "{output / "params.bin"}"\n.section .model_input,"a",@progbits\n.balign 64\n.global gte_inputs\ngte_inputs:\n.incbin "{output / "input.bin"}"\n'
    )


def trace_layout(output, cases, layer):
    by_name = {c["name"]: c for c in cases}
    items = []
    offset = 0
    for call, name in enumerate(sequence()):
        if not (3 + 137 * layer <= call < 3 + 137 * (layer + 1)):
            continue
        c = by_name[name]
        for index in c["outputs"]:
            offset = (offset + 63) & ~63
            argument = c["arguments"][index]
            shape = c["argument_shapes"][index]
            nbytes = (
                int(np.prod(shape)) * {"f32": 4, "i32": 4, "i8": 1}[argument["dtype"]]
            )
            items.append(
                {
                    "call": call,
                    "case": name,
                    "argument": index,
                    "offset": offset,
                    "nbytes": nbytes,
                    "shape": shape,
                    "dtype": {"f32": "<f4", "i32": "<i4", "i8": "|i1"}[
                        argument["dtype"]
                    ],
                }
            )
            offset += nbytes
    layout = {
        "layer": layer,
        "entries": items,
        "nbytes": (offset + 63) & ~63,
        "scope": "raw output captures, no board numerical oracle",
    }
    save(output / "trace-layout.json", layout)
    return layout


def wrappers(output, cases, layers=12, trace=None):
    ids = {case["name"]: i + 1 for i, case in enumerate(cases)}
    calls = sequence()[: 3 + 137 * layers] + sequence()[-2:]
    lines = [
        '#include "support.h"',
        '#include "nh_e6_platform.h"',
        "extern volatile uint64_t model_calls,model_active_case,model_kernel_cycles;",
        "static const unsigned short expected[]={"
        + ",".join(str(ids[n]) for n in calls)
        + "};",
    ]
    for case in cases:
        name = case["name"]
        params = ",".join(
            f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"])
        )
        args = ",".join(f"a{i}" for i in range(len(case["arguments"])))
        symbol = "_mlir_ciface_kernel_" + name
        lines.extend(
            [
                f"extern void __real_{symbol}({params});",
                f"void __wrap_{symbol}({params})" + "{",
                f" if(model_calls>={3 + 137 * layers + 2} || expected[model_calls]!={ids[name]})__builtin_trap();",
                f" model_active_case={ids[name]};",
                " if((model_calls&63)==0)nh_e6_progress(0x100000u+(unsigned)model_calls);",
                " nh_e6_cache_writes();uint64_t begin=nr_cycles();",
                f" __real_{symbol}({args});",
                ' __asm__ volatile("fence iorw,iorw":::"memory");',
                " nh_e6_cache_writes();model_kernel_cycles+=nr_cycles()-begin;",
            ]
        )
        if trace:
            lines.append(" extern unsigned char model_operator_trace[];")
            for item in trace["entries"]:
                if item["case"] == name:
                    lines.append(
                        f" if(model_calls=={item['call']})nr_copy_bytes(model_operator_trace+{item['offset']}u,a{item['argument']}->aligned,{item['nbytes']}u);"
                    )
            lines.append(" nh_e6_cache_writes();")
        lines.extend([" ++model_calls;", "}"])
    (output / "kernel_wrappers.c").write_text("\n".join(lines) + "\n")
    return ["--wrap=_mlir_ciface_kernel_" + c["name"] for c in cases]


def build(output, kernels, layers=12, trace=None):
    cases = inventory()
    objects = []
    includes = [
        ROOT,
        ROOT / "model_runtime",
        RUNTIME / "nh_e6",
        RUNTIME / "runtime",
        RUNTIME / "uart",
        RUNTIME / "nr",
    ]
    mode = (
        ["-DGTE_MODEL_CHECKPOINTS"]
        if trace is None
        else [
            f"-DGTE_MODEL_TRACE_BYTES={trace['nbytes']}",
            f'-DGTE_MODEL_TRACE_SHAPE="{trace["nbytes"] // 4}"',
        ]
    )
    flags = (
        FLAGS
        + [f"-DGTE_MODEL_LAYERS={layers}"]
        + mode
        + [item for path in includes for item in ("-I", str(path))]
    )
    sources = [
        (output / "model_launcher.c", "launcher"),
        (output / "model_blobs.S", "blobs"),
        (ROOT / "model_runtime/model.c", "graph"),
        (ROOT / "model_runtime/board_support.c", "support"),
        (output / "kernel_wrappers.c", "wrappers"),
        (RUNTIME / "nh_e6/nh_e6_boot.S", "boot"),
        (RUNTIME / "nh_e6/nh_e6_platform.c", "platform"),
        (RUNTIME / "runtime/bosc_ame_dma.c", "dma"),
        (RUNTIME / "uart/uart.c", "uart"),
        (RUNTIME / "nr/nr_math.c", "math"),
    ]
    for source, name in sources:
        obj = output / (name + ".o")
        extra = []
        if name == "platform":
            extra = [
                "-Dinit_uart=board_init_uart",
                "-DNH_E6_WAIT_ITERATIONS=0xffffffffu",
            ]
        if name == "dma":
            extra = ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"]
        run(
            [LLVM / "clang", *flags, *extra, "-c", source, "-o", obj],
            output / (name + ".compile.log"),
        )
        objects.append(obj)
    for case in cases:
        name = case["name"]
        source = kernels / name
        manifest = json.loads((source / "frontend.json").read_text())
        if manifest["kernel_module_sha256"] != sha(
            ROOT / name / "kernel.py"
        ) or manifest["adapter_sha256"] != sha(source / "adapter.c"):
            raise ValueError("Stale compiled kernel/adapter " + name)
        obj = output / ("adapter_" + name + ".o")
        run(
            [LLVM / "clang", *flags, "-c", source / "adapter.c", "-o", obj],
            output / ("adapter_" + name + ".log"),
        )
        objects.extend([obj, source / "kernel.o"])
    wrap = ["--wrap=_mlir_ciface_kernel_" + c["name"] for c in cases]
    elf = output / "gte-small-full.elf"
    image = output / "gte-small-full.bin"
    run(
        [
            shutil.which("ld.lld"),
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            ROOT / "model_runtime/board_model.ld",
            "-Map=" + str(output / "gte-small-full.map"),
            *wrap,
            "-o",
            elf,
            *objects,
        ],
        output / "link.log",
    )
    run(
        [
            sys.executable,
            REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            elf,
            "--objdump",
            LLVM / "llvm-objdump",
            "--output",
            output / "elf-audit.json",
        ],
        output / "audit.log",
    )
    run([LLVM / "llvm-objcopy", "-O", "binary", elf, image], output / "objcopy.log")
    sources = list((ROOT / "model_runtime").glob("*")) + [Path(__file__)]
    report = {
        "status": "BUILT",
        "image_bytes": image.stat().st_size,
        "image_sha256": sha(image),
        "elf_sha256": sha(elf),
        "sources": {str(p.relative_to(REPO)): sha(p) for p in sources if p.is_file()},
        "compiled_kernels": {
            c["name"]: {
                "object_sha256": sha(kernels / c["name"] / "kernel.o"),
                "kernel_sha256": sha(ROOT / c["name"] / "kernel.py"),
                "adapter_sha256": sha(kernels / c["name"] / "adapter.c"),
            }
            for c in cases
        },
        "kernel_objects": 30,
        "expected_calls": 3 + 137 * layers + 2,
        "encoder_layers": layers,
        "real_checkpoint": True,
        "board_executed": False,
        "outputs": {
            "sentence_embedding": {"shape": [384], "nbytes": 1536},
            "last_hidden_state": {"shape": [128, 384], "nbytes": 196608},
            "layer_checkpoints": {"shape": [13, 128, 384], "nbytes": 2555904},
        },
    }
    if trace:
        report["outputs"] = {
            "operator_trace": {
                "shape": [trace["nbytes"] // 4],
                "nbytes": trace["nbytes"],
            }
        }
        report["trace_layer"] = trace["layer"]
        report["trace_layout_sha256"] = sha(output / "trace-layout.json")
    save(output / "build-report.json", report)
    print("BUILT", image, image.stat().st_size, "bytes", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument(
        "--kernels", type=Path, default=ROOT / "build/scheduler-output/kernels"
    )
    p.add_argument("--output", type=Path, default=ROOT / "build/full-model")
    p.add_argument("--text", default=TEXT)
    p.add_argument("--reuse-input", action="store_true")
    p.add_argument("--trace-layer", type=int)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.reuse_input:
        contract = json.loads((a.output / "model-input.json").read_text())
        offsets = contract["parameter_layout"]
        assert (
            sha(a.output / "params.bin") == contract["params_sha256"]
            and sha(a.output / "input.bin") == contract["input_sha256"]
        )
    else:
        offsets, contract = prepare(a.output, a.assets, a.text)
    trace = (
        trace_layout(a.output, inventory(), a.trace_layer)
        if a.trace_layer is not None
        else None
    )
    if trace:
        assert 0 <= a.trace_layer < 12 and trace["nbytes"] < 15 * 1024 * 1024
    layers = 12 if trace is None else a.trace_layer + 1
    launcher(a.output, offsets)
    wrappers(a.output, inventory(), layers, trace)
    build(a.output, a.kernels.resolve(), layers, trace)


if __name__ == "__main__":
    main()

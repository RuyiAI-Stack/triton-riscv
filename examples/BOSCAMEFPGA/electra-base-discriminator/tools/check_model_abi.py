#!/usr/bin/env python3
"""Execute actual model.c scheduling against descriptor-checking typed mocks.

No numerical output is computed: this checks argument shape, alignment,
nonaliasing, scratch capacity and the ordered complete 12-layer call trace.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402 - local modules require the example path
from export import argument_shapes  # noqa: E402 - local modules require the example path


def expected_calls():
    """Independent fixed schedule for the dynamic FP32-weight C entry.

    Describe the complete operator ABI without executing a numerical model or
    reading a prior build report. Every dense call quantizes both operands.
    """

    def dense(n, k):
        return [
            f"quantize_128x{k}",
            f"quantize_{n}x{k}",
            f"matmul_i8_128x{n}x{k}",
            f"dequantize_128x{n}",
            f"bias_add_128x{n}",
        ]

    calls = ["embedding_128x768", "layernorm_128x768"]
    for _ in range(12):
        # Each Q/K/V projection is packed before the next projection.
        calls += (dense(768, 768) + ["pack_heads_12x128x64"]) * 3
        calls += [
            "quantize_128x64",
            "quantize_128x64",
            "matmul_i8_128x128x64",
            "dequantize_128x128",
        ] * 12
        calls += ["masked_softmax_12x128x128"]
        calls += [
            "quantize_128x128",
            "quantize_transpose_128x64",
            "matmul_i8_128x64x128",
            "dequantize_128x64",
        ] * 12
        calls += ["unpack_heads_12x128x64"]
        calls += dense(768, 768) + ["add_128x768", "layernorm_128x768"]
        calls += dense(3072, 768) + ["gelu_128x3072"]
        calls += dense(768, 3072) + ["add_128x768", "layernorm_128x768"]
    calls += dense(768, 768) + ["gelu_128x768"] + dense(1, 768)
    return calls


def main():
    destination = ROOT / "build/model-abi"
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        '#include "model_runtime/model.h"',
        '#include "support.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "static void require(int ok) { if (!ok) abort(); }",
    ]
    for case in inventory():
        shapes = argument_shapes(case)
        args = case["arguments"]
        lines.append(
            "void _mlir_ciface_kernel_"
            + case["name"]
            + "("
            + ", ".join(f"MemRef{a['rank']} *a{i}" for i, a in enumerate(args))
            + ") {"
        )
        for i, (arg, shape) in enumerate(zip(args, shapes)):
            size = 1 if arg["dtype"] == "i8" else 4
            count = 1
            stride = 1
            lines.append(
                f"  require(a{i} && a{i}->offset==0 && a{i}->aligned && !((uintptr_t)a{i}->aligned&63));"
            )
            for axis in reversed(range(len(shape))):
                lines.append(
                    f"  require(a{i}->sizes[{axis}]=={shape[axis]} && a{i}->strides[{axis}]=={stride});"
                )
                stride *= shape[axis]
                count *= shape[axis]
            lines.append(
                f"  uintptr_t p{i}=(uintptr_t)a{i}->aligned; size_t bytes{i}={count * size};"
            )
            for j in range(i):
                lines.append(f"  require(p{i}+bytes{i}<=p{j} || p{j}+bytes{j}<=p{i});")
        lines += ['  puts("' + case["name"] + '");', "}"]
    lines.append(r"""
static void *allocate(size_t bytes) {
  void *p=aligned_alloc(64,(bytes+63)&~(size_t)63);
  require(p!=NULL); memset(p,0,bytes); return p;
}
int main(void) {
  ElectraWeights w={0};
  float *weight=allocate(3072u*768u*4u), *bias=allocate(3072u*4u);
  w.word_embeddings=allocate(30522u*768u*4u);
  w.position_embeddings=allocate(512u*768u*4u);
  w.token_type_embeddings=allocate(2u*768u*4u);
  w.embedding_norm=(ElectraNorm){weight,bias};
  for (int i=0;i<12;i++) {
    ElectraLayer *l=&w.layers[i];
    l->query=l->key=l->value=l->attention_output=l->intermediate=l->output=(ElectraLinear){weight,bias};
    l->attention_norm=l->output_norm=(ElectraNorm){weight,bias};
  }
  w.discriminator_dense=w.discriminator_prediction=(ElectraLinear){weight,bias};
  int32_t *ids=allocate(128*4),*types=allocate(128*4),*mask=allocate(128*4);
  for (int i=0;i<128;i++) { ids[i]=i; types[i]=i%2; mask[i]=(i<117); }
  float *logits=allocate(128*4); void *scratch=allocate(ELECTRA_MODEL_WORKSPACE_BYTES);
  require(electra_infer(&w,ids,types,mask,logits,scratch,ELECTRA_MODEL_WORKSPACE_BYTES)==0);
  require(electra_infer(&w,ids,types,mask,logits,scratch,1)==-1);
  ids[0]=30522;
  require(electra_infer(&w,ids,types,mask,logits,scratch,ELECTRA_MODEL_WORKSPACE_BYTES)==-2);
  return 0;
}
""")
    source = destination / "mock.c"
    source.write_text("\n".join(lines) + "\n")
    cc = shutil.which("cc") or "cc"
    command = [
        cc,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-I" + str(ROOT),
        str(ROOT / "model_runtime/model.c"),
        str(source),
        "-o",
        str(destination / "model-abi"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run(
        [str(destination / "model-abi")], check=True, capture_output=True, text=True
    )
    calls = result.stdout.splitlines()
    trace = expected_calls()
    if len(trace) != 1645:
        raise AssertionError("fixed dynamic-weight schedule must contain 1645 calls")
    if calls != trace:
        first = next(
            (i for i, (a, b) in enumerate(zip(calls, trace)) if a != b),
            min(len(calls), len(trace)),
        )
        raise AssertionError(f"ordered C/Python operator trace differs at {first}")
    report = {
        "status": "ABI_SMOKE_PASS",
        "validation": "actual model.c 12-layer scheduling with typed numerical mocks",
        "numeric_outputs_validated": False,
        "board_executed": False,
        "operator_calls": len(calls),
        "ordered_calls_match_static_schedule": True,
        "model_workspace_bytes": 16 * 1024 * 1024,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

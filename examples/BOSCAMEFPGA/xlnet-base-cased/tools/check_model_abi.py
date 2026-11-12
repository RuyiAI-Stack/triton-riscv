#!/usr/bin/env python3
"""Execute the C 12-layer graph with typed ABI mocks and verify routing/counts.

Mocks produce no numerical output. This checks the actual model.c orchestration.
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


def main():
    out = ROOT / "build/model-abi"
    out.mkdir(parents=True, exist_ok=True)
    cases = inventory()
    lines = [
        '#include "model.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "static unsigned counts[" + str(len(cases)) + "];",
        "static void require(int ok){if(!ok)abort();}",
        "static float fdata[32000*768] __attribute__((aligned(64)));",
        "static int8_t qdata[3072*768] __attribute__((aligned(64)));",
        "static unsigned char scratch[XLNET_MODEL_SCRATCH_BYTES] __attribute__((aligned(64)));",
        "static float hidden[128*768] __attribute__((aligned(64)));",
        "static float logits[128*32000] __attribute__((aligned(64)));",
        "static int32_t ids[128] __attribute__((aligned(64))),mask[128] __attribute__((aligned(64))),seg[128] __attribute__((aligned(64)));",
    ]
    for number, c in enumerate(cases):
        params = ", ".join(
            "MemRef" + str(a["rank"]) + " *a" + str(i)
            for i, a in enumerate(c["arguments"])
        )
        lines.append(f"void _mlir_ciface_kernel_{c['name']}({params}) " + "{")
        for i, (a, shape) in enumerate(zip(c["arguments"], argument_shapes(c))):
            lines.append(
                f"require(a{i} && a{i}->aligned && !((uintptr_t)a{i}->aligned&63u) && a{i}->offset==0);"
            )
            stride = 1
            for axis in reversed(range(len(shape))):
                lines.append(
                    f"require(a{i}->sizes[{axis}]=={shape[axis]} && a{i}->strides[{axis}]=={stride});"
                )
                stride *= shape[axis]
        lines += [f"++counts[{number}];", "}"]
    lines += [
        "int main(void){XLNetWeights w={0}; XLNetMatrix m={qdata,fdata}; w.embedding=fdata;w.relative_positions=fdata;",
        "for(int i=0;i<12;++i){XLNetLayer *p=&w.layers[i];p->q=p->k=p->v=p->r=p->o=p->ff1=p->ff2=m;for(int h=0;h<12;++h)p->segment[h]=m;",
        "p->rw_bias=p->rr_bias=p->rs_bias=p->attn_norm_weight=p->attn_norm_bias=p->ffn_norm_weight=p->ffn_norm_bias=p->ff1_bias=p->ff2_bias=fdata;}",
        "for(int i=0;i<32;++i){w.lm_tiles[i]=m;w.lm_bias[i]=fdata;}",
        "for(int i=0;i<128;++i){ids[i]=i;mask[i]=i<117;seg[i]=i<64?0:1;}",
        "require(xlnet_forward(&w,ids,mask,seg,scratch,sizeof(scratch),hidden,logits)==0);",
    ]
    for i, c in enumerate(cases):
        lines.append(f'printf("{c["name"]} %u\\n",counts[{i}]);')
    lines += [
        "ids[0]=-1;require(xlnet_forward(&w,ids,mask,seg,scratch,sizeof(scratch),hidden,logits)==-2);return 0;}"
    ]
    mock = out / "mock.c"
    mock.write_text("\n".join(lines) + "\n")
    exe = out / "check"
    subprocess.run(
        [
            shutil.which("cc"),
            " -std=c11".strip(),
            "-O1",
            "-DHOST_TEST",
            "-I",
            str(ROOT),
            "-I",
            str(ROOT / "model_runtime"),
            str(mock),
            str(ROOT / "model_runtime/model.c"),
            str(ROOT / "support.c"),
            "-lm",
            "-o",
            str(exe),
        ],
        check=True,
    )
    run = subprocess.run([str(exe)], text=True, capture_output=True, check=True)
    counts = {
        name: int(count)
        for name, count in (line.split() for line in run.stdout.splitlines())
    }
    assert counts["embedding_128x768"] == 1
    assert counts["matmul_i8_128x768x768"] == 4 * 12
    assert counts["matmul_i8_256x768x768"] == 12
    assert counts["relative_shift_12x128x128"] == 12
    assert counts["segment_gather_12x128x128"] == 12
    assert counts["softmax_12x128x128"] == 12
    assert counts["matmul_i8_128x128x64"] == 144
    assert counts["matmul_i8_128x256x64"] == 144
    assert counts["matmul_i8_128x2x64"] == 144
    assert counts["matmul_i8_128x64x128"] == 144
    assert counts["matmul_i8_128x1024x768"] == 32
    assert counts["layernorm_128x768"] == 24
    report = {
        "status": "PASS",
        "scope": "actual C model graph / typed descriptor and routing ABI only; mocks do not compute numerical outputs",
        "layers": 12,
        "lm_tiles": 32,
        "counts": counts,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS model typed ABI: 12 layers, 12 heads, 32 LM tiles, XLNet relative/segment paths"
    )


if __name__ == "__main__":
    main()

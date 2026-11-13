#!/usr/bin/env python3
"""Check all 1649 calls from the actual C model runtime using typed ABI mocks.

Mocks verify call order, exact shapes/strides, 64-byte alignment, nonaliasing,
input provenance and output liveness. They do not evaluate numerical kernels.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402 - local modules require the example path


def sequence():
    calls = ["embedding_lookup_128x384", "embedding_add_128x384", "layernorm_128x384"]
    dense = [
        "quantize_128x384",
        "quantize_384x384",
        "matmul_i8_128x384x384",
        "dequantize_128x384",
        "bias_add_128x384",
    ]
    for layer in range(12):
        calls += dense * 3 + ["split_heads_128x384"] * 3
        calls += [
            "quantize_128x32",
            "quantize_128x32",
            "matmul_i8_128x128x32",
            "dequantize_128x128",
        ] * 12
        calls += ["attention_mask_12x128x128", "softmax_12x128x128"]
        calls += [
            "quantize_128x128",
            "quantize_transpose_128x32",
            "matmul_i8_128x32x128",
            "dequantize_128x32",
        ] * 12
        calls += (
            ["merge_heads_12x128x32"] + dense + ["add_128x384", "layernorm_128x384"]
        )
        calls += [
            "quantize_128x384",
            "quantize_1536x384",
            "matmul_i8_128x1536x384",
            "dequantize_128x1536",
            "bias_add_128x1536",
            "gelu_128x1536",
        ]
        calls += [
            "quantize_128x1536",
            "quantize_384x1536",
            "matmul_i8_128x384x1536",
            "dequantize_128x384",
            "bias_add_128x384",
            "add_128x384",
            "layernorm_128x384",
        ]
    return calls + ["mean_pool_128x384", "l2_normalize_1x384"]


def source():
    seq = sequence()
    lines = [
        """#include "model_runtime/model.h"
#include "support.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char *expected[]={"""
        + ",".join(json.dumps(x) for x in seq)
        + """};
static size_t calls;
static unsigned char *arena;
static size_t arena_bytes;
static float *sentence_output,*hidden_output;
typedef struct { uintptr_t lo,hi; } Span;
static Span defined[4096]; static unsigned ndefined;
static void require(int ok,const char *why){if(!ok){fprintf(stderr,"runtime ABI failure call=%zu: %s\\n",calls,why);exit(1);}}
static void mark(void *p,size_t n){require(ndefined<4096,"provenance registry capacity");defined[ndefined++]=(Span){(uintptr_t)p,(uintptr_t)p+n};}
static int ready(void *p,size_t n){uintptr_t lo=(uintptr_t)p,hi=lo+n;while(lo<hi){uintptr_t next=lo;for(unsigned i=0;i<ndefined;i++)if(defined[i].lo<=lo&&defined[i].hi>next)next=defined[i].hi;if(next==lo)return 0;lo=next;}return 1;}
static void *allocation(size_t n){size_t m=(n+63)&~(size_t)63;void *p=aligned_alloc(64,m);require(p!=NULL,"allocation");memset(p,0,m);mark(p,n);return p;}
static void enter(const char *name){require(calls<sizeof(expected)/sizeof(*expected),"excess calls");require(!strcmp(expected[calls],name),name);++calls;}
static void descriptor(void *p,int64_t off,const int64_t *sizes,const int64_t *strides,const int64_t *shape,int rank){require(p&&!((uintptr_t)p&63)&&off==0,"aligned zero-offset descriptor");int64_t stride=1;for(int i=rank-1;i>=0;--i){require(sizes[i]==shape[i]&&strides[i]==stride,"shape/stride");stride*=shape[i];}}
static void nonalias(void *a,size_t an,void *b,size_t bn){require((uintptr_t)a+an<=(uintptr_t)b||(uintptr_t)b+bn<=(uintptr_t)a,"arguments overlap");}
void nr_copy_bytes(void *out,const void *in,size_t bytes){require(ready((void *)in,bytes),"copy reads initialized hidden state");require(out==hidden_output && bytes==128u*384u*4,"last hidden copy extent");memcpy(out,in,bytes);}
"""
    ]
    for case in inventory():
        params = ",".join(
            f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"])
        )
        lines.append(f"void _mlir_ciface_kernel_{case['name']}({params})" + "{")
        lines.append("enter(" + json.dumps(case["name"]) + ");")
        lengths = []
        for i, (a, shape) in enumerate(zip(case["arguments"], case["argument_shapes"])):
            n = math.prod(shape) * (1 if a["dtype"] == "i8" else 4)
            lengths.append(n)
            lines.append(f"int64_t shape{i}[]={{" + ",".join(map(str, shape)) + "};")
            lines.append(
                f"descriptor(a{i}->aligned,a{i}->offset,a{i}->sizes,a{i}->strides,shape{i},{a['rank']});"
            )
            for j in range(i):
                lines.append(f"nonalias(a{i}->aligned,{n},a{j}->aligned,{lengths[j]});")
            if i not in case["outputs"]:
                lines.append(
                    f'require(ready(a{i}->aligned,{n}),"read before producer or outside weights");'
                )
        for i in case["outputs"]:
            if case["name"] == "l2_normalize_1x384":
                lines.append(
                    f'require(a{i}->aligned==sentence_output,"sentence output pointer");'
                )
            else:
                lines.append(
                    f'require((uintptr_t)a{i}->aligned>=(uintptr_t)arena && (uintptr_t)a{i}->aligned+{lengths[i]}<=(uintptr_t)arena+arena_bytes,"workspace output bounds");'
                )
            lines.append(
                f"memset(a{i}->aligned,(int)(calls%199+1),{lengths[i]});mark(a{i}->aligned,{lengths[i]});"
            )
        lines.append("}")
    lines.append("""static GTELinear linear(int n,int k){return (GTELinear){allocation((size_t)n*k*4),allocation((size_t)n*4)};}
static GTELayerNorm norm(void){return (GTELayerNorm){allocation(384*4),allocation(384*4)};}
int main(void){
 GTEWeights w={0};w.word_embeddings=allocation(30522u*384u*4);w.position_embeddings=allocation(512u*384u*4);w.type_embeddings=allocation(2u*384u*4);w.embedding_norm=norm();
 for(int i=0;i<12;++i){GTELayer *l=&w.layers[i];l->query=linear(384,384);l->key=linear(384,384);l->value=linear(384,384);l->attention_output=linear(384,384);l->intermediate=linear(1536,384);l->output=linear(384,1536);l->attention_norm=norm();l->output_norm=norm();}
 int32_t *ids=allocation(128*4),*mask=allocation(128*4),*types=allocation(128*4),*positions=allocation(128*4);
 for(int i=0;i<128;++i){ids[i]=101+i;mask[i]=i<97;types[i]=i%2;positions[i]=i;}
 arena_bytes=gte_workspace_bytes();arena=aligned_alloc(64,arena_bytes);require(arena!=NULL,"workspace allocation");memset(arena,0xd3,arena_bytes);
 float *embedding=allocation(384*4),*last_hidden=allocation(128*384*4);sentence_output=embedding;hidden_output=last_hidden;
 require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes,embedding,last_hidden)==0,"inference return");
 require(calls==sizeof(expected)/sizeof(*expected),"complete schedule");
 size_t completed=calls;
 ids[0]=30522;require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes,embedding,NULL)==-1,"invalid token ID");ids[0]=101;
 positions[0]=512;require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes,embedding,NULL)==-1,"invalid position ID");positions[0]=0;
 types[0]=2;require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes,embedding,NULL)==-1,"invalid segment ID");types[0]=0;
 mask[0]=2;require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes,embedding,NULL)==-1,"invalid binary mask");mask[0]=1;
 require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes-1,embedding,NULL)==-1,"short workspace");
 require(gte_infer(&w,ids,mask,types,positions,arena+4,arena_bytes,embedding,NULL)==-1,"misaligned workspace");
 for(int i=0;i<128;++i)mask[i]=0;require(gte_infer(&w,ids,mask,types,positions,arena,arena_bytes,embedding,NULL)==-1,"empty mask");
 require(calls==completed,"invalid requests dispatch no kernels");
 printf("MODEL_ABI_PASS calls=%zu workspace=%zu numerical_kernels_executed=false\\n",calls,arena_bytes);
 return 0;
}""")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "build/model-runtime-check")
    p.add_argument("--cc", default=shutil.which("clang") or "cc")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    mock = a.output / "mock.c"
    mock.write_text(source())
    binary = a.output / "check-model"
    commands = [
        [
            a.cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-DHOST_TEST",
            "-I",
            str(ROOT),
            str(mock),
            str(ROOT / "model_runtime/model.c"),
            "-o",
            str(binary),
        ],
        [str(binary)],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        logs.append("$ " + " ".join(command) + "\n" + result.stdout + result.stderr)
        if result.returncode:
            (a.output / "check.log").write_text("\n".join(logs))
            print(logs[-1])
            return 1
    (a.output / "check.log").write_text("\n".join(logs))
    report = {
        "status": "MODEL_ABI_PASS",
        "operator_calls": len(sequence()),
        "validation": "actual C model routing with typed shape/stride/alignment/nonalias/order/provenance mocks; no numerical kernel execution",
        "invalid_input_checks": 7,
        "board_executed": False,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(logs[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

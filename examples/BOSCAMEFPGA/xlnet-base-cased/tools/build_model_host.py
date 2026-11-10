#!/usr/bin/env python3
"""Execute the actual C graph and emitted scalar LLVM kernels on the host.

Only AME integer dot products use a separate exact signed-i8/i32 C oracle.
This is a local implementation cross-check, never FPGA numerical evidence.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from build_model_bin import LLVM, ROOT, run

sys.path.insert(0, str(ROOT))
from cases import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=ROOT / "build/model-bin")
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    # Retain old invocations while emitted reference generation is independent
    # of optional mathematical output files.
    p.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, default=ROOT / "build/model-host")
    p.add_argument("--trace-first-call", type=int, default=0)
    p.add_argument("--trace-last-call", type=int, default=0)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    header = a.build / "model_artifacts.h"
    if not header.is_file():
        raise ValueError(
            "first run build_model_bin.py to generate checkpoint/input layout"
        )
    objects, jobs, matmul = [], [], []
    flags = [
        "-O3",
        "-march=native",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-fno-pie",
        "-I" + str(ROOT),
        "-I" + str(ROOT / "model_runtime"),
        "-I" + str(a.build.resolve()),
    ]

    def add(source, name):
        target = a.output / (name + ".o")
        objects.append(target)
        jobs.append(
            (
                [LLVM / "clang", *flags, "-c", source, "-o", target],
                a.output / ("compile-" + name + ".log"),
            )
        )

    for case in inventory():
        name = case["name"]
        if case["family"] == "matmul_i8":
            matmul += [
                f"void _mlir_ciface_kernel_{name}(MemRef2 *a,MemRef2 *b,MemRef2 *c){{",
                "int8_t *x=a->aligned,*w=b->aligned;int32_t *y=c->aligned;",
                "for(int64_t i=0;i<a->sizes[0];++i)for(int64_t j=0;j<b->sizes[0];++j){",
                "int32_t sum=0;for(int64_t k=0;k<a->sizes[1];++k)",
                "sum+=(int32_t)x[i*a->sizes[1]+k]*(int32_t)w[j*b->sizes[1]+k];",
                "y[i*b->sizes[0]+j]=sum;}}",
            ]
        else:
            add(a.kernels.resolve() / name / "adapter.c", "adapter_" + name)
            add(a.kernels.resolve() / name / "kernel.ll", "scalar_" + name)
    source = [
        '#include "model_artifacts.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <math.h>",
        "extern unsigned char xlnet_workspace[];",
        "static const char *destination;",
        "float rsqrtf(float x){return 1.0f/sqrtf(x);}",
        "void nr_copy_bytes(void *d,const void *s,size_t n){memcpy(d,s,n);}",
        "void xlnet_layer_checkpoint(unsigned i,const float *h){",
        'char path[4096];const char *name=i==0?"embedding":i==12?"last_hidden_state":0;',
        'if(name)snprintf(path,sizeof(path),"%s/%s.bin",destination,name);',
        'else snprintf(path,sizeof(path),"%s/layer_%02u.bin",destination,i);',
        'FILE *f=fopen(path,"wb");if(!f||fwrite(h,4,128*768,f)!=128*768||fclose(f))abort();',
        'printf("HOST_LAYER %u\\n",i);fflush(stdout);}',
        *matmul,
        "int main(int argc,char **argv){if(argc!=2)return 1;destination=argv[1];",
        "int result=xlnet_forward(&xlnet_weights,(int32_t*)xlnet_input,",
        "(int32_t*)(xlnet_input+512),(int32_t*)(xlnet_input+1024),xlnet_workspace,",
        "XLNET_MODEL_SCRATCH_BYTES,(float*)(xlnet_workspace+XLNET_MODEL_SCRATCH_BYTES),",
        "#if XLNET_MODEL_LM_HEAD",
        "(float*)(xlnet_workspace+XLNET_MODEL_SCRATCH_BYTES+128u*768u*4));",
        'char path[4096];snprintf(path,sizeof(path),"%s/logits.bin",destination);',
        'FILE *f=fopen(path,"wb");if(!f||fwrite(xlnet_workspace+XLNET_MODEL_SCRATCH_BYTES+128u*768u*4,4,128*32000,f)!=128*32000||fclose(f))abort();',
        "#else",
        "0);",
        "#endif",
        "return result;}",
    ]
    harness = a.output / "host_main.c"
    harness.write_text("\n".join(source) + "\n")
    add(harness, "host_main")
    add(ROOT / "model_runtime/model.c", "model_graph")
    add(a.build.resolve() / "model_blobs.S", "blobs")
    wrap_flags = []
    if a.trace_first_call:
        trace = [
            '#include "model.h"',
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "static unsigned calls;",
            "static void dump(unsigned call,const char *name,const char *argument,void *base,int64_t offset,size_t count,size_t width){",
            'char path[4096];snprintf(path,sizeof(path),"%s/call_%04u_%s_%s.bin",'
            + json.dumps(str(a.output))
            + ",call,name,argument);",
            'FILE *f=fopen(path,"wb");if(!f||fwrite((char*)base+offset*width,width,count,f)!=count||fclose(f))abort();}',
        ]
        for case in inventory():
            name = case["name"]
            symbol = "_mlir_ciface_kernel_" + name
            params = ",".join(
                f"MemRef{arg['rank']} *a{i}" for i, arg in enumerate(case["arguments"])
            )
            values = ",".join(f"a{i}" for i in range(len(case["arguments"])))
            trace += [
                f"extern void __real_{symbol}({params});",
                f"void __wrap_{symbol}({params}){{",
                f"unsigned call=++calls;__real_{symbol}({values});",
                f"if(call>={a.trace_first_call}u&&call<={a.trace_last_call}u){{",
                f'printf("TRACE_CALL %u {name}\\n",call);',
            ]
            for i, arg in enumerate(case["arguments"]):
                count = "*".join(f"a{i}->sizes[{j}]" for j in range(arg["rank"]))
                width = 1 if arg["dtype"] == "i8" else 4
                trace.append(
                    f'dump(call,"{name}","{arg["name"]}",a{i}->aligned,a{i}->offset,{count},{width});'
                )
            trace += ["}}"]
            wrap_flags.append("-Wl,--wrap=" + symbol)
        path = a.output / "trace_wrappers.c"
        path.write_text("\n".join(trace) + "\n")
        add(path, "trace_wrappers")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: run(*job), jobs))
    exe = a.output / "xlnet_host"
    run(
        [LLVM / "clang", "-no-pie", *wrap_flags, *objects, "-lm", "-o", exe],
        a.output / "link.log",
    )
    run([exe, a.output], a.output / "execution.log")
    records = []
    for r in json.loads((a.build / "output-layout.json").read_text()):
        path = a.output / (r["name"] + ".bin")
        assert path.stat().st_size == r["nbytes"], (
            "incomplete emitted output: " + r["name"]
        )
        got = np.fromfile(path, dtype=r["dtype"]).reshape(r["shape"])
        records.append(
            {
                "name": r["name"],
                "shape": r["shape"],
                "dtype": r["dtype"],
                "nbytes": r["nbytes"],
                "finite": bool(np.isfinite(got).all()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    report = {
        "status": "EXECUTED_LOCAL_GRAPH",
        "fpga_executed": False,
        "scope": "actual 12-layer model.c and emitted scalar LLVM; exact C integer-dot oracle replaces AME only",
        "files": records,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

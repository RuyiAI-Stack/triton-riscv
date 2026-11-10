#!/usr/bin/env python3
"""Run the actual C graph and emitted scalar LLVM locally with exact INT8 dots.

This is an implementation reference, never FPGA execution evidence. Only matrix
products use independent exact C INT8/INT32 dots; every other operator uses its
original adapter and emitted kernel.ll. Input and checkpoint blobs are the same
ones packed into the target image.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/model-board")
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/model-host-emitted"
    )
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.kernels = args.kernels.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    buddy = (
        Path(os.environ.get("BUDDY_ROOT", str(REPO / "buddy-mlir")))
        .expanduser()
        .resolve()
    )
    llvm = (
        Path(os.environ.get("LLVM_BIN", str(buddy / "llvm/build/bin")))
        .expanduser()
        .resolve()
    )
    compiler = llvm / "clang"
    runtime = buddy / "runtime/boscame"
    flags = [
        "-O3",
        "-march=native",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-fno-pie",
        "-DE5_TARGET_CHECKPOINTS=1",
    ]
    flags += [
        "-I" + str(path)
        for path in (ROOT, ROOT / "tools", runtime / "nh_e6", runtime / "runtime")
    ]
    objects = []
    jobs = []
    source_hashes = {
        str(path): sha(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "cases.py",
            ROOT / "support.h",
            ROOT / "model_runtime/model.h",
            ROOT / "tools/model_board.h",
        )
    }

    def add(path, name):
        path = path.resolve()
        source_hashes[str(path)] = sha(path)
        output = args.output / (name + ".o")
        objects.append(output)
        jobs.append(([str(compiler), *flags, "-c", str(path), "-o", str(output)], name))

    def run(job):
        command, name = job
        process = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (args.output / (name + ".log")).write_text(
            " ".join(command) + "\n" + process.stdout
        )
        if process.returncode:
            raise RuntimeError(name + ": " + process.stdout[-3000:])

    dots = []
    for case in inventory():
        name = case["name"]
        if case["family"] == "matmul_i8":
            dots += [
                f"void _mlir_ciface_kernel_{name}(MemRef2 *a,MemRef2 *b,MemRef2 *c)"
                + " {",
                "const int8_t *x=a->aligned,*w=b->aligned;int32_t *y=c->aligned;",
                "for(int64_t i=0;i<a->sizes[0];++i)for(int64_t j=0;j<b->sizes[0];++j){",
                "int32_t sum=0;for(int64_t k=0;k<a->sizes[1];++k)",
                "sum+=(int32_t)x[i*a->sizes[1]+k]*(int32_t)w[j*b->sizes[1]+k];",
                "y[i*b->sizes[0]+j]=sum;}}",
            ]
        else:
            add(args.kernels / name / "adapter.c", "adapter_" + name)
            add(args.kernels / name / "kernel.ll", "scalar_" + name)
    harness = [
        '#include "support.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <math.h>",
        "float e5_sentence_output[384] __attribute__((aligned(64)));",
        "float e5_hidden_output[128*384] __attribute__((aligned(64)));",
        "float e5_layer_outputs[13*128*384] __attribute__((aligned(64)));",
        "static unsigned completed;",
        "float rsqrtf(float x){return 1.0f/sqrtf(x);}",
        "void nr_copy_bytes(void *d,const void *s,size_t n){memcpy(d,s,n);}",
        "void e5_model_checkpoint(unsigned layer,const float *hidden){",
        "if(layer!=completed||layer>=13)abort();",
        "memcpy(e5_layer_outputs+(size_t)layer*128*384,hidden,128u*384u*4);",
        '++completed;printf("HOST_CHECKPOINT %u\\n",layer);fflush(stdout);}',
        *dots,
        "int e5_model_launch(void);",
        "static void save(const char *directory,const char *name,const void *data,size_t count){",
        'char path[4096];snprintf(path,sizeof(path),"%s/%s.bin",directory,name);',
        'FILE *f=fopen(path,"wb");if(!f||fwrite(data,4,count,f)!=count||fclose(f))abort();}',
        "int main(int argc,char **argv){if(argc!=2)return 2;",
        "int result=e5_model_launch();if(result||completed!=13)return result?result:3;",
        'save(argv[1],"sentence_embedding",e5_sentence_output,384);',
        'save(argv[1],"last_hidden",e5_hidden_output,128u*384u);',
        'save(argv[1],"layer_hidden",e5_layer_outputs,13u*128u*384u);return 0;}',
    ]
    (args.output / "main.c").write_text("\n".join(harness) + "\n")
    add(args.output / "main.c", "main")
    add(ROOT / "model_runtime/model.c", "graph")
    add(args.build / "model_launch.c", "launch")
    add(args.build / "model_blobs.S", "blobs")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, jobs))
    executable = args.output / "e5-host"
    run(
        (
            [
                str(compiler),
                "-no-pie",
                *map(str, objects),
                "-lm",
                "-o",
                str(executable),
            ],
            "link",
        )
    )
    run(([str(executable), str(args.output)], "execute"))
    files = [
        {"name": path.stem, "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(args.output.glob("*.bin"))
    ]
    report = {
        "status": "LOCAL_GRAPH_EXECUTED",
        "fpga_executed": False,
        "scope": "actual model.c and generated launch with actual emitted scalar LLVM; only matrix products use independent exact C INT8/INT32 dots",
        "params_sha256": sha(args.build / "params.bin"),
        "inputs_sha256": sha(args.build / "inputs.bin"),
        "source_hashes": source_hashes,
        "compiler": str(compiler),
        "compile_flags": flags,
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "tensors": len(files),
                "bytes": sum(item["bytes"] for item in files),
            }
        )
    )


if __name__ == "__main__":
    main()

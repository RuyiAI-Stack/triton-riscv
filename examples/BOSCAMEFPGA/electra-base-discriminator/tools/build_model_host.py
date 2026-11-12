#!/usr/bin/env python3
"""Local implementation oracle: actual C graph/emitted scalar LLVM, exact integer-dot oracle.

This executes locally and supplies expected data; it is never hardware evidence.
The independently checked AME matmuls are replaced only here by exact C i8/i32 dots.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=ROOT / "build/model-board")
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument("--output", type=Path, default=ROOT / "build/model-host-emitted")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    clang = ROOT.parents[2] / "buddy-mlir/llvm/build/bin/clang"
    flags = [
        "-O3",
        "-march=native",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-fno-pie",
        "-DELECTRA_PREQUANTIZED_WEIGHTS",
        "-DELECTRA_MODEL_BOARD",
        "-I" + str(ROOT),
        "-I" + str(a.build.resolve()),
    ]
    objects, jobs, source_hashes = [], [], {}

    def add(path, name):
        path = path.resolve()
        source_hashes[str(path)] = sha(path)
        dest = a.output / (name + ".o")
        objects.append(dest)
        jobs.append(([str(clang), *flags, "-c", str(path), "-o", str(dest)], name))

    def run(job):
        command, name = job
        proc = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (a.output / (name + ".log")).write_text(" ".join(command) + "\n" + proc.stdout)
        if proc.returncode:
            raise RuntimeError(name + ": " + proc.stdout[-3000:])

    dots = []
    for case in inventory():
        name = case["name"]
        if case["family"] == "matmul_i8":
            dots += [
                f"void _mlir_ciface_kernel_{name}(MemRef2 *a,MemRef2 *b,MemRef2 *c) {{",
                "const int8_t *x=a->aligned,*w=b->aligned; int32_t *y=c->aligned;",
                "for (int64_t i=0;i<a->sizes[0];++i) for(int64_t j=0;j<b->sizes[0];++j) {",
                "int32_t sum=0; for(int64_t k=0;k<a->sizes[1];++k)",
                "sum+=(int32_t)x[i*a->sizes[1]+k]*(int32_t)w[j*b->sizes[1]+k];",
                "y[i*b->sizes[0]+j]=sum; }}",
            ]
        else:
            add(a.kernels / name / "adapter.c", "adapter_" + name)
            add(a.kernels / name / "kernel.ll", "scalar_" + name)
    harness = [
        '#include "params.h"',
        '#include "support.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <math.h>",
        "extern const unsigned char electra_params[],electra_params_end[];",
        "extern const int32_t electra_input_ids[],electra_token_type_ids[],electra_attention_mask[];",
        "static const char *directory;",
        "float rsqrtf(float x) { return 1.0f/sqrtf(x); }",
        "void electra_model_trace(unsigned stage,const float *data,size_t count) {",
        'char path[4096]; const char *name=stage==0?"embedding":stage==13?"logits":0;',
        'if(name)snprintf(path,sizeof(path),"%s/%s.bin",directory,name);',
        'else snprintf(path,sizeof(path),"%s/layer_%02u.bin",directory,stage-1);',
        'FILE *f=fopen(path,"wb");if(!f||fwrite(data,4,count,f)!=count||fclose(f))abort();',
        'printf("HOST_STAGE %u\\n",stage);fflush(stdout);}',
        *dots,
        "int main(int argc,char **argv) { if(argc!=2)return 2;directory=argv[1];",
        "void *scratch=aligned_alloc(64,ELECTRA_MODEL_WORKSPACE_BYTES);",
        "float *logits=aligned_alloc(64,128*4);if(!scratch||!logits)return 3;",
        "ElectraWeights weights={0};",
        "if(electra_bind_board_weights(&weights,electra_params,electra_params_end-electra_params))return 4;",
        "int result=electra_infer(&weights,electra_input_ids,electra_token_type_ids,",
        "electra_attention_mask,logits,scratch,ELECTRA_MODEL_WORKSPACE_BYTES);",
        "free(scratch);free(logits);return result;}",
    ]
    (a.output / "main.c").write_text("\n".join(harness) + "\n")
    add(a.output / "main.c", "main")
    add(ROOT / "model_runtime/model.c", "graph")
    add(a.build / "blobs.S", "blobs")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, jobs))
    exe = a.output / "electra-host"
    run(([str(clang), "-no-pie", *map(str, objects), "-lm", "-o", str(exe)], "link"))
    run(([str(exe), str(a.output)], "execute"))
    files = [
        dict(name=p.stem, bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(a.output.glob("*.bin"))
    ]
    report = dict(
        status="LOCAL_GRAPH_EXECUTED",
        fpga_executed=False,
        scope="actual model.c plus actual emitted scalar LLVM; only AME integer products use exact C oracle",
        params_sha256=sha(a.build / "params.bin"),
        source_hashes=source_hashes,
        files=files,
    )
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                tensors=len(files),
                bytes=sum(x["bytes"] for x in files),
            )
        )
    )


if __name__ == "__main__":
    main()

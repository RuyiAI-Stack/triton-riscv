#!/usr/bin/env python3
"""Run the actual GTE C graph and scalar LLVM locally with exact INT8 dots.

This is a local implementation oracle, never FPGA execution evidence. Matrix
products use independent exact signed INT8/INT32 C dots. All other operators use
their original typed adapters and emitted kernel.ll, with the identical packed
checkpoint and inputs used by the target image.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from cases import inventory  # noqa: E402 - local modules require the example path
from check_model_runtime import sequence  # noqa: E402 - local modules require the example path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", type=Path, default=ROOT / "build/full-model-checkpoints"
    )
    parser.add_argument(
        "--kernels", type=Path, default=ROOT / "build/scheduler-output/kernels"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/full-model-host-emitted"
    )
    parser.add_argument(
        "--graph-source",
        type=Path,
        default=ROOT / "model_runtime/model.c",
        help="Allow replaying the exact frozen model.c from an earlier board build",
    )
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.kernels = args.kernels.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((args.build / "model-input.json").read_text())
    build = json.loads((args.build / "build-report.json").read_text())
    layers = build.get("encoder_layers", 12)
    trace = (
        json.loads((args.build / "trace-layout.json").read_text())
        if (args.build / "trace-layout.json").exists()
        else None
    )
    assert sha(args.build / "params.bin") == contract["params_sha256"]
    assert sha(args.build / "input.bin") == contract["input_sha256"]
    compiler = REPO / "buddy-mlir/llvm/build/bin/clang"
    runtime = REPO / "buddy-mlir/runtime/boscame"
    flags = [
        "-O3",
        "-march=native",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-DGTE_MODEL_CHECKPOINTS",
        f"-DGTE_MODEL_LAYERS={layers}",
        "-fno-pie",
    ]
    flags += [
        "-I" + str(path)
        for path in (
            ROOT,
            ROOT / "model_runtime",
            runtime / "nh_e6",
            runtime / "runtime",
        )
    ]
    objects, jobs = [], []
    source_hashes = {
        str(path): sha(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "cases.py",
            ROOT / "support.h",
            ROOT / "model_runtime/model.h",
        )
    }
    kernel_hashes = {}

    def add(path, name):
        path = path.resolve()
        source_hashes[str(path)] = sha(path)
        obj = args.output / (name + ".o")
        objects.append(obj)
        jobs.append(([str(compiler), *flags, "-c", str(path), "-o", str(obj)], name))

    def run(job):
        command, name = job
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (args.output / (name + ".log")).write_text(
            " ".join(command) + "\n" + process.stdout
        )
        if process.returncode:
            raise RuntimeError(name + ": " + process.stdout[-3000:])

    dots = []
    for case in inventory():
        name = case["name"]
        kernel_hashes[name] = {
            file: sha(args.kernels / name / file)
            for file in ("kernel.ll", "adapter.c", "kernel.o", "frontend.json")
        }
        if case["family"] == "matmul_i8":
            # All K <= 1536, so |sum| <= 1536*128*128 < INT32_MAX.
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
        "extern float model_embedding[384],model_hidden[128*384];",
        f"static float model_layers[{layers + 1}*128*384] __attribute__((aligned(64)));",
        "static unsigned completed;",
        "float rsqrtf(float x){return 1.0f/sqrtf(x);}",
        "void nr_copy_bytes(void *d,const void *s,size_t n){memcpy(d,s,n);}",
        "void gte_model_checkpoint(unsigned layer,const float *hidden){",
        f"if(layer!=completed||layer>={layers + 1})abort();",
        "memcpy(model_layers+(size_t)layer*128*384,hidden,128u*384u*4);",
        '++completed;printf("HOST_CHECKPOINT %u\\n",layer);fflush(stdout);}',
        *dots,
        "int model_launch(void);",
        "static void save(const char *directory,const char *name,const void *data,size_t count){",
        'char path[4096];snprintf(path,sizeof(path),"%s/%s.bin",directory,name);',
        'FILE *f=fopen(path,"wb");if(!f||fwrite(data,4,count,f)!=count||fclose(f))abort();}',
        "int main(int argc,char **argv){if(argc!=2)return 2;",
        f"int result=model_launch();if(result||completed!={layers + 1})return result?result:3;",
        'save(argv[1],"sentence_embedding",model_embedding,384);',
        'save(argv[1],"last_hidden_state",model_hidden,128u*384u);',
        f'save(argv[1],"layer_checkpoints",model_layers,{layers + 1}u*128u*384u);',
        "return 0;}",
    ]
    if trace:
        declarations = [
            f"unsigned char model_operator_trace[{trace['nbytes']}] __attribute__((aligned(64)));"
        ]
        harness = declarations + harness
        harness.insert(
            -1,
            f'save(argv[1],"operator_trace",model_operator_trace,{trace["nbytes"] // 4});',
        )
        case_ids = {case["name"]: i + 1 for i, case in enumerate(inventory())}
        expected_calls = sequence()[: 3 + 137 * layers] + sequence()[-2:]
        wrappers = [
            '#include "support.h"',
            "#include <string.h>",
            "#include <stdlib.h>",
            "static const unsigned short expected[]={"
            + ",".join(str(case_ids[name]) for name in expected_calls)
            + "};",
            "extern unsigned char model_operator_trace[];",
            "static unsigned calls;",
        ]
        for case in inventory():
            name = case["name"]
            params = ",".join(
                f"MemRef{argument['rank']} *a{i}"
                for i, argument in enumerate(case["arguments"])
            )
            arguments = ",".join(f"a{i}" for i in range(len(case["arguments"])))
            symbol = "_mlir_ciface_kernel_" + name
            wrappers += [
                f"extern void __real_{symbol}({params});",
                f"void __wrap_{symbol}({params})" + "{",
                f"if(calls>={len(expected_calls)}||expected[calls]!={case_ids[name]})abort();",
                f"__real_{symbol}({arguments});",
            ]
            for item in trace["entries"]:
                if item["case"] == name:
                    wrappers += [
                        f"if(calls=={item['call']})memcpy(model_operator_trace+{item['offset']}u,a{item['argument']}->aligned,{item['nbytes']}u);"
                    ]
            wrappers += ["++calls;", "}"]
        (args.output / "trace_wrappers.c").write_text("\n".join(wrappers) + "\n")
        add(args.output / "trace_wrappers.c", "trace_wrappers")
    (args.output / "main.c").write_text("\n".join(harness) + "\n")
    add(args.output / "main.c", "main")
    add(args.graph_source, "graph")
    add(args.build / "model_launcher.c", "launch")
    add(args.build / "model_blobs.S", "blobs")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, jobs))
    executable = args.output / "gte-host"
    wrap = (
        ["-Wl,--wrap=_mlir_ciface_kernel_" + case["name"] for case in inventory()]
        if trace
        else []
    )
    run(
        (
            [
                str(compiler),
                "-no-pie",
                *map(str, objects),
                *wrap,
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
    frozen_sources = {}
    (args.output / "sources").mkdir(exist_ok=True)
    for name, digest in source_hashes.items():
        source = Path(name)
        destination = args.output / "sources" / (digest + source.suffix)
        destination.write_bytes(source.read_bytes())
        frozen_sources[name] = str(destination.relative_to(args.output))
    report = {
        "status": "LOCAL_GRAPH_EXECUTED",
        "fpga_executed": False,
        "scope": "actual model.c and generated launcher with emitted scalar LLVM; only matrix products use independent exact C INT8/INT32 dots",
        "params_sha256": sha(args.build / "params.bin"),
        "input_sha256": sha(args.build / "input.bin"),
        "encoder_layers": layers,
        "model_graph_sha256": sha(args.graph_source),
        "source_hashes": source_hashes,
        "kernel_hashes": kernel_hashes,
        "frozen_source_files": frozen_sources,
        "compiler": str(compiler),
        "compiler_sha256": sha(compiler),
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

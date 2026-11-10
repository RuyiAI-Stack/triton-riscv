"""
Python lit test 'vit-model-adapters.py' covering ViT model adapter wiring for
FPGA tests.

Inputs: triton-shared-opt (and sometimes the Triton Python plugin) passed via
lit substitutions.

Outputs: assertions / FileCheck on exported contract IR or numeric results;
non-zero exit on mismatch.

When used: BOSCAME FPGA integration suite. Keep RUN lines and inner docstrings
that describe the scenario.
"""

# RUN: %PYTHON %s %t

import importlib.util
import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VIT = REPO / "examples/BOSCAMEFPGA/vit-base-patch16-224"
sys.path.insert(0, str(VIT / "tools"))
spec = importlib.util.spec_from_file_location(
    "vit_replace", VIT / "tools/replace_triton.py"
)
replace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replace)


def record(name, symbol, inputs, output):
    case = replace.describe(name)
    case["adapter_entry"] = "_mlir_ciface_kernel_" + name
    return {
        "symbol": symbol,
        "case": case,
        "family": case["family"],
        "graph_ranks": [len(shape) for shape in inputs],
        "graph_shapes": inputs,
        "result_rank": len(output),
        "result_shape": output,
        "storage": symbol + "_storage",
    }


def test_adapters(work):
    call = record(
        "layernorm_197x768", "test_ln", [[1, 197, 768], [768], [768]], [1, 197, 768]
    )
    adapter = work / "adapters.c"
    replace.generate_adapters([call], adapter)
    assert "(MemRef2 *)" not in adapter.read_text()
    invalid = record(
        "layernorm_197x768", "invalid", [[2, 197, 768], [768], [768]], [1, 197, 768]
    )
    try:
        replace.generate_adapters([invalid], work / "invalid.c")
    except ValueError:
        pass
    else:
        raise AssertionError("non-unit batch was accepted")
    mock = work / "mock.c"
    mock.write_text(r"""
#include "support.h"
static float x[197*768+16] __attribute__((aligned(64)));
static float w[784] __attribute__((aligned(64)));
static float b[784] __attribute__((aligned(64)));
static int errors,calls;
extern void _mlir_ciface_test_ln(MemRef3*,MemRef3*,MemRef1*,MemRef1*);
void _mlir_ciface_kernel_layernorm_197x768(MemRef2*a,MemRef1*weight,MemRef1*bias,MemRef2*out) {
 ++calls; errors += a->aligned!=x || a->offset!=16 || a->sizes[0]!=197 || a->sizes[1]!=768;
 errors += a->strides[0]!=768 || a->strides[1]!=1 || weight->offset!=16 || bias->offset!=16;
 errors += ((uintptr_t)out->aligned&63)!=0; ((float*)out->aligned)[out->offset]=37;
}
int main(int argc,char**argv) {
 MemRef3 a=make_3(x,1,197,768),out; MemRef1 weight=make_1(w,768),bias=make_1(b,768);
 a.offset=weight.offset=bias.offset=16; a.strides[0]=0;
 if(argc>1 && argv[1][0]=='s') a.strides[1]=769;
 if(argc>1 && argv[1][0]=='b') a.sizes[0]=2;
 _mlir_ciface_test_ln(&out,&a,&weight,&bias);
 errors += out.sizes[0]!=1 || out.sizes[1]!=197 || out.sizes[2]!=768 || ((float*)out.aligned)[out.offset]!=37;
 return errors || calls!=1;
}
""")
    clang = shutil.which("clang")
    binary = work / "check"
    subprocess.run(
        [
            clang,
            "-std=c11",
            "-O1",
            "-Wall",
            "-Werror",
            "-I",
            str(VIT),
            str(adapter),
            str(mock),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
    for invalid in ("stride", "batch"):
        assert (
            subprocess.run([str(binary), invalid], capture_output=True).returncode != 0
        )


def test_index(work):
    name = "layernorm_197x768"
    directory = work / "index" / name
    directory.mkdir(parents=True, exist_ok=True)
    case = replace.describe(name)
    manifest = dict(case)
    manifest["kernel_module_sha256"] = replace.digest(case["kernel_source"])
    for filename, key in (
        ("kernel.ttir", "ttir_sha256"),
        ("adapter.c", "adapter_sha256"),
    ):
        (directory / filename).write_text(
            replace.adapter_source(case) if filename == "adapter.c" else "fixture\n"
        )
        manifest[key] = replace.digest(directory / filename)
    (directory / "frontend.json").write_text(json.dumps(manifest))
    module = directory / "kernel.boscame.mlir"
    module.write_text("module {}\n")
    result = {
        "name": name,
        "status": "BLOCKED",
        "stage": "frontend",
        "reason": "unsupported operation",
    }
    result_path = directory / "build-result.json"
    result_path.write_text(json.dumps(result))
    rejected = []
    assert not replace.load_index(work / "index", rejected)
    assert rejected[0]["reason"] == "unsupported operation"
    result.update(status="PASS", stage="ttir")
    result_path.write_text(json.dumps(result))
    assert not replace.load_index(work / "index")
    result.update(
        stage="boscame",
        kernel_module_sha256=manifest["kernel_module_sha256"],
        artifacts={module.name: replace.digest(module)},
    )
    result_path.write_text(json.dumps(result))
    assert name in replace.load_index(work / "index")
    module.write_text("changed after build\n")
    assert not replace.load_index(work / "index")


def test_product_adapters(work):
    calls = []
    for prefix, name in (
        ("product", "matmul_i8_1x1000x768"),
        ("accumulate", "gemm_i8_1x1000x768"),
    ):
        case = replace.describe(name)
        case["adapter_entry"] = "_mlir_ciface_kernel_" + name
        calls.append(
            {
                "symbol": prefix,
                "case": case,
                "family": "w8a8",
                "void": True,
                "graph_ranks": [2, 2, 2],
                "graph_shapes": [[1, 768], [1000, 768], [1, 1000]],
            }
        )
    adapter = work / "product_adapters.c"
    replace.generate_adapters(calls, adapter)
    assert "acc[i] = 0" not in adapter.read_text()
    assert (
        replace._w8a8_case(
            {calls[1]["case"]["name"]: calls[1]["case"]},
            "matmul",
            (1, 1000, 768),
            None,
        )
        is None
    )
    mock = work / "product_mock.c"
    mock.write_text(r"""
#include "support.h"
static int8_t a[768] __attribute__((aligned(64)));
static int8_t b[1000 * 768] __attribute__((aligned(64)));
static int32_t c[1016] __attribute__((aligned(64)));
static int errors;
void _mlir_ciface_kernel_matmul_i8_1x1000x768(MemRef2 *x, MemRef2 *w,
                                                      MemRef2 *out) {
  errors += x->aligned != a || w->aligned != b || out->aligned != c;
  errors += out->offset != 16 || c[16] != 73;
  c[16] = 11;
}
void _mlir_ciface_kernel_gemm_i8_1x1000x768(MemRef2 *x, MemRef2 *w,
                                            MemRef2 *out) {
  errors += x->aligned != a || w->aligned != b || out->aligned != c;
  errors += out->offset != 16 || c[16] != 37;
  c[16] += 11;
}
extern void _mlir_ciface_product(MemRef2 *, MemRef2 *, MemRef2 *);
extern void _mlir_ciface_accumulate(MemRef2 *, MemRef2 *, MemRef2 *);
int main(void) {
  MemRef2 x = make_2(a, 1, 768), w = make_2(b, 1000, 768);
  MemRef2 out = make_2(c, 1, 1000); out.offset = 16;
  c[16] = 73; _mlir_ciface_product(&x, &w, &out); errors += c[16] != 11;
  c[16] = 37; _mlir_ciface_accumulate(&x, &w, &out); errors += c[16] != 48;
  return errors != 0;
}
""")
    clang = REPO / "llvm/build/bin/clang"
    if not clang.is_file():
        clang = shutil.which("clang")
    binary = work / "product_check"
    subprocess.run(
        [
            str(clang),
            "-std=c11",
            "-O1",
            "-Wall",
            "-Werror",
            "-I",
            str(VIT),
            str(adapter),
            str(mock),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
    for mode in (None, "fp32"):
        path = work / ("gate-" + str(mode) + ".json")
        command = [
            sys.executable,
            str(VIT / "tools/model_simcheck.py"),
            "--build-report",
            str(work / "missing-build.json"),
            "--simcheck-report",
            str(work / "missing-sim.json"),
            "--kernels",
            str(work / "missing-kernels"),
            "--report",
            str(path),
        ]
        if mode:
            command += ["--mode", mode]
        result = subprocess.run(command, capture_output=True)
        assert result.returncode == 2
        if mode == "fp32":
            assert b"invalid choice" in result.stderr and not path.exists()
        else:
            report = json.loads(path.read_text())
            assert report["mode"] == "w8a8" and report["status"] == "SKIP"
            assert report["model_executed"] is False
            assert len(report["required_cases"]) == 32
            assert "quantize_197x197" in report["required_cases"]


def test_attention_prerequisite_evidence(work):
    """Primitive prerequisite evidence must cover every round and complete grid."""
    from model_simcheck import PATCH_DEPENDENCIES, SHARED_CASES, W8A8_LINEAR_CASES

    root = work / "prerequisite-fixtures"
    kernels = root / "kernels"
    checks = []
    for name in sorted(SHARED_CASES | W8A8_LINEAR_CASES):
        case = replace.describe(name)
        directory = kernels / name
        directory.mkdir(parents=True, exist_ok=True)
        module = directory / "kernel.boscame.mlir"
        module.write_text("module {} // identity fixture, no numerical execution\n")
        check = {
            "name": name,
            "status": "PASS",
            "programs": math.prod(case["grid"]),
            "total_programs": math.prod(case["grid"]),
            "module_sha256": replace.digest(module),
            "kernel_module_sha256": replace.digest(case["kernel_source"]),
        }
        check["runs"] = [
            {
                key: check[key]
                for key in (
                    "programs",
                    "total_programs",
                    "module_sha256",
                    "kernel_module_sha256",
                )
            }
        ]
        checks.append(check)
    by_name = {check["name"]: check for check in checks}
    patch = by_name["patch_embed_196x768"]
    stages = [
        {
            "case": name,
            **{
                key: by_name[name][key]
                for key in (
                    "programs",
                    "total_programs",
                    "module_sha256",
                    "kernel_module_sha256",
                )
            },
        }
        for name in PATCH_DEPENDENCIES
    ]
    grid = math.prod(replace.describe(patch["name"])["grid"])
    patch["programs"] = patch["total_programs"] = grid + sum(
        stage["programs"] for stage in stages
    )
    patch["dependency_sha256"] = {
        stage["case"]: stage["module_sha256"] for stage in stages
    }
    patch["runs"] = [
        {"programs": patch["programs"], "im2col_programs": grid, "stages": stages}
    ]
    builds = root / "build-report.json"
    builds.write_text(json.dumps({"schema": 1, "cases": checks}))
    simchecks = root / "simcheck-report.json"
    checkpoint = root / "checkpoint-fixture"
    checkpoint.write_text("not a model checkpoint; prerequisite gate fixture only\n")
    for scenario in (
        "current",
        "current-multiround",
        "stale-source",
        "partial-grid",
        "missing-round",
        "partial-round",
        "stale-round",
        "partial-patch-child",
    ):
        records = copy.deepcopy(checks)
        attention = next(item for item in records if item["name"] == "quantize_197x64")
        if scenario in (
            "current-multiround",
            "missing-round",
            "partial-round",
            "stale-round",
        ):
            attention["comparison_rounds"] = 3
            attention["runs"] = [copy.deepcopy(attention["runs"][0]) for _ in range(3)]
            attention["programs"] *= 3
            attention["total_programs"] *= 3
        if scenario == "stale-source":
            attention["kernel_module_sha256"] = "stale"
        elif scenario == "partial-grid":
            attention["programs"] = attention["total_programs"] = 1
        elif scenario == "missing-round":
            attention["runs"].pop()
        elif scenario == "partial-round":
            attention["runs"][1]["programs"] = 1
        elif scenario == "stale-round":
            attention["runs"][1]["module_sha256"] = "stale"
        elif scenario == "partial-patch-child":
            next(item for item in records if item["name"].startswith("patch_embed_"))[
                "runs"
            ][0]["stages"][0]["programs"] = 1
        simchecks.write_text(json.dumps({"schema": 1, "cases": records}))
        path = root / (scenario + ".json")
        result = subprocess.run(
            [
                sys.executable,
                str(VIT / "tools/model_simcheck.py"),
                "--build-report",
                str(builds),
                "--simcheck-report",
                str(simchecks),
                "--kernels",
                str(kernels),
                "--checkpoint",
                str(checkpoint),
                "--report",
                str(path),
            ],
            capture_output=True,
        )
        assert result.returncode == 2, result.stderr.decode()
        report = json.loads(path.read_text())
        assert report["model_pass"] is False and report["model_executed"] is False
        artifacts = [
            reason for reason in report["reasons"] if reason["stage"] == "artifact"
        ]
        if scenario in ("current", "current-multiround"):
            assert not artifacts
            assert [reason["stage"] for reason in report["reasons"]] == [
                "model-execution"
            ]
        elif scenario in (
            "partial-grid",
            "missing-round",
            "partial-round",
            "partial-patch-child",
        ):
            coverage = [
                reason for reason in report["reasons"] if reason["stage"] == "simcheck"
            ]
            expected_case = (
                patch["name"]
                if scenario == "partial-patch-child"
                else attention["name"]
            )
            assert len(coverage) == 1 and coverage[0]["case"] == expected_case
            assert "grid" in coverage[0]["reason"]
        else:
            assert len(artifacts) == 1 and artifacts[0]["case"] == attention["name"]
            assert (
                "direct" in artifacts[0]["reason"] or "source" in artifacts[0]["reason"]
            )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    test_adapters(work)
    test_index(work)
    test_product_adapters(work)
    test_attention_prerequisite_evidence(work)
    print("ViT generic descriptor/index and primitive prerequisite checks passed")


if __name__ == "__main__":
    main()

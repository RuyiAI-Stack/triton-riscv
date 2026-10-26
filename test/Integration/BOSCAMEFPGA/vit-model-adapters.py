# RUN: %PYTHON %s %t
"""ViT graph descriptors retain ranks/offsets and reject blocked kernels."""

import importlib.util
import json
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
    calls = [
        record(
            "layernorm_197x768",
            "test_ln",
            [[1, 197, 768], [768], [768]],
            [1, 197, 768],
        ),
        record(
            "attention_qk_12x197x197x64",
            "test_qk",
            [[1, 12, 197, 64], [1, 12, 64, 197]],
            [1, 12, 197, 197],
        ),
    ]
    adapter = work / "adapters.c"
    replace.generate_adapters(calls, adapter)
    source = adapter.read_text()
    assert "(MemRef2 *)" not in source and "(MemRef3 *)" not in source
    assert "__attribute__((aligned(64)))" in source
    invalid = record(
        "layernorm_197x768",
        "bad_batch",
        [[2, 197, 768], [768], [768]],
        [1, 197, 768],
    )
    try:
        replace.generate_adapters([invalid], work / "invalid.c")
    except ValueError as error:
        assert "leading-unit batch" in str(error)
    else:
        raise AssertionError("non-unit batch was accepted")
    mock = work / "mock.c"
    mock.write_text(r"""
#include "support.h"
static float x[197 * 768 + 16] __attribute__((aligned(64)));
static float w[768 + 16] __attribute__((aligned(64)));
static float b[768 + 16] __attribute__((aligned(64)));
static float q[12 * 197 * 64 + 16] __attribute__((aligned(64)));
static float k[12 * 64 * 197 + 16] __attribute__((aligned(64)));
static int errors, ln_calls, qk_calls;
extern void _mlir_ciface_test_ln(MemRef3 *, MemRef3 *, MemRef1 *, MemRef1 *);
extern void _mlir_ciface_test_qk(MemRef4 *, MemRef4 *, MemRef4 *);
void _mlir_ciface_kernel_layernorm_197x768(MemRef2 *a, MemRef1 *weight,
                                        MemRef1 *bias, MemRef2 *out) {
  ++ln_calls;
  errors += a->aligned != x || a->offset != 16;
  errors += a->sizes[0] != 197 || a->sizes[1] != 768;
  errors += a->strides[0] != 768 || a->strides[1] != 1;
  errors += weight->offset != 16 || bias->offset != 16;
  errors += out->sizes[0] != 197 || out->sizes[1] != 768;
  errors += ((uintptr_t)out->aligned & 63u) != 0;
  ((float *)out->aligned)[out->offset] = 37.0f;
}
void _mlir_ciface_kernel_attention_qk_12x197x197x64(MemRef3 *a, MemRef3 *other,
                                               MemRef3 *out) {
  ++qk_calls;
  errors += a->aligned != q || other->aligned != k;
  errors += a->offset != 16 || other->offset != 16;
  errors += a->sizes[0] != 12 || a->sizes[1] != 197 || a->sizes[2] != 64;
  errors += a->strides[0] != 12608 || a->strides[1] != 64 || a->strides[2] != 1;
  errors += other->sizes[1] != 64 || other->sizes[2] != 197;
  errors += other->strides[0] != 12608 || other->strides[1] != 197 || other->strides[2] != 1;
  errors += out->sizes[0] != 12 || out->sizes[1] != 197 || out->sizes[2] != 197;
  errors += out->strides[0] != 38809 || out->strides[1] != 197 || out->strides[2] != 1;
  ((float *)out->aligned)[out->offset] = 19.0f;
}
int main(int argc, char **argv) {
  MemRef3 a = make_3(x, 1, 197, 768), output;
  MemRef1 weight = make_1(w, 768), bias = make_1(b, 768);
  a.offset = weight.offset = bias.offset = 16;
  a.strides[0] = 0; /* Unit batch strides do not affect contiguity. */
  if (argc > 1 && argv[1][0] == 's') a.strides[1] = 769;
  if (argc > 1 && argv[1][0] == 'b') a.sizes[0] = 2;
  _mlir_ciface_test_ln(&output, &a, &weight, &bias);
  errors += output.sizes[0] != 1 || output.sizes[1] != 197 || output.sizes[2] != 768;
  errors += ((float *)output.aligned)[output.offset] != 37.0f;
  MemRef4 qa = make_4(q, 1, 12, 197, 64), kb = make_4(k, 1, 12, 64, 197), result;
  qa.offset = kb.offset = 16;
  _mlir_ciface_test_qk(&result, &qa, &kb);
  errors += result.sizes[0] != 1 || result.sizes[1] != 12;
  errors += result.sizes[2] != 197 || result.sizes[3] != 197;
  errors += ((float *)result.aligned)[result.offset] != 19.0f;
  return errors || ln_calls != 1 || qk_calls != 1;
}
""")
    clang = REPO / "llvm/build/bin/clang"
    if not clang.is_file():
        clang = shutil.which("clang")
    binary = work / "check"
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
    for invalid in ("stride", "batch"):
        run = subprocess.run([str(binary), invalid], capture_output=True)
        assert run.returncode != 0, "invalid source descriptor was accepted"


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
            replace.adapter_source(case)
            if filename == "adapter.c"
            else "fixture\n"
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
        ("product", "matmul_i8_overwrite_1x1000x768"),
        ("accumulate", "matmul_i8_1x1000x768"),
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
void _mlir_ciface_kernel_matmul_i8_overwrite_1x1000x768(MemRef2 *x, MemRef2 *w,
                                                      MemRef2 *out) {
  errors += x->aligned != a || w->aligned != b || out->aligned != c;
  errors += out->offset != 16 || c[16] != 73;
  c[16] = 11;
}
void _mlir_ciface_kernel_matmul_i8_1x1000x768(MemRef2 *x, MemRef2 *w,
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
        assert subprocess.run(command, capture_output=True).returncode == 2
        report = json.loads(path.read_text())
        assert report["mode"] == (mode or "w8a8")
        assert report["status"] == "SKIP"


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    test_adapters(work)
    test_index(work)
    test_product_adapters(work)
    print("ViT model adapter and compiled-kernel index checks passed")


if __name__ == "__main__":
    main()

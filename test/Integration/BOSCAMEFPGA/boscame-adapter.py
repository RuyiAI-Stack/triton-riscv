# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Compile the emitted MLIR wrappers and real typed-unranked adapters on the host."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from Inputs.pipeline import (  # noqa: E402
    buddy_opt_path,
    experimental_pipeline,
)

REPO = Path(os.environ["BUDDY_ROOT"])


def run(command):
    result = subprocess.run(
        [str(x) for x in command], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = Path(buddy_opt_path())
    clang = REPO / "llvm/build/bin/clang"
    if not clang.is_file():
        clang = shutil.which("clang")
    for name, fixture, symbol, adapter_args, counts, widths, shapes, grid in (
        (
            "gemm",
            "matmul-i8.ttir",
            "triton_matmul_i8_128x1024x128",
            ["--adapter", "experimental-boscame"],
            [128 * 128, 1024 * 128, 128 * 1024],
            [1, 1, 4],
            [(128, 128), (1024, 128), (128, 1024)],
            (2, 16, 1),
        ),
        (
            "tiled-gemm",
            "matmul-tiled-i8.ttir",
            "triton_matmul_i8_196x768x768",
            ["--adapter", "experimental-boscame"],
            [196 * 768, 768 * 768, 196 * 768],
            [1, 1, 4],
            [(196, 768), (768, 768), (196, 768)],
            (7, 6, 1),
        ),
        (
            "layernorm",
            "layernorm.ttir",
            "triton_layernorm_197x768",
            ["--adapter"],
            [197 * 768, 768, 768, 197 * 768],
            [4, 4, 4, 4],
            [(197, 768), (768,), (768,), (197, 768)],
            (197, 1, 1),
        ),
    ):
        entry = symbol.removeprefix("triton_")
        module, lowered, llvm = (
            work / (name + suffix) for suffix in (".mlir", ".llvm.mlir", ".ll")
        )
        input_path = Path(__file__).parent / "Inputs" / fixture
        pipeline = experimental_pipeline(
            symbol + ":2,16,1" if name == "gemm" else None
        )
        if name == "tiled-gemm":
            contract = run(
                [
                    sys.executable,
                    Path(__file__).resolve().parents[3]
                    / "examples/BOSCAMEFPGA"
                    / entry
                    / "build.py",
                    "--launch-grid",
                ]
            ).strip()
            assert contract == symbol + ":7,6,1"
            pipeline[0] += " launch-grid=" + contract
        run(
            [
                opt,
                input_path,
                *pipeline,
                "-o",
                module,
            ]
        )
        run([opt, module, "--lower-bosc-ame", "-o", lowered])
        run(
            [
                os.environ["BUDDY_TRANSLATE"],
                lowered,
                "--buddy-to-llvmir",
                "-o",
                llvm,
            ]
        )
        text = llvm.read_text()
        raw = re.search(rf"^define void @{symbol}\([^\n]+", text, re.M)[0]
        wrapper = re.search(
            rf"^define void @_mlir_ciface_{symbol}\(.*?^\}}", text, re.M | re.S
        )[0]
        wrapper_path = work / (name + ".wrapper.ll")
        wrapper_path.write_text(
            raw.replace("define", "declare", 1).removesuffix(" {")
            + "\n"
            + wrapper
            + "\n"
        )
        adapter = work / (name + ".adapter.c")
        adapter.write_text(
            run(
                [
                    sys.executable,
                    REPO / "examples/FPGA-BOSCAME" / entry / "build.py",
                    *adapter_args,
                ]
            )
        )
        params = ", ".join(
            f"int64_t r{i}, void *p{i}" for i in range(len(counts))
        )
        scalars = ", ".join(
            "int32_t " + n for n in ("gx", "gy", "gz", "x", "y", "z")
        )
        arrays = "\n".join(
            f"static unsigned char a{i}[{count * width + 128}] __attribute__((aligned(64)));"
            for i, (count, width) in enumerate(zip(counts, widths))
        )
        checks = "\n".join(
            f"Descriptor *d{i} = p{i}; failures += r{i} != 1 || d{i}->allocated != a{i} || d{i}->aligned != a{i} + 64 || d{i}->offset != 0 || d{i}->size != {count} || d{i}->stride != 1;"
            for i, count in enumerate(counts)
        )
        descs = []
        for i, (shape, width) in enumerate(zip(shapes, widths)):
            if len(shape) == 2:
                descs.append(
                    f"MemRef2 m{i} = {{a{i}, a{i}, {64 // width}, {{{shape[0]}, {shape[1]}}}, {{{shape[1]}, 1}}}};"
                )
            else:
                descs.append(
                    f"Descriptor m{i} = {{a{i}, a{i}, {64 // width}, {shape[0]}, 1}};"
                )
        args = ", ".join(f"&m{i}" for i in range(len(counts)))
        decl = ", ".join(
            "MemRef2 *" if len(shape) == 2 else "Descriptor *"
            for shape in shapes
        )
        probe = work / (name + ".probe.c")
        probe.write_text(f"""#include <stdint.h>
typedef struct {{ void *allocated, *aligned; int64_t offset, size, stride; }} Descriptor;
typedef struct {{ void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; }} MemRef2;
{arrays}
static int calls, failures;
void {symbol}({params}, {scalars}) {{
  {checks}
  failures += gx != {grid[0]} || gy != {grid[1]} || gz != 1 || x != calls / {grid[1]} || y != calls % {grid[1]} || z != 0;
  ++calls;
}}
extern void _mlir_ciface_kernel_{entry}({decl});
int main(int argc, char **argv) {{
  (void)argv;
  {" ".join(descs)}
  if (argc > 1) m1.aligned = a0;
  _mlir_ciface_kernel_{entry}({args});
  return failures || calls != {grid[0] * grid[1]};
}}
""")
        binary = work / name
        run([clang, "-O0", adapter, wrapper_path, probe, "-o", binary])
        run([binary])
        bad = subprocess.run(
            [str(binary), "invalid-alias"], capture_output=True
        )
        assert bad.returncode != 0, "adapter accepted aliasing DDR arguments"
    # CHECK: PASS adapters: actual C wrappers, typed unranked descriptors, element offsets/counts, six grid scalars and nonalias checks
    print(
        "PASS adapters: actual C wrappers, typed unranked descriptors, element offsets/counts, six grid scalars and nonalias checks"
    )


if __name__ == "__main__":
    main()

"""Build and verify a nonzero classifier DDR-output probe using actual kernel.o."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2]
BUDDY = REPO / "buddy-mlir"
LLVM = BUDDY / "llvm/build/bin"
CASE = "linear_1x1000x1536"
sys.path.insert(0, str(ROOT / "tools"))
from quantized_linear_oracle import reference  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    p.write_text(json.dumps(v, indent=2) + "\n")


def command(args, log):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    with log.open("a") as f:
        f.write(" ".join(map(str, args)) + "\n" + p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(p.stderr)


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    original = ROOT / "build/kernels" / CASE
    launch = """#include "support.h"
extern void _mlir_ciface_kernel_linear_1x1000x1536(MemRef2 *,MemRef2 *,MemRef1 *,MemRef2 *);
int launch(void) {
  float *out=workspace(64), *a=workspace(4160), *bias=workspace(10304), *w=workspace(14336);
  uint32_t *before=workspace(0), *after=workspace(4096);
  for (int i=0;i<16;++i) before[i]=after[i]=0x513a7e29u;
  for (int i=0;i<1000;++i) { ((uint32_t *)out)[i]=0x7fc12345u; bias[i]=(float)(i%7-3)*0.0625f; }
  for (int i=0;i<1536;++i) a[i]=(float)(i%29-14)*0.0625f;
  for (int i=0;i<1536000;++i) w[i]=(float)(i%23-11)*0.03125f;
  __asm__ volatile("fence rw,rw" ::: "memory");
  MemRef2 A=make_2(a,1,1536), W=make_2(w,1000,1536), O=make_2(out,1,1000);
  MemRef1 B=make_1(bias,1000);
  _mlir_ciface_kernel_linear_1x1000x1536(&A,&W,&B,&O);
  return 0;
}
"""
    (directory / "launch.c").write_text(launch)
    support = (ROOT / "board_support.c").read_text()
    exports = []
    for name, off, dtype, shape in [
        ("Logits", 64, "<f4", [1, 1000]),
        ("Before", 0, "<u4", [16]),
        ("After", 4096, "<u4", [16]),
    ]:
        value = dict(
            version=1,
            name=name,
            address=hex(0x80100000 + off),
            nbytes=int(np.prod(shape)) * 4,
            dtype=dtype,
            shape=shape,
        )
        line = "@@FPGA_OUTPUT " + json.dumps(value, separators=(",", ":")) + "\r\n"
        exports.append("    print_uart(" + json.dumps(line) + ");")
    anchor = '    print_uart(\n        ok ? "@@FPGA_RESULT "'
    assert anchor in support
    support = support.replace(
        anchor, "    if (ok) {\n" + "\n".join(exports) + "\n    }\n" + anchor
    )
    (directory / "board_support.c").write_text(support)
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdc_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-nostdlib",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-pie",
        "-fno-stack-protector",
        "-ffunction-sections",
        "-fdata-sections",
        "-ffp-contract=off",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-DVIT_WORKSPACE_BYTES=16777216",
        "-DUART_FREQ_HZ=5529600",
        "-DBOSC_AME_DMA_ASSERT(x)=",
    ]
    flags += [
        "-I" + str(p)
        for p in [
            ROOT,
            BUDDY / "runtime/boscame/runtime",
            BUDDY / "runtime/boscame/uart",
            BUDDY / "runtime/boscame/nh_e6",
            BUDDY / "runtime/boscame/nr",
        ]
    ]
    log = directory / "build.log"
    for name in ("launch", "board_support"):
        command(
            [
                LLVM / "clang",
                *flags,
                "-c",
                directory / (name + ".c"),
                "-o",
                directory / (name + ".o"),
            ],
            log,
        )
    objects = [
        directory / "launch.o",
        directory / "board_support.o",
        original / "kernel.o",
    ]
    objects += [
        original / "obj" / (x + ".o")
        for x in (
            "nh_e6_boot",
            "nh_e6_platform",
            "uart",
            "bosc_ame_dma",
            "nr_math",
            "ame_sync",
            "adapter",
        )
    ]
    command(
        [
            shutil.which("ld.lld"),
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            ROOT / "link.ld",
            "-Map=" + str(directory / "image.map"),
            "-o",
            directory / "image.elf",
            *objects,
        ],
        log,
    )
    command(
        [
            sys.executable,
            BUDDY / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            directory / "image.elf",
            "--objdump",
            LLVM / "llvm-objdump",
            "--output",
            directory / "elf-audit.json",
        ],
        log,
    )
    command(
        [
            LLVM / "llvm-objcopy",
            "-O",
            "binary",
            directory / "image.elf",
            directory / "image.bin",
        ],
        log,
    )
    save(
        directory / "build.json",
        dict(
            case=CASE,
            image_sha256=sha(directory / "image.bin"),
            production_object_sha256=sha(original / "kernel.o"),
            production_adapter_sha256=sha(original / "adapter.c"),
            fixture="dyadic_nonzero_v1",
            outputs=["Logits", "Before", "After"],
        ),
    )


def verify(directory):
    config = json.loads((directory / "build.json").read_text())
    a = ((np.arange(1536) % 29 - 14).astype(np.float32) / 16).reshape(1, 1536)
    w = ((np.arange(1536000) % 23 - 11).astype(np.float32) / 32).reshape(1000, 1536)
    b = (np.arange(1000) % 7 - 3).astype(np.float32) / 16
    expected = reference(dict(A=a, W=w, Bias=b))["Out"]
    actual = np.fromfile(directory / "Logits.bin", dtype="<f4").reshape(1, 1000)
    for name in ("Before", "After"):
        guard = np.fromfile(directory / (name + ".bin"), dtype="<u4")
        assert guard.shape == (16,) and np.all(guard == 0x513A7E29), name
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    result = dict(
        case=CASE,
        status="PASS" if actual.tobytes() == expected.tobytes() else "FAIL",
        bit_exact_w8a8=actual.tobytes() == expected.tobytes(),
        max_abs=float(error.max()),
        rmse=float(np.sqrt(np.mean(error**2))),
        output_sha256=sha(directory / "Logits.bin"),
        reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        image_sha256=config["image_sha256"],
    )
    save(directory / "numeric-result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def execute(directory):
    sys.path.insert(0, "/home/chh/gitprojects/fpga-scheduler/src")
    from fpga_scheduler import Client

    cfg = json.loads((directory / "build.json").read_text())
    with Client(
        "http://127.0.0.1:18762",
        token_file=str(REPO / ".agent/secrets/fpga112.token"),
        request_timeout=360,
    ) as client:
        with client.run_test(
            str(directory / "image.bin"), name="effnet-classifier-nonzero", timeout=600
        ) as stream:
            with (directory / "run.log").open("w") as f:
                for chunk in stream:
                    f.write(chunk)
                    f.flush()
        task = client.job(stream.id)
        save(directory / "task.json", asdict(task))
        assert stream.result.ok, stream.result
        assert stream.result.details["image_sha256"] == cfg["image_sha256"]
        save(directory / "artifacts.json", client.artifacts(stream.id))
        for name in cfg["outputs"]:
            client.download_artifact(stream.id, name, directory / (name + ".bin"))
    return verify(directory)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "build/classifier-board-check")
    p.add_argument("--run", action="store_true")
    p.add_argument("--collect", action="store_true")
    a = p.parse_args()
    if a.collect:
        return verify(a.output)
    build(a.output)
    if a.run:
        return execute(a.output)


if __name__ == "__main__":
    main()

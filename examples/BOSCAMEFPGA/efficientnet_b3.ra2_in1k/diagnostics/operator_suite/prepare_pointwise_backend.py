#!/usr/bin/env python3
"""Freeze pointwise nonzero fixtures; production payloads stay unchanged."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
BUDDY = REPO / "buddy-mlir"
LLVM = BUDDY / "llvm/build/bin"
sys.path.insert(0, str(MODEL))
from cases import inventory  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "backend_builder", HERE / "build_backend_abab.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

C_TEMPLATE = r"""
#include "support.h"
typedef struct { void *allocated,*aligned; int64_t offset,size,stride; } Descriptor1D;
typedef struct { int64_t rank; void *descriptor; } Unranked;
extern void @RAW@(Unranked*,Unranked*,Unranked*,Unranked*,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t);
extern void @PUBLIC@(MemRef2*,MemRef2*,MemRef1*,MemRef2*);
static int8_t xv(int i) { return (int8_t)((i*13+7)%251-125); }
static int8_t wv(int i) { return (int8_t)((i*17+11)%255-128); }
static const int points[@POINT_COUNT@][3] = {@POINTS@};
int launch(void) {
  int8_t *x=workspace(@X@),*w=workspace(@W@);
  int32_t *b=workspace(@B@),*o=workspace(64);
  uint32_t *before=workspace(0),*after=workspace(@AFTER@);
  for(int i=0;i<16;++i)before[i]=after[i]=0x513a7e29u;
  for(int i=0;i<@X_SIZE@;++i)x[i]=xv(i);
  for(int i=0;i<@W_SIZE@;++i)w[i]=wv(i);
  for(int i=0;i<@OC@;++i)b[i]=i%31-15;
  for(int i=0;i<@OUT_SIZE@;++i)o[i]=(int32_t)0x6c9d3a17u;
  Descriptor1D dx={x,x,0,@X_SIZE@,1},dw={w,w,0,@W_SIZE@,1};
  Descriptor1D db={b,b,0,@OC@,1},dout={o,o,0,@OUT_SIZE@,1};
  Unranked ux={1,&dx},uw={1,&dw},ub={1,&db},uo={1,&dout};
  __asm__ volatile("fence rw,rw" ::: "memory");
  uint64_t pair_begin=nr_cycles(),pair_end=nr_cycles();
  uint64_t begin=nr_cycles(),cycles=0;
  @COMPUTE@
  uint64_t adapter_total=nr_cycles()-begin;
  __asm__ volatile("fence rw,rw" ::: "memory");
  unsigned errors=0;
  @VERIFY@
  for(int i=0;i<16;++i)if(before[i]!=0x513a7e29u||after[i]!=0x513a7e29u)++errors;
  nr_puts("PROFILE case=@CASE@ mode=fused selected=all\r\n");
  nr_puts("PROFILE stage=raw_gemm calls=0x@CALLS_HEX@ cycles=0x");nr_hex64(cycles);nr_puts("\r\n");
  nr_puts("PROFILE adapter_total_cycles=0x");nr_hex64(adapter_total);
  nr_puts(" timer_pairs=0x1 total_empty_pair_cycles=0x");nr_hex64(pair_end-pair_begin);nr_puts("\r\n");
  return print_check("pointwise_backend_fixture",errors,errors?1.0f:0.0f);
}
"""


def prepare(case, args):
    source = args.source / case["name"]
    directory = args.output / case["name"]
    if directory.exists():
        raise RuntimeError("refuse to overwrite fixture: " + str(directory))
    directory.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix not in (".bin", ".elf", ".map"):
            shutil.copy2(path, directory / path.name)
    spatial, channels, oc = (case["constexprs"][key] for key in ("M", "K", "N"))
    gx, gy, gz = case["grid"]
    total = math.prod(case["grid"])
    indices = sorted({round(i * (total - 1) / 31) for i in range(32)})
    points = [(i // (gy * gz), (i // gz) % gy, i % gz) for i in indices]

    def align(value):
        return (value + 63) & ~63

    offsets = {"o": 64, "before": 0, "after": align(64 + spatial * oc * 4)}
    offsets["x"] = offsets["after"] + 64
    offsets["w"] = align(offsets["x"] + spatial * channels)
    offsets["b"] = align(offsets["w"] + oc * channels)
    if offsets["b"] + oc * 4 >= 16777216:
        raise RuntimeError("fixture exceeds workspace")
    public = "_mlir_ciface_kernel_" + case["name"]
    raw = "_mlir_ciface_" + case["symbol"]
    if args.full:
        compute = f"MemRef2 X=make_2(x,{spatial},{channels}),W=make_2(w,{oc},{channels}),O=make_2(o,{spatial},{oc}); MemRef1 B=make_1(b,{oc}); {public}(&X,&W,&B,&O); cycles=nr_cycles()-begin;"
        verify = f"for(int i=0;i<{spatial * oc};++i)o[i]+=b[i%{oc}];"
        calls = total
    else:
        compute = f"for(int i=0;i<32;++i){{uint64_t one=nr_cycles(); {raw}(&ux,&uw,&ub,&uo,{gx},{gy},{gz},points[i][0],points[i][1],points[i][2]); cycles+=nr_cycles()-one;}}"
        bm, bn = case["constexprs"]["BM"], case["constexprs"]["BN"]
        verify = f"for(int p=0;p<32;++p)for(int r=0;r<{bm};++r)for(int c=0;c<{bn};++c){{int row=points[p][0]*{bm}+r,col=points[p][1]*{bn}+c;if(row>={spatial}||col>={oc})continue;int32_t expected=0;for(int k=0;k<{channels};++k)expected+=(int)xv(row*{channels}+k)*(int)wv(col*{channels}+k);if(o[row*{oc}+col]!=expected)++errors;}}"
        calls = 32
    replacements = {
        "RAW": raw,
        "PUBLIC": public,
        "POINT_COUNT": 32,
        "POINTS": ",".join("{" + ",".join(map(str, p)) + "}" for p in points),
        "X": offsets["x"],
        "W": offsets["w"],
        "B": offsets["b"],
        "AFTER": offsets["after"],
        "X_SIZE": spatial * channels,
        "W_SIZE": oc * channels,
        "OUT_SIZE": spatial * oc,
        "OC": oc,
        "COMPUTE": compute,
        "VERIFY": verify,
        "CASE": case["name"],
        "CALLS_HEX": format(calls, "x"),
    }
    text = C_TEMPLATE
    for key, value in replacements.items():
        text = text.replace("@" + key + "@", str(value))
    fixture = directory / "pointwise-launch.c"
    fixture.write_text(text)
    command = [
        "make",
        "--no-print-directory",
        "-f",
        str(MODEL / "board.mk"),
        "bin",
        "KERNELS=" + str(args.output),
        "BUDDY_ROOT=" + str(BUDDY),
        "LLVM_BIN=" + str(LLVM),
        "PYTHON=" + sys.executable,
    ]
    with (
        (directory / "seed-build.stdout").open("w") as stdout,
        (directory / "seed-build.stderr").open("w") as stderr,
    ):
        subprocess.run(
            command, cwd=MODEL / case["name"], stdout=stdout, stderr=stderr, check=True
        )
    flags = [
        LLVM / "clang",
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
        "-I" + str(MODEL),
        "-DVIT_WORKSPACE_BYTES=16777216",
        "-DUART_FREQ_HZ=5529600",
        "-DBOSC_AME_DMA_ASSERT(x)=",
    ]
    flags += [
        "-I" + str(BUDDY / "runtime/boscame" / name)
        for name in ("runtime", "uart", "nh_e6", "nr")
    ]
    builder.run(
        [*flags, "-c", fixture, "-o", directory / "obj/launch.o"],
        directory,
        "fixture-launch",
    )
    support = (MODEL / "board_support.c").read_text()
    if args.full:
        declarations = []
        for name, offset, size, dtype, shape in (
            ("Output", 64, spatial * oc * 4, "<i4", [spatial, oc]),
            ("Before", 0, 64, "<u4", [16]),
            ("After", offsets["after"], 64, "<u4", [16]),
        ):
            record = {
                "version": 1,
                "name": name,
                "address": hex(0x80100000 + offset),
                "nbytes": size,
                "dtype": dtype,
                "shape": shape,
            }
            declarations.append(
                "    print_uart("
                + json.dumps(
                    "@@FPGA_OUTPUT "
                    + json.dumps(record, separators=(",", ":"))
                    + "\r\n"
                )
                + ");"
            )
        needle = "    print_uart(\n        ok ?"
        if needle not in support:
            raise RuntimeError("cannot locate board report success emission")
        support = support.replace(
            needle, "    if (ok) {\n" + "\n".join(declarations) + "\n    }\n" + needle
        )
    board = directory / "pointwise-board-support.c"
    board.write_text(support)
    builder.run(
        [*flags, "-c", board, "-o", directory / "obj/board_support.o"],
        directory,
        "fixture-support",
    )
    manifest = {
        "case": case["name"],
        "mode": "fused",
        "selected_stage": "all",
        "stage_metadata": [
            {
                "stage": "raw_gemm",
                "symbol": case["symbol"],
                "grid": case["grid"],
                "constexprs": case["constexprs"],
            }
        ],
        "coverage": "full public adapter output"
        if args.full
        else "32 uniform original raw grid programs including tails",
        "selected_tile_pairs": [{"x": x, "y": y, "z": z} for x, y, z in points],
        "selected_tiles": calls,
        "full_grid_tiles": total,
        "expected_stage_calls": {"raw_gemm": calls},
        "original_full_grid_stage_calls": {"raw_gemm": total},
        "zero_initialized_launcher": False,
        "fixture": {
            "kind": "deterministic_nonzero_v1",
            "X": "(i*13+7)%251-125",
            "Weight": "(i*17+11)%255-128",
            "Bias": "i%31-15",
            "sentinel": "0x6c9d3a17",
            "guards": "0x513a7e29",
        },
        "timeout_seconds": 1200 if args.full else 60,
        "timing": "raw-return timers exclude initialization/oracle/bias; no added per-call hardware fence",
        "offsets": offsets,
        "case_metadata": {
            key: case[key] for key in ("name", "shape", "grid", "constexprs", "symbol")
        },
        "frozen_payload_sha256": {
            name: builder.helpers.sha256(directory / name)
            for name in ("kernel.o", "adapter.c", "kernel.boscame.mlir")
        },
        "launcher_sha256": builder.helpers.sha256(fixture),
        "support_sha256": builder.helpers.sha256(board),
    }
    builder.helpers.write_json(directory / "profile-manifest.json", manifest)
    print("FIXTURE " + case["name"] + " calls=" + str(calls), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=MODEL / "build/operator-suite/suite-linked80"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    names = {"conv2d_144x144_24x144_k1_s1", "conv2d_18x18_136x816_k1_s1"}
    for case in inventory():
        if case["name"] in names:
            prepare(case, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

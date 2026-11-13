#!/usr/bin/env python3
"""Paired board diagnostics; download actual DDR output and check on host.

Uses the scheduler SDK from --scheduler-root (including artifact support).
Variants change only the singleton SPM store and its ordering/cache policy.
No production kernel, runtime, Makefile or remote service is modified.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from cases import describe  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(argv, log):
    result = subprocess.run(
        [str(x) for x in argv],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with log.open("a") as handle:
        handle.write(" ".join(str(x) for x in argv) + "\n" + result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)


def build(case, variant, destination):
    config = describe(case)
    m, n = config["shape"]
    count = m * n
    directory = destination / variant / case
    directory.mkdir(parents=True, exist_ok=True)
    original = ROOT / "build/kernels" / case
    board = original / "board"
    buddy = REPO / "buddy-mlir"
    llvm = buddy / "llvm/build/bin"
    platform = buddy / "runtime/boscame/nh_e6"
    runtime = buddy / "runtime/boscame/runtime"
    uart = buddy / "runtime/boscame/uart"
    nr = buddy / "runtime/boscame/nr"
    # Output first: even the largest case and snapshots fit the scheduler's
    # default first-4-MiB capture window. All inputs/guards are disjoint.
    tail = ((64 + count * 4 + 63) // 64) * 64
    diag = tail + 64
    xoff = diag + 256
    roff = ((xoff + count * 4 + 63) // 64) * 64
    soff = ((roff + m * 4 + 63) // 64) * 64
    launch = f"""#include "board_support.h"
#define M {m}
#define N {n}
extern void _mlir_ciface_kernel_{case}(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);
extern void dequantize_snapshot(void *);
int launch(void) {{
  float *y = workspace(64), *r = workspace({roff}), *s = workspace({soff});
  int32_t *x = workspace({xoff});
  uint32_t *before = workspace(0), *after = workspace({tail});
  for (int i=0; i<16; ++i) before[i] = after[i] = 0x513a7e29u;
  for (int i=0; i<M; ++i) r[i] = (i % 17 + 1) * 0.00390625f;
  for (int j=0; j<N; ++j) s[j] = (j % 31 + 1) * 0.001953125f;
  for (size_t i=0; i<(size_t)M*N; ++i) {{
    x[i] = ((int)i % 1023 - 511) * 7919;
    ((uint32_t *)y)[i] = 0x7fc12345u;
  }}
  MemRef2 X=make_2(x,M,N), Y=make_2(y,M,N);
  MemRef1 R=make_1(r,M), S=make_1(s,N);
  _mlir_ciface_kernel_{case}(&X,&R,&S,&Y);
  nh_e6_cache_writes();
  dequantize_snapshot(workspace({diag}));
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  return 0; /* Transport success only: numerical verification is on host. */
}}
"""
    (directory / "launch.c").write_text(launch)
    support = (ROOT / "tools/board_support.c").read_text()
    support = support.replace(
        "int ok = done == NH_E6_DONE_PASS && validation_executed &&",
        "int ok = done == NH_E6_DONE_PASS &&",
    )
    support = support.replace(
        "checked == VIT_BOARD_EXPECTED_CHECKED && !errors;", "!errors;"
    )

    def export(name, offset, dtype, shape):
        record = dict(
            version=1,
            name=name,
            address=hex(0x80100000 + offset),
            nbytes=int(np.prod(shape)) * 4,
            dtype=dtype,
            shape=shape,
        )
        line = "@@FPGA_OUTPUT " + json.dumps(record, separators=(",", ":")) + "\r\n"
        return "    print_uart(" + json.dumps(line) + ");\n"

    exports = "".join(
        (
            export("C", 64, "<f4", [m, n]),
            export("Before", 0, "<u4", [16]),
            export("After", tail, "<u4", [16]),
            export("D", diag, "<u4", [4, 16]),
        )
    )
    support = support.replace(
        "    print_uart(ok\n",
        "    if (ok) {\n" + exports + "    }\n    print_uart(ok\n",
    )
    (directory / "support.c").write_text(support)
    assembly = (original / "kernel.nr.s").read_text()
    replacements = {
        "vector-store": """\taddi a5, a5, 512
\tvsetivli zero, 1, e32, m1, ta, ma
\tvmv.v.x v16, a4
\tfence rw, rw
\tvse32.v v16, (a5)
\tfence rw, rw
\taddi a5, a5, -512""",
        "scalar-fence": """\tfence iorw, iorw
\tsw a4, 512(a5)
\tfence iorw, iorw""",
        "scalar-write-policy": """\tcsrr t0, 0xbc0
\tli t1, 0x0f0f0f0f0f0f0f0f
\tfence iorw, iorw
\tcsrw 0xbc0, t1
\tsw a4, 512(a5)
\tfence iorw, iorw
\tcsrw 0xbc0, t0""",
    }
    if variant != "original":
        replacement = replacements[variant]
        assembly, changes = re.subn(r"\tsw\s+a4, 512\(a5\)", replacement, assembly)
        assert changes == (2 if m == 1 else 1), changes
        assert not re.search(r"\bv16\b", (original / "kernel.nr.s").read_text())
        # t0/t1 are caller-saved and dead in the numerical kernel. The ABI
        # wrapper below .Lfunc_end0 has its own unrelated uses.
        if variant == "scalar-write-policy":
            body = (original / "kernel.nr.s").read_text().split(".Lfunc_end0:")[0]
            assert not re.search(r"\bt[01]\b", body)
    (directory / "kernel.s").write_text(assembly)
    snapshot = [
        ".text",
        ".globl dequantize_snapshot",
        "dequantize_snapshot:",
        "vsetivli zero,16,e32,m4,ta,ma",
    ]
    for address in (0x20400200, 0x20400000, 0x20000000, 0x20000200):
        snapshot += [
            f"li t0,{address}",
            "fence rw,rw",
            "vle32.v v8,(t0)",
            "fence rw,rw",
            "vse32.v v8,(a0)",
            "fence rw,rw",
            "addi a0,a0,64",
        ]
    snapshot += ["ret", '.section .note.GNU-stack,"",@progbits']
    (directory / "snapshot.S").write_text("\n".join(snapshot) + "\n")
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
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
        "-DBOSC_AME_DMA_ASSERT(x)=",
    ]
    flags += [
        "-I" + str(p)
        for p in (ROOT, ROOT / "tools", board, platform, runtime, uart, nr)
    ]
    log = directory / "build.log"
    for source, obj in [("launch.c", "launch.o"), ("support.c", "support.o")]:
        command(
            [llvm / "clang", *flags, "-c", directory / source, "-o", directory / obj],
            log,
        )
    for source in ("kernel.s", "snapshot.S"):
        command(
            [
                llvm / "clang",
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdcv_zicsr_zifencei",
                "-mabi=lp64d",
                "-mno-relax",
                "-c",
                directory / source,
                "-o",
                directory / (Path(source).stem + ".o"),
            ],
            log,
        )
    objects = [
        directory / "launch.o",
        directory / "support.o",
        directory / "kernel.o",
        directory / "snapshot.o",
    ]
    objects += [
        board / (name + ".o")
        for name in ("adapter", "wrapper", "boot", "platform", "uart", "dma", "math")
    ]
    command(
        [
            shutil.which("ld.lld"),
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            ROOT / "board.ld",
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
            buddy / "examples/FPGA-BOSCAME/tools/check_nr_elf.py",
            directory / "image.elf",
            "--objdump",
            llvm / "llvm-objdump",
            "--output",
            directory / "elf-audit.json",
        ],
        log,
    )
    command(
        [
            llvm / "llvm-objcopy",
            "-O",
            "binary",
            directory / "image.elf",
            directory / "image.bin",
        ],
        log,
    )
    (directory / "image.sha256").write_text(
        hashlib.sha256((directory / "image.bin").read_bytes()).hexdigest()
        + "  image.bin\n"
    )
    save(
        directory / "build.json",
        dict(
            case=case,
            variant=variant,
            shape=[m, n],
            image_sha256=hashlib.sha256(
                (directory / "image.bin").read_bytes()
            ).hexdigest(),
            original_kernel_sha256=hashlib.sha256(
                (original / "kernel.nr.s").read_bytes()
            ).hexdigest(),
            output_offset=64,
            guard_after_offset=tail,
            snapshot_offset=diag,
            snapshot_rows=[
                "row_scale_spm1_512",
                "x_spm1_0",
                "column_spm0_0",
                "result_spm0_512",
            ],
        ),
    )
    return directory


def run(directory):
    from fpga_scheduler import Client

    config = json.loads((directory / "build.json").read_text())
    with Client(
        "http://127.0.0.1:18765",
        token_file=str(ROOT.parent / "client.token"),
        request_timeout=360,
    ) as client:
        with client.run_test(str(directory / "image.bin"), timeout=120) as stream:
            stream.on_update = lambda t: print(
                f"[FPGA] {t.id} {t.state} position={t.position}", flush=True
            )
            with (directory / "console.log").open("w") as log:
                try:
                    for chunk in stream:
                        print(chunk, end="", flush=True)
                        log.write(chunk)
                        log.flush()
                except Exception:
                    save(directory / "task.json", asdict(client.job(stream.id)))
                    raise
        task = client.job(stream.id)
        save(directory / "task.json", asdict(task))
        result = stream.result
        uart = result.uart_output if result.uart_output is not None else result.output
        (directory / "uart.log").write_text(uart)
        assert result.details["image_sha256"] == config["image_sha256"]
        if not result.ok:
            raise RuntimeError(
                f"Task {stream.id}: {result.execution_status} {result.details}"
            )
        save(directory / "artifacts.json", client.artifacts(stream.id))
        for name in ("C", "Before", "After", "D"):
            client.download_artifact(stream.id, name, directory / (name + ".bin"))
    return verify(directory)


def verify(directory):
    config = json.loads((directory / "build.json").read_text())
    task = json.loads((directory / "task.json").read_text())
    result = task["result"]
    assert task["state"] == "completed" and result["exit_code"] == 0
    assert result["execution_status"] == "exited" and result["app_exit_code"] == 0
    assert result["details"]["image_sha256"] == config["image_sha256"]
    assert (
        hashlib.sha256((directory / "image.bin").read_bytes()).hexdigest()
        == config["image_sha256"]
    )
    uart = (directory / "uart.log").read_text()
    m, n = config["shape"]
    artifacts = {
        item["name"]: item
        for item in json.loads((directory / "artifacts.json").read_text())
    }
    expected_artifacts = {
        "C": (m * n * 4, "<f4", [m, n], config["output_offset"]),
        "Before": (64, "<u4", [16], 0),
        "After": (64, "<u4", [16], config["guard_after_offset"]),
        "D": (256, "<u4", [4, 16], config["snapshot_offset"]),
    }
    for name, (size, dtype, shape, offset) in expected_artifacts.items():
        metadata = artifacts[name]
        payload = (directory / (name + ".bin")).read_bytes()
        assert len(payload) == metadata["nbytes"] == size, name
        assert metadata["dtype"] == dtype and metadata["shape"] == shape, name
        assert int(metadata["address"], 0) == 0x80100000 + offset, name
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"], name
    cycles = [int(v, 16) for v in re.findall(r"kernel_cycles=0x([0-9a-fA-F]+)", uart)]
    assert len(cycles) == 1 and cycles[0] > 0, cycles
    assert re.search(r"kernel_returned=1\b", uart), "Incomplete invocation"
    indices = np.arange(m * n, dtype=np.int64).reshape(m, n)
    x = ((indices % 1023 - 511) * 7919).astype(np.int32)
    row = (np.arange(m, dtype=np.int32) % 17 + 1).astype(np.float32) * np.float32(
        0.00390625
    )
    col = (np.arange(n, dtype=np.int32) % 31 + 1).astype(np.float32) * np.float32(
        0.001953125
    )
    expected = (x.astype(np.float32) * row[:, None]) * col[None, :]
    actual = np.fromfile(directory / "C.bin", dtype="<f4").reshape(m, n)
    delta = np.abs(actual - expected)
    wrong = ~(delta <= np.float32(1e-5) + np.float32(1e-6) * np.abs(expected))
    positions = np.flatnonzero(wrong)[:12]
    guards = all(
        np.all(np.fromfile(directory / (name + ".bin"), dtype="<u4") == 0x513A7E29)
        for name in ("Before", "After")
    )
    d = np.fromfile(directory / "D.bin", dtype="<u4").reshape(4, 16)
    expected_scale = row[-1:].view(np.uint32).item()
    record = dict(
        config,
        task_id=task["id"],
        node=task["node"],
        fpga=task["fpga"],
        execution_status=result["execution_status"],
        app_exit_code=result["app_exit_code"],
        checked=m * n,
        errors=int(wrong.sum()),
        bitwise_differences=int(
            np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))
        ),
        max_abs_error=float(np.nanmax(delta)),
        guards_ok=bool(guards),
        status="PASS" if guards and not wrong.any() else "FAIL",
        kernel_cycles=cycles,
        artifact_integrity_ok=True,
        expected_final_row_scale_bits=f"{expected_scale:08x}",
        observed_final_row_scale_bits=f"{d[0, 0]:08x}",
        row_scale_snapshot_ok=bool(d[0, 0] == expected_scale),
        expected_sha256=hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
        output_sha256=hashlib.sha256((directory / "C.bin").read_bytes()).hexdigest(),
        tolerance=dict(atol=1e-5, rtol=1e-6),
        input_rounds=1,
        snapshot_hex=[[f"{v:08x}" for v in r] for r in d],
        first_mismatches=[
            dict(
                index=int(i),
                got=float(actual.flat[i]),
                expected=float(expected.flat[i]),
                got_bits=f"{actual.view(np.uint32).flat[i]:08x}",
                expected_bits=f"{expected.view(np.uint32).flat[i]:08x}",
            )
            for i in positions
        ],
        details=result["details"],
    )
    save(directory / "result.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "case",
                    "variant",
                    "task_id",
                    "status",
                    "checked",
                    "errors",
                    "bitwise_differences",
                    "max_abs_error",
                    "guards_ok",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", required=True)
    p.add_argument(
        "--variant",
        choices=["original", "vector-store", "scalar-fence", "scalar-write-policy"],
        default="original",
    )
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--scheduler-root", type=Path, required=True)
    p.add_argument("--build-only", action="store_true")
    p.add_argument(
        "--collect",
        action="store_true",
        help="Verify saved DDR files without submitting",
    )
    args = p.parse_args()
    sys.path.insert(0, str(args.scheduler_root / "src"))
    failed = False
    for case in args.case:
        if args.collect:
            record = verify(args.report_dir.resolve() / args.variant / case)
            failed |= record["status"] != "PASS"
            continue
        directory = args.report_dir.resolve() / args.variant / case
        if (directory / "task.json").exists():
            p.error(
                f"{directory} already contains task evidence; use --collect "
                "or a new --report-dir for an intentional new run"
            )
        directory = build(case, args.variant, args.report_dir.resolve())
        if not args.build_only:
            failed |= run(directory)["status"] != "PASS"
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())

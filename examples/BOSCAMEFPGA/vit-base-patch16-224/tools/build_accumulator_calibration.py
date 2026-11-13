"""Build a local V2 DDR/SPM/ACC calibration image; never submit board jobs.

DDR and explicit fixed-SPM kernels have two terminal outputs. The ACC kernel
has one output; its measured adapter copies C into the second snapshot. Thus
all routes satisfy the same observable contract, including the copy cost.
This tool does not assert that the compiler fuses a two-output source graph.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BUDDY = REPO / "buddy-mlir"
VIT = REPO / "examples/BOSCAMEFPGA/vit-base-patch16-224"
TOOLS = BUDDY / "examples/FPGA-BOSCAME/tools"
LLVM = BUDDY / "llvm/build/bin"
RUNTIME = BUDDY / "runtime/boscame"
PRIVATE_DDR_RUNTIME_FLAGS = ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"]
_spec = importlib.util.spec_from_file_location(
    "hot", BUDDY / "tests/Conversion/LowerLinalgToBOSCAME/PlanHotAccumulator.py"
)
_hot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hot)
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(value):
    return value if value < 128 else value - 256


def a_value(row, q, stress):
    return (
        [127, -128, -1, 0, 1][(row * 3 + q) % 5]
        if stress
        else signed((row * 37 + q * 13 + 7) % 256)
    )


def b_value(column, q, stress):
    return (
        [-128, 127, 1, -1, 0][(column * 7 + q * 3) % 5]
        if stress
        else signed((column * 17 + q * 11 + 211) % 256)
    )


def old_c_value(index, stress):
    return ((0x7FFFFF00 if stress else 0x6A020000) + index * 17) & 0xFFFFFFFF


def simulate(out, trips, golden):
    """Check raw kernels plus the ACC adapter against an independent oracle."""
    k = trips * 128
    reports = []
    for stress in (False, True):
        for variant in ("ddr", "spm", "acc"):
            a = bytearray(
                a_value(r, q, stress) & 255 for r in range(32) for q in range(k)
            ) + bytearray([0xA1] * 64)
            b = bytearray(
                b_value(n, q, stress) & 255 for n in range(64) for q in range(k)
            ) + bytearray([0xB2] * 64)
            c = bytearray().join(
                struct.pack("<I", old_c_value(q, stress)) for q in range(32 * k)
            ) + bytearray([0xC3] * 64)
            snapshot = bytearray([0xD4] * (8192 + 64))
            saved_a, saved_b = bytes(a), bytes(b)
            expected = bytearray(c)
            gold = golden[int(stress) * 8192 : (int(stress) + 1) * 8192]
            for r in range(32):
                expected[r * k * 4 : r * k * 4 + 256] = gold[r * 256 : (r + 1) * 256]
            machine = Machine(out / f"hot_{variant}.emitted.mlir", f"hot_{variant}")
            machine.private_poison = 0x5A if stress else 0xA5
            machine.launch({"%arg0": a, "%arg1": b, "%arg2": c, "%arg3": snapshot}, {})
            assert c == expected, (stress, variant, "C/row padding/guard")
            if variant == "acc":
                assert snapshot == bytearray([0xD4] * (8192 + 64))
                for r in range(32):
                    snapshot[r * 256 : (r + 1) * 256] = c[r * k * 4 : r * k * 4 + 256]
            assert snapshot == gold + bytearray([0xD4] * 64), (
                stress,
                variant,
                "snapshot/guard",
            )
            assert bytes(a) == saved_a and bytes(b) == saved_b, (
                stress,
                variant,
                "input",
            )
            traffic = {}
            for _, begin, end, access in machine.ame_frame_accesses:
                traffic[access] = traffic.get(access, 0) + end - begin
            if variant == "ddr":
                assert len(machine.frame_log) == 2 and not machine.private_top
                assert traffic == {"read": trips * 8192, "write": trips * 8192}, traffic
            else:
                assert not machine.frame_log and not traffic
            reports.append(
                {
                    "variant": variant,
                    "stress": stress,
                    "status": "PASS",
                    "ame_frame_bytes": traffic,
                    "frame_log": machine.frame_log,
                    "actual_outputs_sha256": hashlib.sha256(c + snapshot).hexdigest(),
                }
            )
    (out / "simulation-report.json").write_text(json.dumps(reports, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=int, choices=[6, 24], default=6)
    parser.add_argument(
        "--stress",
        type=int,
        choices=[0, 1],
        help="build eight ABAB calls for one data regime; default is all sixteen",
    )
    parser.add_argument(
        "--post-add",
        action="store_true",
        help="calibrate normal old-C plus product, including the mixed SPM epilogue",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="audit route objects and simulate all routes without linking a board image",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--buddy-opt", type=Path, default=BUDDY / "build/bin/buddy-opt")
    parser.add_argument(
        "--scheduler-root",
        type=Path,
        default=REPO.parent / "fpga-scheduler",
        help="scheduler checkout containing runtime/include/fpga_output.h",
    )
    parser.add_argument(
        "--ddr-opt",
        type=Path,
        help="frozen V2 DDR reference tool, before automatic hot-P placement",
    )
    parser.add_argument(
        "--spm-mode",
        choices=("fixed", "automatic"),
        default="fixed",
        help="explicit physical P constraint, or measured automatic candidate",
    )
    args = parser.parse_args()
    stresses = (0, 1) if args.stress is None else (args.stress,)
    calls = len(stresses) * 8
    stress_schedule = [stress for stress in stresses for _ in range(8)]
    out = args.out.resolve()
    for filename in ("manifest.json", "preparation-manifest.json"):
        previous = out / filename
        if previous.exists():
            identity = json.loads(previous.read_text())
            if (
                identity.get("trips") != args.trips
                or identity.get("post_add", False) != args.post_add
                or identity.get("spm_mode", "fixed") != args.spm_mode
                or identity.get("stress") != args.stress
            ):
                raise RuntimeError(
                    "output directory contains a different K length, accumulator mode or SPM selection"
                )
    if args.prepare_only and any(
        (out / filename).exists()
        for filename in ("acc-path-calibration.elf", "acc-path-calibration.bin")
    ):
        raise RuntimeError(
            "prepare-only requires an output directory without board images"
        )
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "tools"
    frozen.mkdir(exist_ok=True)
    opt = frozen / "buddy-opt"
    ddr_opt = frozen / "buddy-opt-ddr" if args.ddr_opt else opt
    translate = frozen / "buddy-translate"
    freeze = [
        (args.buddy_opt, opt),
        (BUDDY / "build/bin/buddy-translate", translate),
    ]
    if args.ddr_opt:
        freeze.append((args.ddr_opt, ddr_opt))
    for original, destination in freeze:
        if destination.exists() and digest(original) != digest(destination):
            raise RuntimeError(
                "output directory contains different frozen tools: " + str(destination)
            )
        if not destination.exists():
            shutil.copy2(original, destination)
    commands = []

    def run(stage, command, *, stdin=None):
        command = [str(arg) for arg in command]
        result = subprocess.run(command, input=stdin, text=True, capture_output=True)
        (out / (stage + ".log")).write_text(result.stdout + result.stderr)
        commands.append({"stage": stage, "argv": command, "exit": result.returncode})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(stage + ": " + result.stderr[-5000:])
        return result.stdout

    objects = []
    variants = []
    for name in ("ddr", "spm", "acc"):
        version = 2
        route_opt = ddr_opt if name == "ddr" else opt
        stem = "hot_" + name
        src = out / (stem + ".mlir")
        source = _hot.source(args.trips, observe=name != "acc", post_add=args.post_add)
        if name == "spm" and args.spm_mode == "fixed":
            source = _hot.fixed_spm(source)
        src.write_text(source.replace("@matrix(", "@" + stem + "("))
        planned = out / (stem + ".planned.mlir")
        emitted = out / (stem + ".emitted.mlir")
        if name == "spm" and args.spm_mode == "fixed":
            # Supported target-input adapter imports and verifies the physical
            # window constraint; no scheduling metadata is edited by this tool.
            run(
                stem + ".fixed-input",
                [
                    route_opt,
                    src,
                    "--linalg-to-boscame=plan-version=2 enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga",
                    "-o",
                    emitted,
                ],
            )
        else:
            run(
                stem + ".plan",
                [
                    route_opt,
                    src,
                    "--kernel-contract-annotate",
                    f"--bosc-ame-plan=plan-version={version}",
                    "--bosc-ame-plan-verify",
                    "-o",
                    planned,
                ],
            )
            run(stem + ".emit", [route_opt, planned, "--bosc-ame-emit", "-o", emitted])
        body = emitted.read_text()
        if name != "acc":
            assert "bosc_ame.register_schedule" not in body
            assert (
                "bosc_ame.tile.load_acc" in body and "bosc_ame.tile.store_acc" in body
            )
        else:
            assert "bosc_ame.register_schedule" in body
            assert body.count("bosc_ame.tile.load_acc") == int(args.post_add)
            assert ("bosc_ame.tile.zero_acc" in body) == (not args.post_add)
            assert body.count("bosc_ame.tile.store_acc") == 1
        frame = re.search(r"bosc_ame.ddr.frame[^\n]*memref<(\d+)xi8>", body)
        old_c_workspace = args.post_add and name != "acc"
        assert body.count("memref.alloca") == int(old_c_workspace), body
        if old_c_workspace:
            assert re.search(r"memref.alloca[^\n]* : memref<32x64xi32>", body), body
            assert "bosc_ame.cpu_ddr_policy" in body, body
        if name == "ddr":
            assert (
                frame and int(frame[1]) == 8192 and "bosc_ame.spm.view" not in body
            ), body
        elif name == "spm":
            assert not frame and body.count("bosc_ame.spm.view") == 1, body
        else:
            assert not frame and "bosc_ame.spm.view" not in body, body
        llvmmlir = out / (stem + ".llvm.mlir")
        ll = out / (stem + ".ll")
        assembly = out / (stem + ".s")
        run(stem + ".llvm", [route_opt, emitted, "--lower-bosc-ame", "-o", llvmmlir])
        run(stem + ".translate", [translate, "--buddy-to-llvmir", llvmmlir, "-o", ll])
        run(
            stem + ".llc",
            [
                LLVM / "llc",
                ll,
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                assembly,
            ],
        )
        encoded = run(
            stem + ".encode",
            [
                "python3",
                TOOLS / "ame_to_word.py",
                "--gpr-mode=fixed",
                "--preserve-matrix-registers",
            ],
            stdin=assembly.read_text(),
        )
        restricted = run(
            stem + ".restrict",
            [
                "python3",
                TOOLS / "restrict_fpga_assembly.py",
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
            ],
            stdin=encoded,
        )
        nr = out / (stem + ".nr.s")
        nr.write_text(restricted)
        obj = out / (stem + ".o")
        run(
            stem + ".object",
            [
                LLVM / "clang",
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdcv_zicsr_zifencei",
                "-mabi=lp64d",
                "-mno-relax",
                "-c",
                nr,
                "-o",
                obj,
            ],
        )
        run(
            stem + ".object-audit",
            [
                "python3",
                TOOLS / "check_nr_elf.py",
                obj,
                "--objdump",
                LLVM / "llvm-objdump",
                "--output",
                out / (stem + ".object-audit.json"),
            ],
        )
        objects.append(obj)
        variants.append(
            {
                "name": name,
                "plan_version": version,
                "buddy_opt_sha256": digest(route_opt),
                "source_sha256": digest(src),
                "planned_sha256": digest(planned) if planned.exists() else None,
                "emitted_sha256": digest(emitted),
                "private_frame_bytes": 8192 if frame else 0,
                "cpu_old_c_workspace_bytes": 8192 if old_c_workspace else 0,
                "spm_windows": body.count("bosc_ame.spm.view"),
                "asm_load_acc": assembly.read_text().count("mlce32.m"),
                "asm_store_acc": assembly.read_text().count("msce32.m"),
                "acc_initialization": ("old-C" if args.post_add else "zero")
                if name == "acc"
                else "materialized-P",
                "p_initialization": None if name == "acc" else "zero",
                "post_loop_old_c_add": old_c_workspace,
                "object_sha256": digest(obj),
                "object_audit_sha256": digest(out / (stem + ".object-audit.json")),
            }
        )

    k = args.trips * 128
    expected = []
    for stress in (False, True):
        for row in range(32):
            for col in range(64):
                value = sum(
                    a_value(row, q, stress) * b_value(col, q, stress) for q in range(k)
                )
                if args.post_add:
                    value += old_c_value(row * k + col, stress)
                expected.append(value & 0xFFFFFFFF)
    golden = out / "golden.bin"
    golden.write_bytes(struct.pack("<" + "I" * len(expected), *expected))
    simulate(out, args.trips, golden.read_bytes())
    if args.prepare_only:
        preparation = {
            "status": "PREPARED_NOT_LINKED",
            "trips": args.trips,
            "k": k,
            "post_add": args.post_add,
            "mode": "normal-post-add" if args.post_add else "overwrite",
            "calls": calls,
            "stress": args.stress,
            "stress_schedule": stress_schedule,
            "spm_mode": args.spm_mode,
            "buddy_opt_sha256": digest(opt),
            "buddy_translate_sha256": digest(translate),
            "golden_sha256": digest(golden),
            "builder_sha256": digest(__file__),
            "variants": variants,
            "simulation_report_sha256": digest(out / "simulation-report.json"),
            "object_audits": {
                name: digest(out / ("hot_" + name + ".object-audit.json"))
                for name in ("ddr", "spm", "acc")
            },
        }
        (out / "preparation-manifest.json").write_text(
            json.dumps(preparation, indent=2) + "\n"
        )
        print(json.dumps(preparation, indent=2))
        return
    goldS = out / "golden.S"
    goldS.write_text(
        '.section .rodata.golden,"a",@progbits\n.balign 64\n.global bench_golden\nbench_golden:\n.incbin "'
        + str(golden)
        + '"\n'
    )
    harness = out / "harness.c"
    harness.write_text((VIT / "tools/accumulator_calibration.c").read_text())
    platform = RUNTIME / "nh_e6"
    dma = RUNTIME / "runtime"
    uart = RUNTIME / "uart"
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
        "-ffunction-sections",
        "-fdata-sections",
        "-DBOSC_AME_DMA_ASSERT(x)=",
        f"-DCALIBRATION_K={k}",
        f"-DCALIBRATION_TRIPS={args.trips}",
        f"-DCALIBRATION_POST_ADD={int(args.post_add)}",
        f"-DCALIBRATION_STRESS={-1 if args.stress is None else args.stress}",
        "-I" + str(args.scheduler_root / "runtime/include"),
        "-I" + str(platform),
        "-I" + str(dma),
        "-I" + str(uart),
    ]
    for name, src, extra in [
        ("harness", harness, []),
        ("boot", platform / "nh_e6_boot.S", []),
        (
            "platform",
            platform / "nh_e6_platform.c",
            ["-Dinit_uart=bench_init_uart", "-DNH_E6_WAIT_ITERATIONS=0xffffffffu"],
        ),
        ("uart", uart / "uart.c", []),
        ("dma", dma / "bosc_ame_dma.c", PRIVATE_DDR_RUNTIME_FLAGS),
        ("golden", goldS, []),
    ]:
        obj = out / (name + ".o")
        run(name + ".compile", [LLVM / "clang", *flags, *extra, "-c", src, "-o", obj])
        objects.append(obj)
    elf = out / "acc-path-calibration.elf"
    binary = out / "acc-path-calibration.bin"
    run(
        "link",
        [
            LLVM / "ld.lld" if (LLVM / "ld.lld").exists() else shutil.which("ld.lld"),
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            VIT / "board.ld",
            "-Map=" + str(out / "acc-path-calibration.map"),
            "-o",
            elf,
            *objects,
        ],
    )
    run(
        "audit",
        [
            "python3",
            TOOLS / "check_nr_elf.py",
            elf,
            "--objdump",
            LLVM / "llvm-objdump",
            "--output",
            out / "elf-audit.json",
        ],
    )
    run("objcopy", [LLVM / "llvm-objcopy", "-O", "binary", elf, binary])
    symbols = run("symbols", [LLVM / "llvm-nm", "--defined-only", "--print-size", elf])
    ledger = re.search(
        r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\w\s+accumulator_calls$",
        symbols,
        re.MULTILINE,
    )
    assert ledger, "missing exported DDR output ledger"
    ledger_address, ledger_bytes = int(ledger[1], 16), int(ledger[2], 16)
    assert ledger_bytes == calls * 4096 * 4
    assert 0x80000000 <= ledger_address <= 0x80400000 - ledger_bytes
    ledger_golden = out / "readmem-golden.bin"
    ledger_golden.write_bytes(
        b"".join(
            golden.read_bytes()[stress * 8192 : (stress + 1) * 8192] * 2
            for stress in stresses
            for _ in range(8)
        )
    )
    manifest = {
        "trips": args.trips,
        "k": k,
        "post_add": args.post_add,
        "mode": "normal-post-add" if args.post_add else "overwrite",
        "calls": calls,
        "stress": args.stress,
        "stress_schedule": stress_schedule,
        "spm_mode": args.spm_mode,
        "private_ddr_runtime": {
            "mode": "nr-single-e6-irq",
            "compile_flags": PRIVATE_DDR_RUNTIME_FLAGS,
            "scope": "bosc_ame_dma.c only",
        },
        "comparison": "V2 neutral DDR versus "
        + ("explicit fixed-SPM" if args.spm_mode == "fixed" else "automatic hot-SPM")
        + " versus full-K ACC; identical two-output observable contract; ACC measured adapter performs second output copy",
        "schedule": "DDR,SPM,DDR,SPM,DDR,ACC,DDR,ACC; "
        + (
            "normal then signed extremes"
            if args.stress is None
            else f"stress={args.stress} only"
        ),
        "measurement": "rdcycle around complete typed adapter + final fence; ACC second output copy included; input initialization/prewarming/oracle excluded; includes frame/cache costs",
        "buddy_opt_sha256": digest(opt),
        "buddy_translate_sha256": digest(translate),
        "golden_sha256": digest(golden),
        "image_sha256": digest(binary),
        "variants": variants,
        "progress_markers": {
            "base": "0x300000 + (call << 8)",
            "phases": {
                "1": "write policy",
                "2": "fills",
                "3": "A/B initialization",
                "4": "old C",
                "5": "private frame poison writes",
                "6": "private frame prewarm",
                "7": "external C prewarm",
                "8": "initialization fence",
                "16": "before measured adapter",
                "17": "after measured adapter",
                "18": "after scalar inspection",
                "19": "after DDR output capture",
            },
        },
        "ddr_outputs": [
            {
                "name": "accumulator_calls",
                "symbol": "accumulator_calls",
                "address": hex(ledger_address),
                "nbytes": ledger_bytes,
                "dtype": "<i4",
                "shape": [calls, 4096],
                "layout": "each row is valid C[32,64] followed by snapshot[32,64]; stress_schedule gives each row's data regime",
                "golden_file": ledger_golden.name,
                "golden_sha256": digest(ledger_golden),
            }
        ],
        "output_runtime_header_sha256": digest(
            args.scheduler_root / "runtime/include/fpga_output.h"
        ),
    }
    rounds = []
    order = ["ddr", "spm", "ddr", "spm", "ddr", "acc", "ddr", "acc"]
    for stress in stresses:
        a = bytes(a_value(r, q, stress) & 255 for r in range(32) for q in range(k))
        b = bytes(b_value(n, q, stress) & 255 for n in range(64) for q in range(k))
        c = bytearray().join(
            struct.pack("<I", old_c_value(q, stress)) for q in range(32 * k)
        )
        inputs_hash = hashlib.sha256(a + b + c).hexdigest()
        gold = golden.read_bytes()[int(stress) * 8192 : (int(stress) + 1) * 8192]
        for r in range(32):
            c[r * k * 4 : r * k * 4 + 256] = gold[r * 256 : (r + 1) * 256]
        outputs_hash = hashlib.sha256(c + gold).hexdigest()
        rounds.extend(
            {
                "route": route,
                "stress": stress,
                "inputs_sha256": inputs_hash,
                "output_sha256": outputs_hash,
            }
            for route in order
        )
    (out / "accumulator-manifest.json").write_text(
        json.dumps(
            {
                "shape": [32, 64, k],
                "post_add": args.post_add,
                "mode": "normal-post-add" if args.post_add else "overwrite",
                "calls": calls,
                "stress": args.stress,
                "stress_schedule": stress_schedule,
                "orders": [order for _ in stresses],
                "rounds": rounds,
            },
            indent=2,
        )
        + "\n"
    )
    manifest["observable_contract"] = (
        "C[32,64] with row pitch K and snapshot[32,64] both equal modulo-i32 "
        + ("old C plus " if args.post_add else "")
        + "signed-i8 product; C padding and input bytes unchanged"
    )
    manifest["acc_adapter"] = (
        "unique-output generated ACC kernel followed by uncached C-to-stack reads, write-through stack-to-snapshot stores inside measurement; includes copy and both policy transitions"
    )
    manifest["spm_constraint"] = (
        "explicit bank0 offset0 private P; physical-source compatibility adapter, plan-version2; this is not automatic profitability selection"
        if args.spm_mode == "fixed"
        else "neutral source with automatic V2 hot-P placement; identical source except symbol to the DDR reference"
    )
    manifest["source_inputs"] = {
        str(path.relative_to(REPO)): digest(path)
        for path in (
            Path(__file__),
            VIT / "tools/accumulator_calibration.c",
            VIT / "board.ld",
            BUDDY / "tests/Conversion/LowerLinalgToBOSCAME/PlanHotAccumulator.py",
            BUDDY / "tests/Conversion/LowerLinalgToBOSCAME/PlanRegisterSchedule.py",
            platform / "nh_e6_boot.S",
            platform / "nh_e6_platform.c",
            platform / "nh_e6_platform.h",
            dma / "bosc_ame_dma.c",
            dma / "bosc_ame_dma.h",
            uart / "uart.c",
            uart / "uart.h",
            TOOLS / "ame_to_word.py",
            TOOLS / "restrict_fpga_assembly.py",
            TOOLS / "boscame_buffer_sim.py",
        )
    }
    manifest["backend_tools"] = {
        name: digest(LLVM / name)
        for name in ("llc", "clang", "llvm-objdump", "llvm-objcopy", "llvm-nm")
    }
    manifest["linker_sha256"] = digest(
        LLVM / "ld.lld" if (LLVM / "ld.lld").exists() else shutil.which("ld.lld")
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(binary)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

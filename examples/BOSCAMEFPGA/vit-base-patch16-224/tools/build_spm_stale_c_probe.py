"""Build an independent normal-SPM stale-old-C probe; never submit board jobs."""

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import build_accumulator_calibration as calibration

K = 768
CALLS = 4
SEED_SOURCE = """module attributes {bosc_ame.target = "nr-fpga", bosc_ame.dma.abi = "args_aligned64_nonalias", contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}} {
  func.func @ame_seed(%arg0: memref<*xi32>, %arg1: memref<*xi32>) {
    %seed = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<*xi32> to memref<32x64xi32>
    %output = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 64], strides: [768, 1] : memref<*xi32> to memref<32x64xi32, strided<[768, 1]>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<32x64xi32, #bosc_ame.spm<0>>
    %value = bosc_ame.tile.load_acc %seed {bosc_ame.direct_ddr} : memref<32x64xi32> -> vector<32x64xi32>
    bosc_ame.tile.store_acc %value, %spm : vector<32x64xi32>, memref<32x64xi32, #bosc_ame.spm<0>>
    bosc_ame.tile.store_acc %value, %output {bosc_ame.direct_ddr} : vector<32x64xi32>, memref<32x64xi32, strided<[768, 1]>>
    return
  }
}
"""


def simulate(out, golden):
    """Check exact seed/SPM/target effects; the simulator does not model cache."""
    seed_machine = calibration.Machine(out / "seed.mlir", "ame_seed")
    target = calibration.Machine(out / "hot_spm.emitted.mlir", "hot_spm")
    target.spm = seed_machine.spm
    report = []
    for call in range(CALLS):
        stress = call >= CALLS // 2
        a = bytearray([0xA1] * 64)
        a.extend(
            calibration.a_value(r, q, stress) & 255 for r in range(32) for q in range(K)
        )
        a.extend([0xA1] * 64)
        b = bytearray([0xB2] * 64)
        b.extend(
            calibration.b_value(n, q, stress) & 255 for n in range(64) for q in range(K)
        )
        b.extend([0xB2] * 64)
        c = bytearray([0xC3] * 64)
        c.extend(
            b"".join(
                struct.pack("<I", calibration.old_c_value(q, stress))
                for q in range(32 * K)
            )
        )
        c.extend([0xC3] * 64)
        original_c = bytearray(c)
        snapshot = bytearray([0xD4] * (8192 + 128))
        seed = bytearray([0xE5] * 64)
        seed.extend(
            b"".join(
                struct.pack("<I", calibration.old_c_value(r * K + n, stress))
                for r in range(32)
                for n in range(64)
            )
        )
        seed.extend([0xE5] * 64)
        saved = bytes(a), bytes(b), bytes(seed)
        for r in range(32):
            for n in range(64):
                value = calibration.old_c_value(r * K + n, stress)
                struct.pack_into(
                    "<I",
                    c,
                    64 + 4 * (r * K + n),
                    value ^ (0x5555AAAA ^ (call * 0x11111111)),
                )
        for bank in seed_machine.spm:
            bank[:] = bytes([0x69 ^ call]) * len(bank)
        seed_machine.launch(
            {"%arg0": memoryview(seed)[64:], "%arg1": memoryview(c)[64:]}
        )
        assert c == original_c, "seed copy/row padding/guards"
        assert seed_machine.spm[0][:8192] == seed[64:-64], "dirty P seed"
        target.launch(
            {
                "%arg0": memoryview(a)[64:],
                "%arg1": memoryview(b)[64:],
                "%arg2": memoryview(c)[64:],
                "%arg3": memoryview(snapshot)[64:],
            }
        )
        gold = golden[int(stress) * 8192 : (int(stress) + 1) * 8192]
        expected_c = original_c
        for r in range(32):
            expected_c[64 + r * K * 4 : 64 + r * K * 4 + 256] = gold[
                r * 256 : (r + 1) * 256
            ]
        assert c == expected_c, "target C/row padding/guards"
        assert snapshot == bytes([0xD4]) * 64 + gold + bytes([0xD4]) * 64
        assert (bytes(a), bytes(b), bytes(seed)) == saved, "inputs/seed/guards"
        assert not target.frame_log and not target.private_top
        report.append(
            {
                "call": call,
                "stress": stress,
                "status": "PASS",
                "actual_outputs_sha256": hashlib.sha256(c + snapshot).hexdigest(),
                "cache_model": "none; physical cached control is required",
            }
        )
    (out / "simulation-report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.calibration.resolve()
    out = args.out.resolve()
    if out.exists():
        raise RuntimeError("use a new probe output directory")
    manifest = json.loads((baseline / "manifest.json").read_text())
    assert manifest["mode"] == "normal-post-add" and manifest["post_add"] is True
    assert manifest["k"] == K and manifest["trips"] == 6
    assert manifest["spm_mode"] == "fixed"
    spm = next(v for v in manifest["variants"] if v["name"] == "spm")
    assert spm["spm_windows"] == 1 and spm["private_frame_bytes"] == 0
    assert calibration.digest(baseline / "hot_spm.o") == spm["object_sha256"]
    assert (
        calibration.digest(baseline / "hot_spm.emitted.mlir") == spm["emitted_sha256"]
    )
    assert calibration.digest(baseline / "golden.bin") == manifest["golden_sha256"]
    assert (
        calibration.digest(baseline / "tools/buddy-opt") == manifest["buddy_opt_sha256"]
    )
    assert (
        calibration.digest(baseline / "tools/buddy-translate")
        == manifest["buddy_translate_sha256"]
    )
    out.mkdir(parents=True)
    copied = [
        "hot_spm.o",
        "hot_spm.emitted.mlir",
        "boot.o",
        "platform.o",
        "uart.o",
        "dma.o",
        "golden.o",
        "golden.bin",
    ]
    for name in copied:
        shutil.copy2(baseline / name, out / name)
    linker = calibration.VIT / "board.ld"
    assert (
        calibration.digest(linker)
        == manifest["source_inputs"][str(linker.relative_to(calibration.REPO))]
    )
    shutil.copy2(linker, out / "board.ld")
    harness_source = Path(__file__).with_name("spm_stale_c_probe.c")
    shutil.copy2(harness_source, out / "harness.c")
    commands = []

    def run(stage, command, *, stdin=None):
        command = list(map(str, command))
        result = subprocess.run(command, input=stdin, text=True, capture_output=True)
        (out / (stage + ".log")).write_text(result.stdout + result.stderr)
        commands.append({"stage": stage, "argv": command, "exit": result.returncode})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(stage + ": " + result.stderr[-5000:])
        return result.stdout

    opt, translate = baseline / "tools/buddy-opt", baseline / "tools/buddy-translate"
    llvm, tools = calibration.LLVM, calibration.TOOLS
    (out / "seed.mlir").write_text(SEED_SOURCE)
    run(
        "seed.lower",
        [opt, out / "seed.mlir", "--lower-bosc-ame", "-o", out / "seed.llvm.mlir"],
    )
    run(
        "seed.translate",
        [translate, "--buddy-to-llvmir", out / "seed.llvm.mlir", "-o", out / "seed.ll"],
    )
    run(
        "seed.llc",
        [
            llvm / "llc",
            out / "seed.ll",
            "-O2",
            "-filetype=asm",
            "-mtriple=riscv64",
            "-target-abi=lp64d",
            "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
            "-code-model=medium",
            "-o",
            out / "seed.s",
        ],
    )
    encoded = run(
        "seed.encode",
        [
            "python3",
            tools / "ame_to_word.py",
            "--gpr-mode=fixed",
            "--preserve-matrix-registers",
        ],
        stdin=(out / "seed.s").read_text(),
    )
    restricted = run(
        "seed.restrict",
        [
            "python3",
            tools / "restrict_fpga_assembly.py",
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
        ],
        stdin=encoded,
    )
    (out / "seed.nr.s").write_text(restricted)
    run(
        "seed.object",
        [
            llvm / "clang",
            "--target=riscv64-unknown-elf",
            "-march=rv64imafdcv_zicsr_zifencei",
            "-mabi=lp64d",
            "-mno-relax",
            "-c",
            out / "seed.nr.s",
            "-o",
            out / "seed.o",
        ],
    )
    assert "csrw\t0xbc0" not in (out / "seed.s").read_text()
    baseline_commands = json.loads((baseline / "commands.json").read_text())
    harness_command = next(
        c["argv"] for c in baseline_commands if c["stage"] == "harness.compile"
    )[:]
    harness_command[harness_command.index("-c") + 1] = str(out / "harness.c")
    harness_command[harness_command.index("-o") + 1] = str(out / "harness.o")
    run("harness.compile", harness_command)
    elf, binary = out / "spm-stale-c-probe.elf", out / "spm-stale-c-probe.bin"
    linker_tool = next(c["argv"][0] for c in baseline_commands if c["stage"] == "link")
    run(
        "link",
        [
            linker_tool,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            out / "board.ld",
            "-Map=" + str(out / "spm-stale-c-probe.map"),
            "-o",
            elf,
            *[
                out / n
                for n in (
                    "hot_spm.o",
                    "seed.o",
                    "harness.o",
                    "boot.o",
                    "platform.o",
                    "uart.o",
                    "dma.o",
                    "golden.o",
                )
            ],
        ],
    )
    for name, artifact in (("seed", out / "seed.o"), ("elf", elf)):
        run(
            name + ".audit",
            [
                "python3",
                tools / "check_nr_elf.py",
                artifact,
                "--objdump",
                llvm / "llvm-objdump",
                "--output",
                out / (name + "-audit.json"),
            ],
        )
    run("objcopy", [llvm / "llvm-objcopy", "-O", "binary", elf, binary])
    disassembly = run("disassembly", [llvm / "llvm-objdump", "-d", elf])
    (out / "spm-stale-c-probe.elf.disassembly.txt").write_text(disassembly)
    assert not re.search(r"\b(?:lr\.[wd]|sc\.[wd]|amo\w+\.[wd])\b", disassembly)
    symbols = run("symbols", [llvm / "llvm-nm", "--defined-only", "--print-size", elf])
    ledger = re.search(
        r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\w\s+accumulator_calls$",
        symbols,
        re.MULTILINE,
    )
    assert ledger
    address, nbytes = int(ledger[1], 16), int(ledger[2], 16)
    assert nbytes == CALLS * 4096 * 4 and 0x80100000 <= address <= 0x80400000 - nbytes
    golden = (out / "golden.bin").read_bytes()
    (out / "readmem-golden.bin").write_bytes(
        b"".join(
            golden[
                int(call >= CALLS // 2) * 8192 : (int(call >= CALLS // 2) + 1) * 8192
            ]
            * 2
            for call in range(CALLS)
        )
    )
    simulate(out, golden)
    result = {
        "kind": "normal-spm-stale-c-correctness-probe",
        "mode": "normal-post-add",
        "post_add": True,
        "shape": [32, 64, K],
        "calls": CALLS,
        "image_sha256": calibration.digest(binary),
        "calibration_manifest_sha256": calibration.digest(baseline / "manifest.json"),
        "target_object_sha256": calibration.digest(out / "hot_spm.o"),
        "target_emitted_sha256": calibration.digest(out / "hot_spm.emitted.mlir"),
        "seed_source_sha256": calibration.digest(out / "seed.mlir"),
        "seed_object_sha256": calibration.digest(out / "seed.o"),
        "builder_sha256": calibration.digest(__file__),
        "harness_sha256": calibration.digest(out / "harness.c"),
        "objects": {n: calibration.digest(out / n) for n in copied if n.endswith(".o")},
        "simulation_report_sha256": calibration.digest(out / "simulation-report.json"),
        "checks": "four dirty calls, normal then signed extremes; complete outputs, inputs, seed, row padding and guards; zero frame acquisitions",
        "cache_control": "write-policy cached prewarm of wrong sentinel; AME seed writes correct old C and dirties bank0 P; cached control classifies stale/current/other before unchanged normal-SPM target; no harness read-policy transition before target",
        "interpretation": "numerical PASS alone is insufficient to claim stale-cache exposure; require at least one reported cached_stale>0 with zero prewarm_errors/cached_other",
        "measurement": "correctness fixture, not a profitability calibration; seed and cached control excluded from target cycle counter",
        "ddr_outputs": [
            {
                "name": "accumulator_calls",
                "symbol": "accumulator_calls",
                "address": hex(address),
                "nbytes": nbytes,
                "dtype": "<i4",
                "shape": [CALLS, 4096],
                "golden_file": "readmem-golden.bin",
                "golden_sha256": calibration.digest(out / "readmem-golden.bin"),
            }
        ],
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()

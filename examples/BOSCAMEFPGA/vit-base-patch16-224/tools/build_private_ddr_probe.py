#!/usr/bin/env python3
"""Build the private A/B/P visibility probe; never submit a board job."""

import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def amoswap_allocator(source):
    """Diagnostic candidate: one word AMOSWAP, owner state under its guard."""
    substitutions = (
        (
            "static unsigned char bosc_ame_private_ddr_busy;",
            "static _Alignas(4) uint32_t bosc_ame_private_ddr_busy;",
        ),
        (
            "__atomic_test_and_set(&bosc_ame_private_ddr_busy, __ATOMIC_ACQUIRE)",
            "__atomic_exchange_n(&bosc_ame_private_ddr_busy, UINT32_C(1), __ATOMIC_ACQUIRE)",
        ),
        (
            "  uintptr_t expected = 0;\n  nh_e6_progress(0x7104u);",
            "  nh_e6_progress(0x7104u);\n"
            "  uintptr_t expected = bosc_ame_private_ddr_owner;",
        ),
        (
            "  if (!__atomic_compare_exchange_n(&bosc_ame_private_ddr_owner, &expected,\n"
            "                                   owner, 0, __ATOMIC_ACQ_REL,\n"
            "                                   __ATOMIC_ACQUIRE) &&\n"
            "      expected != owner)",
            "  if (expected && expected != owner)",
        ),
        (
            "  nh_e6_progress(0x7105u);\n  return owner;",
            "  if (!expected)\n    bosc_ame_private_ddr_owner = owner;\n"
            "  nh_e6_progress(0x7105u);\n  return owner;",
        ),
        (
            "__atomic_clear(&bosc_ame_private_ddr_busy, __ATOMIC_RELEASE);",
            "__atomic_store_n(&bosc_ame_private_ddr_busy, UINT32_C(0), __ATOMIC_RELEASE);",
        ),
        (
            "__atomic_store_n(&bosc_ame_private_ddr_owner, 0u, __ATOMIC_RELEASE);",
            "bosc_ame_private_ddr_owner = 0u;",
        ),
    )
    for old, new in substitutions:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def trace_allocator(source, single_e6=False):
    """Instrument a copied runtime only; never edit the production allocator."""

    def before(body, anchor, marker):
        assert body.count(anchor) == 1, anchor
        return body.replace(anchor, f"  nh_e6_progress(0x{marker:x}u);\n" + anchor)

    prefix, acquire = source.split("uint64_t bosc_ame_private_ddr_acquire(", 1)
    acquire, release = acquire.split("void bosc_ame_private_ddr_release(", 1)
    prefix = prefix.replace(
        '#include "bosc_ame_dma.h"',
        '#include "bosc_ame_dma.h"\nextern void nh_e6_progress(uint32_t);',
    )
    start = prefix.index("static uintptr_t bosc_ame_private_ddr_enter(")
    generic = prefix.index("static uintptr_t bosc_ame_private_ddr_enter(", start + 1)
    end = prefix.index("#endif", generic)
    left, body, right = (
        (prefix[:start], prefix[start:generic], prefix[generic:])
        if single_e6
        else (prefix[:generic], prefix[generic:end], prefix[end:])
    )
    if single_e6:
        body = before(body, "  uintptr_t hart_id =", 0x7100)
        body = before(body, "  if (hart_id !=", 0x7101)
        body = before(
            body, "  uintptr_t saved = BOSC_AME_PRIVATE_DDR_IRQ_SAVE();", 0x7110
        )
        body = before(body, "  if (bosc_ame_private_ddr_busy)", 0x7111)
        body = before(body, "  if (bosc_ame_private_ddr_busy)", 0x7102)
        body = before(body, "  uintptr_t owner = hart_id + 1u;", 0x7103)
        body = before(
            body, "  uintptr_t expected = bosc_ame_private_ddr_owner;", 0x7104
        )
    else:
        body = before(
            body, "  uintptr_t owner = BOSC_AME_PRIVATE_DDR_HART_ID() + 1u;", 0x7100
        )
        body = before(body, "  if (!owner ||", 0x7101)
        body = before(body, "  if (!owner ||", 0x7102)
        body = before(body, "  uintptr_t expected = 0;", 0x7103)
        body = before(
            body,
            "  if (!__atomic_compare_exchange_n(&bosc_ame_private_ddr_owner",
            0x7104,
        )
    body = before(body, "  return owner;", 0x7105)
    prefix = left + body + right
    acquire = before(acquire, "  if (!bosc_ame_private_ddr_window_valid())", 0x7106)
    acquire = before(acquire, "  BOSC_AME_DMA_SHARED_WRITES();", 0x7200)
    acquire = before(acquire, "  volatile uint64_t *header =", 0x7201)
    acquire = before(acquire, "  header[0] = BOSC_AME_PRIVATE_DDR_MAGIC;", 0x7202)
    acquire = before(acquire, "  volatile uint64_t *tail =", 0x7203)
    acquire = before(acquire, "  bosc_ame_private_ddr_active = payload;", 0x7204)
    acquire = before(acquire, "  BOSC_AME_DMA_FENCE();", 0x7205)
    acquire = before(acquire, "  bosc_ame_private_ddr_leave();", 0x7206)
    acquire = before(acquire, "  return payload;", 0x7207)
    release = before(
        release, "  uintptr_t owner = bosc_ame_private_ddr_enter(bytes);", 0x7300
    )
    release = before(release, "  BOSC_AME_DMA_SHARED_READS();", 0x7301)
    release = before(release, "  volatile uint64_t *header =", 0x7302)
    release = before(release, "  uintptr_t parent = header[4];", 0x7303)
    release = before(release, "  bosc_ame_private_ddr_leave();", 0x7304)
    return (
        prefix
        + "uint64_t bosc_ame_private_ddr_acquire("
        + acquire
        + "void bosc_ame_private_ddr_release("
        + release
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--buddy-opt", type=Path)
    parser.add_argument("--kernel-markers", action="store_true")
    parser.add_argument("--allocator-markers", action="store_true")
    parser.add_argument(
        "--allocator-lock",
        choices=("production", "amoswap32", "e6-irq"),
        default="production",
    )
    args = parser.parse_args()
    if args.allocator_lock == "amoswap32" and not args.allocator_markers:
        parser.error("a diagnostic allocator lock requires --allocator-markers")
    vit = Path(__file__).resolve().parents[1]
    repo = vit.parents[2]
    buddy = repo / "buddy-mlir"
    llvm = buddy / "llvm/build/bin"
    tools = buddy / "examples/FPGA-BOSCAME/tools"
    out = args.out.resolve()
    if (out / "manifest.json").exists():
        parser.error("output already contains evidence; choose a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "tools"
    frozen.mkdir(exist_ok=True)
    opt = frozen / "buddy-opt"
    translate = frozen / "buddy-translate"
    shutil.copy2(args.buddy_opt or buddy / "build/bin/buddy-opt", opt)
    shutil.copy2(buddy / "build/bin/buddy-translate", translate)
    commands = []

    def run(stage, command, source=None):
        command = list(map(str, command))
        result = subprocess.run(command, input=source, text=True, capture_output=True)
        (out / (stage + ".log")).write_text(result.stdout + result.stderr)
        commands.append({"stage": stage, "argv": command, "exit": result.returncode})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(stage + ": " + result.stderr[-6000:])
        return result.stdout

    # Use the exact native fixture covered by the functional/object regression.
    fixture = buddy / "tests/Conversion/BOSCAME/private-ddr-execution.py"
    source = out / "private.mlir"
    fixture_source = runpy.run_path(str(fixture))["SOURCE"]
    if args.kernel_markers:
        fixture_source = fixture_source.replace(
            "  func.func @private_chain(",
            "  func.func private @nh_e6_progress(i32)\n  func.func @private_chain(",
        )

        def marker(value):
            return (
                f"    %marker{value} = arith.constant {value} : i32\n"
                f"    func.call @nh_e6_progress(%marker{value}) : (i32) -> ()\n"
            )

        for anchor, value in (
            ("    %av = arith.constant", 0x6100),
            ("    scf.for %k =", 0x6200),
            ("    scf.for %i = %c0 to %c2 step %c1 {\n      %v =", 0x6300),
            ("    bosc_ame.ddr.release", 0x6400),
        ):
            assert fixture_source.count(anchor) == 1, anchor
            fixture_source = fixture_source.replace(anchor, marker(value) + anchor)
    source.write_text(fixture_source)
    lowered = out / "private.llvm.mlir"
    ll = out / "private.ll"
    assembly = out / "private.s"
    run("llvm", [opt, source, "--lower-bosc-ame", "-o", lowered])
    run("translate", [translate, "--buddy-to-llvmir", lowered, "-o", ll])
    run(
        "llc",
        [
            llvm / "llc",
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
    body = assembly.read_text()
    if "vlenb" in body or "call\tbosc_ame_ddr_" in body:
        raise RuntimeError("CPU policy transitions still require an RVV ABI spill")
    encoded = run(
        "encode",
        [
            "python3",
            tools / "ame_to_word.py",
            "--gpr-mode=fixed",
            "--preserve-matrix-registers",
        ],
        body,
    )
    restricted = run(
        "restrict",
        [
            "python3",
            tools / "restrict_fpga_assembly.py",
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
        ],
        encoded,
    )
    nr = out / "private.nr.s"
    nr.write_text(restricted)
    kernel = out / "private.o"
    run(
        "object",
        [
            llvm / "clang",
            "--target=riscv64-unknown-elf",
            "-march=rv64imafdcv_zicsr_zifencei",
            "-mabi=lp64d",
            "-mno-relax",
            "-c",
            nr,
            "-o",
            kernel,
        ],
    )
    runtime = buddy / "runtime/boscame"
    platform = runtime / "nh_e6"
    dma = runtime / "runtime"
    uart = runtime / "uart"
    harness = Path(__file__).with_name("private_ddr_probe.c")
    dma_source = dma / "bosc_ame_dma.c"
    if args.allocator_markers:
        dma_source = out / "dma.diagnostic.c"
        diagnostic_runtime = trace_allocator(
            (dma / "bosc_ame_dma.c").read_text(), args.allocator_lock == "e6-irq"
        )
        if args.allocator_lock == "amoswap32":
            diagnostic_runtime = amoswap_allocator(diagnostic_runtime)
        dma_source.write_text(diagnostic_runtime)
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
        "-I" + str(platform),
        "-I" + str(dma),
        "-I" + str(uart),
    ]
    objects = [kernel]
    for name, path, extra in (
        ("harness", harness, []),
        ("boot", platform / "nh_e6_boot.S", []),
        (
            "platform",
            platform / "nh_e6_platform.c",
            ["-Dinit_uart=probe_init_uart", "-DNH_E6_WAIT_ITERATIONS=0xffffffffu"],
        ),
        ("uart", uart / "uart.c", []),
        (
            "dma",
            dma_source,
            ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"]
            if args.allocator_lock == "e6-irq"
            else [],
        ),
    ):
        obj = out / (name + ".o")
        run(name + ".compile", [llvm / "clang", *flags, *extra, "-c", path, "-o", obj])
        objects.append(obj)
    elf = out / "private-ddr-probe.elf"
    binary = out / "private-ddr-probe.bin"
    linker = llvm / "ld.lld"
    run(
        "link",
        [
            linker if linker.exists() else shutil.which("ld.lld"),
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            vit / "board.ld",
            "-Map=" + str(out / "private-ddr-probe.map"),
            "-o",
            elf,
            *objects,
        ],
    )
    run(
        "audit",
        [
            "python3",
            tools / "check_nr_elf.py",
            elf,
            "--objdump",
            llvm / "llvm-objdump",
            "--output",
            out / "elf-audit.json",
        ],
    )
    run("objcopy", [llvm / "llvm-objcopy", "-O", "binary", elf, binary])
    busy_path = {
        "production": "LR/SC",
        "amoswap32": "AMOSWAP32",
        "e6-irq": "ordinary check/set under saved MIE",
    }[args.allocator_lock]
    manifest = {
        "purpose": "native private A/B/P visibility and guarded repeated lifetimes",
        "expected_i32": -2671,
        "invocations": 3,
        "preflight_lifetimes": 1,
        "private_frame_bytes": 1536,
        "allocator_lock": args.allocator_lock,
        "allocator_instrumentation": args.allocator_markers,
        "kernel_instrumentation": args.kernel_markers,
        **(
            {
                "diagnostic_execution_contract": {
                    "opt_in": "BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1, dma object only",
                    "allowed_hart": 0,
                    "context": "SP inside valid disjoint E6 linker stack",
                    "interrupts": "save/clear/restore MSTATUS.MIE only",
                    "unsupported": "resumable interrupt/NMI/monitor allocator entry",
                    "generic_host_path": "original atomic implementation retained",
                }
            }
            if args.allocator_lock == "e6-irq"
            else {}
        ),
        "private_view_byte_offsets": {"A": 64, "B": 256, "P": 1344},
        "poisons": [165, 90, 231],
        "phase_markers": {
            "0x5100": "entry",
            "0x5200": "before direct allocator acquire",
            "0x5201": "after direct allocator acquire",
            "0x5202": "after direct allocator release",
            "0x5300..2": "before generated kernel",
            "0x5400..2": "after generated kernel",
            "0x5500..2": "after exact output/frame/guard inspection",
            **(
                {
                    "0x7100": "allocator enter before mhartid",
                    "0x7101": "allocator enter after mhartid",
                    "0x7102": "allocator before busy " + busy_path,
                    "0x7103": "allocator after busy " + busy_path,
                    "0x7104": "allocator before "
                    + (
                        "guarded owner load/check/set"
                        if args.allocator_lock != "production"
                        else "owner compare-exchange LR/SC"
                    ),
                    "0x7105": "allocator after "
                    + (
                        "guarded owner load/check/set"
                        if args.allocator_lock != "production"
                        else "owner compare-exchange LR/SC"
                    ),
                    "0x7106": "acquire after allocator enter, before capacity/window check",
                    "0x7200": "acquire before write policy",
                    "0x7201": "acquire after write policy",
                    "0x7202": "acquire before header stores",
                    "0x7203": "acquire after header stores, before tail stores",
                    "0x7204": "acquire after tail stores",
                    "0x7205": "acquire before final fence",
                    "0x7206": "acquire after final fence, before unlock",
                    "0x7207": "acquire after unlock",
                    "0x7300": "release before allocator enter",
                    "0x7301": "release before read policy",
                    "0x7302": "release after read policy, before guard checks",
                    "0x7303": "release after all guard checks",
                    "0x7304": "release before unlock",
                }
                if args.allocator_markers
                else {}
            ),
            **(
                {
                    "0x7110": "hart0/E6 stack validated, before save/clear MIE",
                    "0x7111": "MIE cleared, before ordinary busy check/set",
                }
                if args.allocator_lock == "e6-irq" and args.allocator_markers
                else {}
            ),
            **(
                {
                    "0x6100": "generated frame acquired, before first CPU/RVV fill",
                    "0x6200": "CPU/RVV initialization done, before native AME loop",
                    "0x6300": "native AME loop done, before CPU/RVV final read",
                    "0x6400": "CPU/RVV final read done, before frame release",
                }
                if args.kernel_markers
                else {}
            ),
        },
        "buddy_opt_sha256": digest(opt),
        "buddy_translate_sha256": digest(translate),
        "fixture_sha256": digest(fixture),
        "source_sha256": digest(source),
        "lowered_sha256": digest(lowered),
        "harness_sha256": digest(harness),
        "runtime_sha256": digest(dma / "bosc_ame_dma.c"),
        "compiled_runtime_sha256": digest(dma_source),
        "linker_sha256": digest(vit / "board.ld"),
        "image_sha256": digest(binary),
        "board_submission": "not submitted by this builder",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(binary)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build backend controls from identical frozen nonzero profile BOSCAME IR.

Only the raw kernel object is lowered with each selected buddy-opt binary;
adapter, launcher and runtime objects are byte-identical frozen copies.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
BUDDY = REPO / "buddy-mlir"
LLVM = BUDDY / "llvm/build/bin"
LINKER = (
    LLVM / "ld.lld"
    if (LLVM / "ld.lld").is_file()
    else Path(shutil.which("ld.lld") or "ld.lld")
)
spec = importlib.util.spec_from_file_location(
    "profile_helpers", HERE.parent / "conv_profile/run_profile.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
RUNTIME_OBJECTS = [
    "nh_e6_boot.o",
    "nh_e6_platform.o",
    "uart.o",
    "bosc_ame_dma.o",
    "nr_math.o",
    "ame_sync.o",
    "launch.o",
    "adapter.o",
    "board_support.o",
]


def run(command, directory, label, input_text=None):
    result = subprocess.run(
        [str(item) for item in command],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    (directory / (label + ".command")).write_text(shlex.join(map(str, command)) + "\n")
    (directory / (label + ".stdout")).write_text(result.stdout)
    (directory / (label + ".stderr")).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed: " + result.stderr[-1500:])
    return result


def build(case, variant, opt, args):
    source = args.profiles / case
    directory = args.output / variant / "kernels" / case
    if directory.exists():
        raise RuntimeError("refusing to overwrite ABAB output: " + str(directory))
    directory.mkdir(parents=True)
    for name in ["kernel.boscame.mlir", "adapter.c"]:
        shutil.copy2(source / name, directory / name)
    shutil.copytree(source / "obj", directory / "obj")
    manifest = json.loads((source / "profile-manifest.json").read_text())
    manifest["backend_comparison"] = {
        "variant": variant,
        "buddy_opt": str(opt),
        "buddy_opt_sha256": helpers.sha256(opt),
        "input_profile": str(source),
        "frozen_boscame_sha256": helpers.sha256(source / "kernel.boscame.mlir"),
        "non_kernel_objects": {
            name: helpers.sha256(directory / "obj" / name) for name in RUNTIME_OBJECTS
        },
        "scope": "same frozen BOSCAME IR, adapter and runtime objects; selected buddy-opt raw kernel.o",
    }
    run(
        [
            opt,
            directory / "kernel.boscame.mlir",
            "--lower-bosc-ame=scalar-f32-ordered-reductions=true scalar-f32-max-reductions=true",
            "-o",
            directory / "kernel.llvm.mlir",
        ],
        directory,
        "llvm",
    )
    run(
        [
            BUDDY / "build/bin/buddy-translate",
            "--buddy-to-llvmir",
            directory / "kernel.llvm.mlir",
            "-o",
            directory / "kernel.ll",
        ],
        directory,
        "llvm-ir",
    )
    run(
        [
            LLVM / "llc",
            directory / "kernel.ll",
            "-O2",
            "-filetype=asm",
            "-mtriple=riscv64",
            "-target-abi=lp64d",
            "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
            "-code-model=medium",
            "-o",
            directory / "kernel.s",
        ],
        directory,
        "asm",
    )
    tools = BUDDY / "examples/FPGA-BOSCAME/tools"
    encoded = run(
        [
            sys.executable,
            tools / "ame_to_word.py",
            "--gpr-mode=fixed",
            "--preserve-matrix-registers",
        ],
        directory,
        "encode",
        (directory / "kernel.s").read_text(),
    )
    (directory / "kernel.encoded.s").write_text(encoded.stdout)
    restricted = run(
        [
            sys.executable,
            tools / "restrict_fpga_assembly.py",
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
            "--reject-vector-ordered-reductions",
            "--reject-vector-max-reductions",
        ],
        directory,
        "restrict",
        encoded.stdout,
    )
    (directory / "kernel.nr.s").write_text(restricted.stdout)
    run(
        [
            LLVM / "clang",
            "--target=riscv64-unknown-elf",
            "-march=rv64imafdcv_zicsr_zifencei",
            "-mabi=lp64d",
            "-mno-relax",
            "-c",
            directory / "kernel.nr.s",
            "-o",
            directory / "kernel.o",
        ],
        directory,
        "object",
    )
    elf = directory / (case + ".elf")
    run(
        [
            LINKER,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            MODEL / "link.ld",
            "-Map=" + str(directory / (case + ".map")),
            "-o",
            elf,
            *[directory / "obj" / name for name in RUNTIME_OBJECTS],
            directory / "kernel.o",
        ],
        directory,
        "link",
    )
    run(
        [
            sys.executable,
            tools / "check_nr_elf.py",
            elf,
            "--objdump",
            LLVM / "llvm-objdump",
            "--output",
            directory / "elf-audit.json",
        ],
        directory,
        "audit",
    )
    image = directory / (case + ".bin")
    run([LLVM / "llvm-objcopy", "-O", "binary", elf, image], directory, "bin")
    manifest["kernel_object_sha256"] = helpers.sha256(directory / "kernel.o")
    manifest["bin_sha256"] = helpers.sha256(image)
    manifest["elf_audit_sha256"] = helpers.sha256(directory / "elf-audit.json")
    manifest["backend_comparison"]["emitted_llvm_sha256"] = helpers.sha256(
        directory / "kernel.llvm.mlir"
    )
    helpers.write_json(directory / "profile-manifest.json", manifest)
    print(f"READY {variant} {case} sha256={manifest['bin_sha256']}", flush=True)
    return directory / "profile-manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-opt", type=Path, required=True)
    parser.add_argument("--candidate-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=MODEL / "build/w8a8-profile/bounded32/fused/all/kernels",
    )
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.profiles = args.profiles.resolve()
    cases = args.case or ["conv2d_288x288_3x40_k3_s2", "dwconv2d_72x72x192_k3_s1"]
    pairs = []
    for case in cases:
        baseline = build(case, "baseline", args.baseline_opt.resolve(), args)
        candidate = build(case, "candidate", args.candidate_opt.resolve(), args)
        pairs.append(
            {
                "case": case,
                "baseline_manifest": str(baseline),
                "candidate_manifest": str(candidate),
            }
        )
    helpers.write_json(args.output / "pairs.json", pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

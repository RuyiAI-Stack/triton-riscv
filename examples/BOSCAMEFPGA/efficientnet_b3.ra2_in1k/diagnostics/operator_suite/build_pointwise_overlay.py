#!/usr/bin/env python3
"""Relower frozen pointwise BOSCAME using a fixed compiler into an unvalidated overlay."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_backend_abab import BUDDY, LINKER, LLVM, RUNTIME_OBJECTS, run  # noqa: E402

FIXED_COMPILER_SHA = "29d3ca691086720158f08cfc774dfc20adffc76b0c2fec0e93db4d84698928d8"
SOURCE_FILES = [
    "kernel.ttir",
    "kernel.frontend.mlir",
    "kernel.bufferized.mlir",
    "kernel.planned.mlir",
    "kernel.boscame.mlir",
    "frontend.json",
    "contract-manifest.json",
    "adapter.c",
    ".ordered-reduction-policy",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(source, name):
    result = {
        filename: sha(source / filename)
        for filename in SOURCE_FILES
        if (source / filename).is_file()
    }
    result.update(
        {
            "obj/" + filename: sha(source / "obj" / filename)
            for filename in RUNTIME_OBJECTS
        }
    )
    result["kernel.py"] = sha(MODEL / name / "kernel.py")
    result["source-kernel.o"] = sha(source / "kernel.o")
    result["source-bin"] = sha(source / (name + ".bin"))
    return result


def counts(path):
    assembly = path.read_text()
    return {
        "mma_i8_static_sites": len(
            re.findall(r"^\s*mqma\.b\.mm\b", assembly, re.MULTILINE)
        ),
        "cache_policy_csr_static_sites": len(
            re.findall(r"^\s*csrw\s+(?:3008|0xbc0)\b", assembly, re.MULTILINE)
        ),
        "scalar_byte_load_sites": len(
            re.findall(r"^\s*(?:lb|lbu)\b", assembly, re.MULTILINE)
        ),
        "scalar_byte_store_sites": len(re.findall(r"^\s*sb\b", assembly, re.MULTILINE)),
        "vector_memory_sites": len(
            re.findall(r"^\s*v(?:l|s)(?:e|se)\d+\.v\b", assembly, re.MULTILINE)
        ),
    }


def build(name, args):
    source = args.source / name
    directory = args.output / name
    if directory.exists():
        raise RuntimeError("refusing to overwrite overlay: " + str(directory))
    before = snapshot(source, name)
    directory.mkdir(parents=True)
    write(directory / "input-before.json", before)
    for filename in SOURCE_FILES:
        if (source / filename).is_file():
            shutil.copy2(source / filename, directory / filename)
    shutil.copytree(source / "obj", directory / "obj")
    run(
        [
            args.compiler,
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
    elf = directory / (name + ".elf")
    run(
        [
            LINKER,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            MODEL / "link.ld",
            "-Map=" + str(directory / (name + ".map")),
            "-o",
            elf,
            *[directory / "obj" / filename for filename in RUNTIME_OBJECTS],
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
    image = directory / (name + ".bin")
    run([LLVM / "llvm-objcopy", "-O", "binary", elf, image], directory, "bin")
    after = snapshot(source, name)
    write(directory / "input-after.json", after)
    unchanged = {
        filename: value == after[filename] for filename, value in before.items()
    }
    copies_equal = {
        filename: sha(directory / filename) == value
        for filename, value in before.items()
        if filename in SOURCE_FILES or filename.startswith("obj/")
    }
    old_counts, new_counts = counts(source / "kernel.s"), counts(directory / "kernel.s")
    old_audit = json.loads((source / "elf-audit.json").read_text())
    new_audit = json.loads((directory / "elf-audit.json").read_text())
    record = {
        "case": name,
        "status": "BUILT_UNVALIDATED",
        "publishable": False,
        "validation": "local lowering/link/restricted ELF audit only; no numerical or FPGA validation",
        "source": str(source),
        "output": str(directory),
        "compiler": str(args.compiler),
        "compiler_sha256": sha(args.compiler),
        "input_source_unchanged": unchanged,
        "copied_input_and_runtime_identical": copies_equal,
        "all_inputs_preserved": all(unchanged.values()) and all(copies_equal.values()),
        "source_hashes": before,
        "kernel_object_sha256": sha(directory / "kernel.o"),
        "source_object_sha256": before["source-kernel.o"],
        "object_changed": sha(directory / "kernel.o") != before["source-kernel.o"],
        "bin_sha256": sha(image),
        "source_bin_sha256": before["source-bin"],
        "bin_changed": sha(image) != before["source-bin"],
        "old_asm_counts": old_counts,
        "new_asm_counts": new_counts,
        "mma_static_sites_preserved": old_counts["mma_i8_static_sites"]
        == new_counts["mma_i8_static_sites"]
        > 0,
        "elf_audit_before": old_audit,
        "elf_audit_after": new_audit,
        "elf_audit_sha256": sha(directory / "elf-audit.json"),
    }
    assert record["all_inputs_preserved"] and record["mma_static_sites_preserved"]
    assert (
        new_audit["status"] == "PASS"
        and record["compiler_sha256"] == FIXED_COMPILER_SHA
    )
    write(directory / "overlay-manifest.json", record)
    print(
        "BUILT_UNVALIDATED",
        name,
        "mma=",
        new_counts["mma_i8_static_sites"],
        "vectors=",
        new_counts["vector_memory_sites"],
        "object_changed=",
        record["object_changed"],
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=MODEL / "build/operator-suite/suite-linked80"
    )
    parser.add_argument(
        "--output", type=Path, default=MODEL / "build/memory-task-lowering/pointwise37"
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        default=MODEL / "build/optimization-baseline/tools/buddy-opt-memory-vector",
    )
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.compiler = args.compiler.resolve()
    if sha(args.compiler) != FIXED_COMPILER_SHA:
        raise RuntimeError("compiler does not match frozen requested SHA")
    baseline = json.loads((MODEL / "build/pointwise-source-sha256.json").read_text())
    names = sorted(Path(name).parent.name for name in baseline)
    assert len(names) == 37
    for name, digest in baseline.items():
        if sha(MODEL / name) != digest:
            raise RuntimeError("pointwise source changed: " + name)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda name: build(name, args), names))
    write(args.output / "build-report.json", records)
    summary = {
        "status": "BUILT_UNVALIDATED",
        "publishable": False,
        "cases": len(records),
        "all_local_elf_audits_pass": all(
            row["elf_audit_after"]["status"] == "PASS" for row in records
        ),
        "all_inputs_preserved": all(row["all_inputs_preserved"] for row in records),
        "all_MMA_static_sites_preserved": all(
            row["mma_static_sites_preserved"] for row in records
        ),
        "changed_objects": sum(row["object_changed"] for row in records),
        "changed_bins": sum(row["bin_changed"] for row in records),
        "compiler_sha256": sha(args.compiler),
        "numeric_validation": "not performed",
        "board_validation": "not performed",
        "production_replaced": False,
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

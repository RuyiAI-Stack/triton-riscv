#!/usr/bin/env python3
"""Insert isolated DDR phases around one existing im2col patch; no math rewrite.

This fixed diagnostic handles conv2d_144x144_40x24_k1_s1 only. It checks the
pinned assembly structure, retains every original instruction, and emits both
an isolated raw-function object for same-layout injection and a composite
kernel object with the weight/MMA/dequant stages unchanged. Never publishes or
submits a board job.
"""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
from build_model_image import run, sha  # noqa: E402

CASE = "conv2d_144x144_40x24_k1_s1"
RAW = "triton_" + CASE + "__im2col_quantize"
READ = 0x0F0F000F000F000F
WRITE = 0x0F0F0F0F0F0F0F0F


def phase(kind, policy):
    # Preserve t0 and the exact original stack pointer. The only temporary
    # stores are to the E6 private stack, never to operand DDR or Q scratch.
    return "\n".join(
        [
            "# BEGIN_ISOLATED_" + kind + "_PHASE",
            "\taddi\tsp, sp, -16",
            "\tsd\tt0, 0(sp)",
            "\tli\tt0, 0x%016x" % policy,
            "\tfence\tiorw, iorw",
            "\tcsrw\t0xbc0, t0",
            "\tld\tt0, 0(sp)",
            "\taddi\tsp, sp, 16",
            "# END_ISOLATED_" + kind + "_PHASE",
            "",
        ]
    )


def function_bytes(path):
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count = struct.unpack_from("<HH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + size * i) for i in range(count)
    ]
    result = {}
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        strings = sections[section[6]]
        names = data[strings[4] : strings[4] + strings[5]]
        for start in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, value, length = struct.unpack_from(
                "<IBBHQQ", data, start
            )
            if info & 15 == 2 and length:
                target = sections[index]
                begin = target[4] + value - target[3]
                symbol = names[name : names.index(b"\0", name)].decode()
                result[symbol] = data[begin : begin + length]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kernel-dir",
        type=Path,
        default=ROOT
        / (
            "build/model-integration/model-image-silu-direct-output-20261104/input-snapshots/existing/"
            + CASE
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / ("build/diagnostics/conv-im2col-phase-candidate/" + CASE),
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    baseline = a.kernel_dir / "kernel.nr.s"
    assembly = baseline.read_text()
    start = assembly.index("\t.globl\t" + RAW + " ")
    end_marker = "\t.size\t" + RAW + ", .Lfunc_end2-" + RAW + "\n"
    end = assembly.index(end_marker, start) + len(end_marker)
    original = assembly[start:end]
    if original.count(".LBB2_2:") != 1 or original.count(".LBB2_7:") != 1:
        raise ValueError("unexpected pinned im2col loop/scale structure")
    lines = original.splitlines(keepends=True)
    read_at = next(i for i, line in enumerate(lines) if line.startswith(".LBB2_2:"))
    write_at = next(i for i, line in enumerate(lines) if line.startswith(".LBB2_7:"))
    drain_at = next(i for i, line in enumerate(lines) if line.startswith(".LBB2_87:"))
    # All scalar X loads must precede the write phase. Later loads are private
    # stack reloads or descriptor pointers, not FP operand reads.
    for line in lines[write_at + 1 :]:
        if re.match(r"\s+flw\s", line) and "(sp)" not in line:
            raise ValueError("FP operand load after proposed write phase: " + line)
    if (
        sum(
            bool(re.match(r"\s+flw\s", line)) and "(sp)" not in line
            for line in lines[read_at:write_at]
        )
        != 40
    ):
        raise ValueError("expected exactly40 input tap loads per patch")
    if any(re.match(r"\s+sb\s", line) for line in lines[read_at:write_at]):
        raise ValueError("QPatch byte store inside proposed read phase")
    # Every DDR byte store and scale store is in the write part. The final
    # byte+scale block is laid out before the loop header, reached from the
    # bottom of the quantization block; its successor is the next read phase.
    first_store = next(line for line in lines[write_at:] if re.match(r"\s+sb\s", line))
    if "-39(ra)" not in first_store:
        raise ValueError("unexpected first QPatch store")
    counts = {
        "input_float_loads": 40,
        "qpatch_byte_store_instructions": sum(
            bool(re.match(r"\s+sb\s", x)) for x in lines
        ),
        "patch_scale_store_instructions": sum(
            bool(re.match(r"\s+fsw\s", x)) and "(sp)" not in x for x in lines
        ),
    }
    if (
        counts["qpatch_byte_store_instructions"] != 40
        or counts["patch_scale_store_instructions"] != 1
    ):
        raise ValueError("unexpected output store structure")
    candidate_lines = []
    for i, line in enumerate(lines):
        candidate_lines.append(line)
        if i == read_at:
            candidate_lines.append(phase("READ", READ))
        elif i == write_at:
            candidate_lines.append(phase("WRITE", WRITE))
        elif i == drain_at:
            candidate_lines.append(
                "# BEGIN_ISOLATED_DRAIN_PHASE\n\tfence\tiorw, iorw\n# END_ISOLATED_DRAIN_PHASE\n"
            )
    candidate = "".join(candidate_lines)
    stripped = re.sub(
        r"# BEGIN_ISOLATED_(?:READ|WRITE|DRAIN)_PHASE\n.*?# END_ISOLATED_(?:READ|WRITE|DRAIN)_PHASE\n",
        "",
        candidate,
        flags=re.S,
    )
    if stripped != original:
        raise ValueError(
            "candidate changes original instructions beyond phase insertions"
        )
    (a.output / "baseline-im2col.s").write_text(original)
    (a.output / "raw-im2col.s").write_text("\t.attribute\t4, 16\n\t.text\n" + candidate)
    (a.output / "kernel.s").write_text(assembly[:start] + candidate + assembly[end:])
    (a.output / "adapter.c").write_bytes((a.kernel_dir / "adapter.c").read_bytes())
    clang = REPO / "buddy-mlir/llvm/build/bin/clang"
    restrict = REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools/restrict_fpga_assembly.py"
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mno-relax",
    ]
    for stem in ("kernel", "raw-im2col"):
        result = subprocess.run(
            [
                sys.executable,
                str(restrict),
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
                "--no-fences",
            ],
            input=(a.output / (stem + ".s")).read_text(),
            capture_output=True,
            text=True,
        )
        (a.output / (stem + ".restrict.log")).write_text(result.stderr)
        if result.returncode:
            raise ValueError("assembly restriction failed: " + stem)
        (a.output / (stem + ".nr.s")).write_text(result.stdout)
        run(
            [
                clang,
                *flags,
                "-c",
                a.output / (stem + ".nr.s"),
                "-o",
                a.output / (stem + ".o"),
            ],
            a.output / (stem + ".object.log"),
        )
    before = function_bytes(a.kernel_dir / "kernel.o")
    after = function_bytes(a.output / "kernel.o")
    comparisons = {name: before[name] == after[name] for name in before if name != RAW}
    if before.keys() != after.keys() or not all(comparisons.values()):
        raise ValueError("a non-im2col function changed: " + str(comparisons))
    report = {
        "status": "PASS_BUILD_AND_STATIC_AUDIT_ONLY",
        "board_status": "NOT_RUN",
        "case": CASE,
        "raw_entry": RAW,
        "baseline_directory": str(a.kernel_dir),
        "baseline_kernel_sha256": sha(a.kernel_dir / "kernel.o"),
        "baseline_assembly_sha256": sha(baseline),
        "kernel_sha256": sha(a.output / "kernel.o"),
        "raw_im2col_object_sha256": sha(a.output / "raw-im2col.o"),
        "source_path": str(a.output / "kernel.s"),
        "source_sha256": sha(a.output / "kernel.s"),
        "builder_source_sha256": sha(Path(__file__)),
        "all_original_im2col_instructions_unchanged": True,
        "non_im2col_function_bytes_unchanged": comparisons,
        "adapter_byte_identical": sha(a.output / "adapter.c")
        == sha(a.kernel_dir / "adapter.c"),
        "static_counts": counts,
        "read_policy": hex(READ),
        "write_policy": hex(WRITE),
        "protocol": "each patch: fenced CSR read phase; all40 scalar X loads; scale and existing quantization; fenced CSR write phase; all40 QPatch byte stores plus scale; next patch read-phase fence drains writes; final return fence drains last patch writes",
        "private_stack_only_during_read_phase": True,
        "temporary_stack_bytes": 16,
        "MMA_modified": False,
        "production_modified": False,
        "limitations": [
            "No hardware result or numeric/coherence diagnosis established.",
            "Fixed assembly-structure diagnostic for one convolution specialization.",
            "Weight quantization, MMA and dequantization deliberately retained.",
        ],
    }
    (a.output / "candidate-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

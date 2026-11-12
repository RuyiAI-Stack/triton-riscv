#!/usr/bin/env python3
"""Count bounded emitted scalar-convolution paths without executing FPGA code.

This traces integer address/control instructions in kernel.nr.s, skipping FP
values. It counts instruction mnemonics, not cycles; each compressed instruction
also counts once. Static instruction counts must agree with the object disassembly.
The C adapter, raw-entry wrapper, initialization and hardware stalls are excluded.
"""

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe  # noqa: E402

NAMES = (
    "conv2d_288x288_3x40_k3_s2",
    "dwconv2d_18x18x816_k5_s2",
)


def parse_core(assembly, symbol):
    body = assembly[assembly.index(symbol + ":") : assembly.index(".Lfunc_end0:")]
    code, labels = [], {}
    for line in body.splitlines()[1:]:
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(code)
        elif not line.startswith("."):
            fields = line.split(None, 1)
            args = (
                [value.strip() for value in fields[1].split(",")]
                if len(fields) > 1
                else []
            )
            code.append((fields[0], args))
    return code, labels


def signed32(value):
    return (value + 2**31) % 2**32 - 2**31


def trace(code, labels, point):
    """Follow all branches with concrete program coordinates and no FP inputs.

    The scalar raw function receives its final three pid arguments at incoming
    sp+24/32/40. Descriptor loads have inert synthetic addresses: no descriptor or
    floating value participates in control flow. Unknown opcodes fail closed.
    """
    registers = defaultdict(int)
    registers["sp"] = 0x100000
    unknown = defaultdict(lambda: True)
    unknown.update(sp=False, zero=False)
    memory = {
        registers["sp"] + offset: value for offset, value in zip((24, 32, 40), point)
    }
    memory_unknown = {address: False for address in memory}
    counts, pc = Counter(), 0

    def value(operand):
        return registers[operand] if operand[0].isalpha() else int(operand, 0)

    def address(operand):
        offset, register = operand.split("(")
        return value(offset) + registers[register[:-1]]

    def uncertain(operand):
        return unknown[operand] if operand[0].isalpha() else False

    while True:
        opcode, args = code[pc]
        counts[opcode] += 1
        if counts.total() > 10000:
            raise ValueError("Unexpectedly long per-program control path")
        pc += 1
        if opcode == "ret":
            return counts
        if opcode in ("flw", "fsw", "fmul.s", "fadd.s", "fmv.w.x"):
            continue
        # Descriptor loads may have arbitrary values. Reject a trace if those
        # values ever affect a branch, rather than silently treating them as 0.
        if opcode in ("blt", "bnez", "blez"):
            inputs = args[:-1]
            if any(uncertain(operand) for operand in inputs):
                raise ValueError("Non-coordinate data affects control flow")
        elif opcode in ("ld", "lw"):
            unknown[args[0]] = memory_unknown.get(address(args[1]), True)
        elif opcode == "sd":
            memory_unknown[address(args[1])] = uncertain(args[0])
        elif opcode != "j":
            unknown[args[0]] = any(uncertain(operand) for operand in args[1:])
        if opcode == "li":
            registers[args[0]] = value(args[1])
        elif opcode == "lui":
            registers[args[0]] = value(args[1]) << 12
        elif opcode == "mv":
            registers[args[0]] = value(args[1])
        elif opcode in ("ld", "lw"):
            registers[args[0]] = memory.get(address(args[1]), 0)
        elif opcode == "sd":
            memory[address(args[1])] = value(args[0])
        elif opcode in ("addi", "addiw", "add", "addw", "mul", "mulw"):
            left, right = value(args[1]), value(args[2])
            result = left * right if opcode.startswith("mul") else left + right
            registers[args[0]] = signed32(result) if opcode.endswith("w") else result
        elif opcode == "slli":
            registers[args[0]] = value(args[1]) << value(args[2])
        elif opcode == "srliw":
            registers[args[0]] = (value(args[1]) % 2**32) >> value(args[2])
        elif opcode == "slti":
            registers[args[0]] = int(value(args[1]) < value(args[2]))
        elif opcode == "not":
            registers[args[0]] = ~value(args[1])
        elif opcode == "and":
            registers[args[0]] = value(args[1]) & value(args[2])
        elif opcode == "or":
            registers[args[0]] = value(args[1]) | value(args[2])
        elif opcode == "xori":
            registers[args[0]] = value(args[1]) ^ value(args[2])
        elif opcode == "j":
            pc = labels[args[0]]
        elif opcode == "blt":
            pc = labels[args[2]] if value(args[0]) < value(args[1]) else pc
        elif opcode == "bnez":
            pc = labels[args[1]] if value(args[0]) != 0 else pc
        elif opcode == "blez":
            pc = labels[args[1]] if value(args[0]) <= 0 else pc
        else:
            raise ValueError("Unsupported opcode: " + opcode)


def axis_classes(extent, kernel, stride, padding, output):
    groups = defaultdict(list)
    for point in range(output):
        mask = tuple(
            0 <= point * stride - padding + tap < extent for tap in range(kernel)
        )
        groups[mask].append(point)
    return list(groups.values())


def analyze(case, kernels, objdump):
    directory = kernels / case["name"]
    assembly_path = directory / "kernel.nr.s"
    assembly = assembly_path.read_text()
    symbol = "triton_" + case["name"]
    code, labels = parse_core(assembly, symbol)
    disassembly = subprocess.run(
        [
            str(objdump),
            "-d",
            "--no-show-raw-insn",
            "--disassemble-symbols=" + symbol,
            str(directory / "kernel.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    decoded = re.findall(
        r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9.]*)\s", disassembly, re.MULTILINE
    )
    # ret has no following operand; the line may lack trailing whitespace.
    decoded_count = len(re.findall(r"^\s*[0-9a-f]+:\s+\S+", disassembly, re.MULTILINE))
    if decoded_count != len(code):
        raise ValueError(
            "Assembly pseudos expanded; mnemonic count is not an instruction count"
        )
    if any(opcode in ("call", "ecall", "wfi") for opcode, _ in code):
        raise ValueError("Unexpected external call or wait in scalar core")
    spec = case["constexprs"]
    heights = axis_classes(spec["H"], spec["KH"], spec["SH"], spec["PH"], spec["OH"])
    widths = axis_classes(spec["W"], spec["KW"], spec["SW"], spec["PW"], spec["OW"])
    total, paths = Counter(), []
    channels = case["grid"][0]
    for ys, xs in itertools.product(heights, widths):
        counts = trace(code, labels, (0, ys[0], xs[0]))
        # Last coordinates of each padding class and the final channel must
        # share that class's control path. Every kernel bound is a constexpr.
        assert trace(code, labels, (channels - 1, ys[-1], xs[-1])) == counts
        multiplicity = len(ys) * len(xs) * channels
        total.update({opcode: count * multiplicity for opcode, count in counts.items()})
        paths.append(
            {
                "representative_pid": [0, ys[0], xs[0]],
                "matching_programs": multiplicity,
                "instructions": counts.total(),
                "multiply_adds": counts["fmul.s"],
            }
        )
    valid_h = sum(
        sum(
            0 <= y * spec["SH"] - spec["PH"] + tap < spec["H"]
            for tap in range(spec["KH"])
        )
        for y in range(spec["OH"])
    )
    valid_w = sum(
        sum(
            0 <= x * spec["SW"] - spec["PW"] + tap < spec["W"]
            for tap in range(spec["KW"])
        )
        for x in range(spec["OW"])
    )
    expected_madds = channels * valid_h * valid_w
    if case["family"] == "conv2d_f32":
        expected_madds *= spec["C"]
    assert total["fmul.s"] == expected_madds
    audit = json.loads((directory / "elf-audit.json").read_text())
    return {
        "name": case["name"],
        "programs": math.prod(case["grid"]),
        "core_static_instructions": len(code),
        "decoded_core_instructions": decoded_count,
        "core_total_instructions": total.total(),
        "multiply_adds": total["fmul.s"],
        "fp32_loads": total["flw"],
        "fp32_stores": total["fsw"],
        "opcode_counts": dict(sorted(total.items())),
        "paths": paths,
        "whole_elf_ame_instructions": audit["ame_instructions"],
        "whole_elf_rvv_opcodes": audit["rvv_instructions"],
        "decoded_core_opcodes": sorted(set(decoded)),
        "assembly_sha256": hashlib.sha256(assembly_path.read_bytes()).hexdigest(),
        "object_sha256": hashlib.sha256(
            (directory / "kernel.o").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--objdump",
        type=Path,
        default=ROOT.parents[2] / "buddy-mlir/llvm/build/bin/llvm-objdump",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/conv-timeout-analysis/static.json"
    )
    args = parser.parse_args()
    cases = [analyze(describe(name), args.kernels, args.objdump) for name in NAMES]
    report = {
        "method": "Bounded integer-control trace of emitted scalar assembly, grouped by padding masks",
        "instruction_count_unit": "decoded core instructions, compressed and uncompressed each count once",
        "excluded": [
            "C adapter",
            "raw entry wrapper",
            "launcher initialization",
            "hardware stalls",
        ],
        "not_a_cycle_model": True,
        "cases": cases,
        "conv_to_depth_core_instruction_ratio": cases[0]["core_total_instructions"]
        / cases[1]["core_total_instructions"],
        "conv_to_depth_program_ratio": cases[0]["programs"] / cases[1]["programs"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    for case in cases:
        print(
            case["name"],
            case["programs"],
            case["core_total_instructions"],
            case["multiply_adds"],
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reject E6/vector/unknown instructions in an NH-only ELF before upload."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = args.elf.read_bytes()
    errors = []
    if image[:6] != b"\x7fELF\x02\x01" or len(image) < 64:
        raise SystemExit("expected a little-endian ELF64 file")
    machine = struct.unpack_from("<H", image, 18)[0]
    entry = struct.unpack_from("<Q", image, 24)[0]
    if machine != 243 or entry != 0x80000000:
        errors.append("expected RISC-V with entry 0x80000000")
    assembly = subprocess.check_output(
        [args.objdump, "--disassemble", "--no-show-raw-insn", str(args.elf)], text=True
    )
    symbols = subprocess.check_output(
        [args.objdump, "--syms", str(args.elf)], text=True
    )
    if re.search(r"\be6_start\b", symbols):
        errors.append("E6 startup is linked")
    instructions = []
    for line in assembly.splitlines():
        match = re.match(r"\s*[0-9a-f]+:\s+(\S+)", line)
        if match:
            instructions.append(match.group(1))
            if match.group(1).startswith("v") or match.group(1) in (
                "<unknown>",
                ".word",
                ".2byte",
                ".4byte",
            ):
                errors.append(line.strip())
    if not instructions:
        errors.append("objdump returned no instructions")
    for marker in (b"@@FPGA_START", b"@@FPGA_RESULT"):
        if marker not in image:
            errors.append("missing UART protocol marker " + marker.decode())
    report = dict(
        profile="nh-only",
        elf=str(args.elf),
        sha256=hashlib.sha256(image).hexdigest(),
        entry=hex(entry),
        instruction_count=len(instructions),
        ok=not errors,
        errors=errors,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise SystemExit("NH ELF audit failed: " + "; ".join(errors[:10]))


if __name__ == "__main__":
    main()

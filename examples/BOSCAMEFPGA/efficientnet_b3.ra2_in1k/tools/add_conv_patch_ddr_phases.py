#!/usr/bin/env python3
"""Add explicit per-patch DDR phases to the bounded scalar im2col LLVM entry.

All original LLVM instructions remain unchanged. Other raw stages, including
INT8 MMA, are outside the transformed function. Apply before the existing scalar
LLVM O2 stage; source/ABI and numerical acceptance are separate gates.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))
from cases import describe  # noqa: E402

READ_POLICY = 0x0F0F000F000F000F
WRITE_POLICY = 0x0F0F0F0F0F0F0F0F


def phase(kind, policy=None):
    operation = "fence iorw,iorw"
    if policy is None:
        call = '  call void asm sideeffect "' + operation + '", "~{memory}"()\n'
    else:
        call = (
            '  call void asm sideeffect "' + operation + '; csrw 0xbc0,$0", '
            '"r,~{memory}"(i64 %d)\n' % policy
        )
    return (
        "  ; BEGIN_CONV_PATCH_"
        + kind
        + "\n"
        + call
        + "  ; END_CONV_PATCH_"
        + kind
        + "\n"
    )


def transform(text, case):
    if case["family"] not in ("conv2d_f32", "dwconv2d_f32"):
        raise ValueError("explicit bounded W8A8 convolution case required")
    if "BEGIN_CONV_PATCH_" in text:
        raise ValueError("input already contains patch phases")
    symbol = "triton_" + case["name"] + "__im2col_quantize"
    pattern = r"(^define void @" + re.escape(symbol) + r"\([^\n]+\) \{\n)(.*?)(^\}\n)"
    match = re.search(pattern, text, re.M | re.S)
    if not match:
        raise ValueError("one exact raw im2col LLVM function required")
    body = match.group(2)
    spec = case["constexprs"]
    width = (
        spec["KH"] * spec["KW"] * (1 if case["family"] == "dwconv2d_f32" else spec["C"])
    )
    if len(re.findall(r"^  store i8 ", body, re.M)) != width:
        raise ValueError("unexpected logical-K byte store count")
    if len(re.findall(r"^  store float ", body, re.M)) != 1:
        raise ValueError("expected one patch scale store")
    if len(re.findall(r"^  %\d+ = load float, ", body, re.M)) != width:
        raise ValueError("expected one input load per logical tap")
    loops = list(
        re.finditer(
            r"  %(?P<condition>\d+) = icmp slt i32 %\d+, 64\n"
            r"  br i1 %(?P=condition), label %(?P<body>\d+), label %\d+",
            body,
        )
    )
    if len(loops) != 1:
        raise ValueError("expected one outer64 patch loop")
    label = loops[0].group("body")
    target = re.search(r"^" + label + r":[^\n]*\n", body, re.M)
    if not target or re.match(r"  %\d+ = phi ", body[target.end() :]):
        raise ValueError("unexpected patch body entry/PHI structure")
    first_store = re.search(r"^  store i8 ", body, re.M)
    if not first_store or target.end() >= first_store.start():
        raise ValueError("input gathering must precede QPatch stores")
    # The static unrolled logical taps all precede the first output byte.
    # Pointer/descriptor loads and private spills do not read input FP values.
    if re.search(r"^  %\d+ = load float, ", body[first_store.start() :], re.M):
        raise ValueError("input FP load after QPatch write phase")
    modified = (
        body[: first_store.start()]
        + phase("WRITE", WRITE_POLICY)
        + body[first_store.start() :]
    )
    modified = (
        modified[: target.end()] + phase("READ", READ_POLICY) + modified[target.end() :]
    )
    if modified.count("  ret void\n") != 1:
        raise ValueError("expected one raw-entry return")
    modified = modified.replace("  ret void\n", phase("DRAIN") + "  ret void\n")
    stripped = re.sub(
        r"  ; BEGIN_CONV_PATCH_(READ|WRITE|DRAIN)\n.*?  ; END_CONV_PATCH_\1\n",
        "",
        modified,
        flags=re.S,
    )
    if stripped != body:
        raise ValueError("transformation changed original LLVM math")
    result = text[: match.start(2)] + modified + text[match.end(2) :]
    report = {
        "status": "PASS_TRANSFORM_ONLY",
        "case": case["name"],
        "raw_entry": symbol,
        "logical_K": width,
        "input_fp_loads": width,
        "QPatch_byte_stores": width,
        "PatchScale_stores": 1,
        "read_policy": hex(READ_POLICY),
        "write_policy": hex(WRITE_POLICY),
        "all_original_LLVM_instructions_unchanged": True,
        "other_functions_unchanged": True,
        "output_stores_under_read_policy": False,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "board_status": "NOT_RUN",
    }
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output, report = transform(args.input.read_text(), describe(args.case))
    args.output.write_text(output)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS_PATCH_DDR_PHASES", args.case, report["logical_K"])


if __name__ == "__main__":
    main()

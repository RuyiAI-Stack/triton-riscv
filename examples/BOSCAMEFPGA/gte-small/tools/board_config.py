#!/usr/bin/env python3
"""Generate the board ABI wrappers from the literal per-case configuration."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe  # noqa: E402 - local modules require the example path

sys.path.insert(0, str(ROOT / "tools"))
from export import argument_shapes  # noqa: E402 - local modules require the example path

sys.path.insert(0, str(ROOT.parent / "tools"))
from scheduler_outputs import (  # noqa: E402 - local modules require the example path
    board_launch,
    capture_lines,
    describe_outputs,
    output_header,
)

PATCH_DEPENDENCIES = (
    "quantize_196x768",
    "quantize_768x768",
    "matmul_i8_196x768x768",
    "dequantize_196x768",
)


def dependencies(case):
    return tuple(case.get("dependencies", ())) or (
        PATCH_DEPENDENCIES if case["family"] == "patch_embed" else ()
    )


def configuration(case, accumulator_validation=False, accumulator_round=None):
    family, shape = case["family"], case["shape"]
    if accumulator_validation and family not in ("gemm_i8", "matmul_i8"):
        raise ValueError("accumulator diagnostic requires an integer matmul")
    round_ids = [accumulator_round] if accumulator_round is not None else [0, 1, 2]
    trials = len(round_ids) if accumulator_validation else 1
    if family in ("linear", "gemm_i8", "matmul_i8"):
        checked = shape[0] * shape[1] * trials
    elif family == "patch_embed":
        checked = 196 * 768
    elif family == "quantize_transpose":
        checked = math.prod(shape) + shape[1]
    elif family == "head_layout":
        checked = math.prod(shape)
    elif family == "per_token_quantization":
        checked = trials * (math.prod(shape) + shape[0])
    else:
        checked = math.prod(shape)
    return {
        "case": case["name"],
        "expected_invocations": trials,
        "expected_checked": case.get("checked_elements", checked),
        "dependencies": list(dependencies(case)),
        "cycle_scope": "complete typed adapter including grid dispatch and DMA; final fence; excludes initialization, validation, UART",
        **(
            {
                "validation_mode": "accumulator_three_round_bit_exact",
                "program_orders": [
                    ["forward", "tail-first", "reverse"][round_id]
                    for round_id in round_ids
                ],
                "validation_rounds": trials,
                "validation_round_ids": round_ids,
                "expected_checked_per_round": shape[0] * shape[1],
                "guard_bytes_per_side_per_argument": 64,
                "inputs_checked_bytewise": True,
            }
            if accumulator_validation
            else {}
        ),
    }


def wrapper(case, timed, export_invocation=None):
    name = case["name"]
    params = ", ".join(
        f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"])
    )
    args = ", ".join(f"a{i}" for i in range(len(case["arguments"])))
    lines = [
        f"extern void vit_board_impl_{name}({params});",
        f"void _mlir_ciface_kernel_{name}({params}) {{",
    ]
    if timed:
        lines += ["  unsigned slot = vit_board_kernel_begin();"]
    lines += ["  nh_e6_cache_writes();"]
    if timed:
        lines += ["  uint64_t begin = nr_cycles();"]
    lines += [
        f"  vit_board_impl_{name}({args});",
        '  __asm__ volatile("fence iorw, iorw" ::: "memory");',
    ]
    if timed:
        lines += [
            "  uint64_t end = nr_cycles();",
            "  vit_board_kernel_end(slot, end - begin);",
        ]
    lines += ["  nh_e6_cache_uncached_loads();"]
    if timed:
        lines += capture_lines(case, export_invocation)
    lines += ["}"]
    return "\n".join(lines)


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--adapter-source", type=Path)
    parser.add_argument("--accumulator-validation", action="store_true")
    parser.add_argument("--accumulator-round", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--export-invocation",
        type=int,
        choices=(0, 1),
        help="Return one invocation per task; still executes the complete original two-invocation launch",
    )
    args = parser.parse_args()
    case = describe(args.case)
    if args.dependencies:
        print(" ".join(dependencies(case)))
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.adapter_source is not None:
        source = args.adapter_source.read_text()
        if args.accumulator_validation:
            configuration(case, True)
            # Keep the real entry and all descriptor checks. Only dispatch
            # order changes; each real program appears exactly once per call.
            prefix = "extern unsigned vit_accumulator_order;\n"
            for axis, variable in enumerate(("x", "y", "z")):
                extent = case["grid"][axis]
                original = (
                    f"for (int32_t {variable} = 0; {variable} < {extent}; ++{variable})"
                )
                replacement = (
                    f"for (int32_t ordinal_{variable} = 0, {variable} = 0; "
                    f"ordinal_{variable} < {extent} && "
                    f"(({variable} = (vit_accumulator_order == 2 || "
                    f"(vit_accumulator_order == 1 && {axis} == 0)) ? "
                    f"{extent} - 1 - ordinal_{variable} : ordinal_{variable}), 1); "
                    f"++ordinal_{variable})"
                )
                if source.count(original) != 1:
                    raise ValueError(
                        "adapter dispatch changed; diagnostic order requires review"
                    )
                source = source.replace(original, replacement)
            source = prefix + source
        if case["family"] == "patch_embed":
            original = (
                "  for (int p = 0; p < 196; ++p)\n"
                "    for (int o = 0; o < 768; ++o)\n"
                "      ((float *)p3)[o * 196 + p] = patch_gemm[p * 768 + o] + ((const float *)p2)[o];"
            )
            replacement = (
                "  /* Stage each row on the stable stack before switching to DDR writes. */\n"
                "  float row[768];\n"
                "  for (int p = 0; p < 196; ++p) {\n"
                "    nh_e6_cache_uncached_loads();\n"
                "    for (int o = 0; o < 768; ++o)\n"
                "      row[o] = patch_gemm[p * 768 + o] + ((const float *)p2)[o];\n"
                "    nh_e6_cache_writes();\n"
                "    for (int o = 0; o < 768; ++o)\n"
                "      ((float *)p3)[o * 196 + p] = row[o];\n"
                "  }"
            )
            if source.count(original) != 1:
                raise ValueError(
                    "patch adapter projection changed; cache staging requires review"
                )
            source = source.replace(original, replacement)
        write_if_changed(args.output / "adapter.c", source)
        return
    if args.accumulator_validation:
        parser.error(
            "Use frozen diagnostic images for legacy accumulator checks; this path returns outputs for host validation"
        )
    config = describe_outputs(
        case, argument_shapes(case), configuration(case), args.export_invocation
    )
    header = (
        "/* Generated from kernel.py by tools/board_config.py. */\n"
        f'#define VIT_BOARD_CASE "{case["name"]}"\n'
        f"#define VIT_BOARD_EXPECTED_INVOCATIONS {config['expected_invocations']}u\n"
        f"#define VIT_BOARD_EXPECTED_CHECKED {config['expected_checked']}u\n"
        + (
            "#define VIT_BOARD_ACCUMULATOR_VALIDATION 1\n"
            f"#define VIT_BOARD_VALIDATION_ROUNDS {config['validation_rounds']}u\n"
            f"#define VIT_BOARD_VALIDATION_FIRST_ROUND {config['validation_round_ids'][0]}u\n"
            if args.accumulator_validation
            else ""
        )
    )
    header += output_header(config)
    source = [
        "/* Wrappers preserve the typed adapter's numerical body. */",
        '#include "board_support.h"',
        wrapper(case, True, args.export_invocation),
    ]
    source.extend(wrapper(describe(dep), False) for dep in dependencies(case))
    write_if_changed(args.output / "board_case.h", header)
    write_if_changed(args.output / "board_wrapper.c", "\n\n".join(source) + "\n")
    write_if_changed(
        args.output / "board_config.json", json.dumps(config, indent=2) + "\n"
    )
    launch = board_launch(case, config)
    write_if_changed(args.output / "board_launch.c", launch)


if __name__ == "__main__":
    main()

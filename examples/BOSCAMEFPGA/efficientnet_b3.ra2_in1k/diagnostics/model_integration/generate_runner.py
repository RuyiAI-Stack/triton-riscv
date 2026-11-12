#!/usr/bin/env python3
"""Generate and compile an isolated whole-graph C call runner for review.

The board runtime must provide workspace() and both cache/coherency hooks.
This tool does not link a board ELF or claim whole-model execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan-dir", type=Path, default=ROOT / "build/model-integration/physical-plan"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "build/model-integration/runner"
    )
    p.add_argument(
        "--clang", type=Path, default=REPO / "buddy-mlir/llvm/build/bin/clang"
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((a.plan_dir / "physical-plan.json").read_text())
    params = json.loads((a.plan_dir / "parameters.json").read_text())
    if plan["workspace"]["provisional"]:
        raise ValueError(
            "cannot generate runner with provisional composite scratch plan"
        )
    if sha(a.plan_dir / "parameters.bin") != params["sha256"]:
        raise ValueError("parameters.bin differs from layout manifest")
    tensor = plan["tensors"]
    parameter = {v["name"]: v for v in params["entries"]}
    wbytes = plan["workspace"]["provisional_total_bytes"]
    header = [
        "#ifndef EFFICIENTNET_B3_GRAPH_RUNNER_H",
        "#define EFFICIENTNET_B3_GRAPH_RUNNER_H",
        '#include "support.h"',
        "#define EFFICIENTNET_B3_WORKSPACE_BYTES %du" % wbytes,
        "#define EFFICIENTNET_B3_PARAMETER_BYTES %du" % params["bytes"],
        "#define EFFICIENTNET_B3_PHYSICAL_CALLS %du" % len(plan["calls"]),
        "#define EFFICIENTNET_B3_OUTPUT_ELEMENTS %du"
        % tensor[plan["output"]["value"]]["elements"],
        "#define EFFICIENTNET_B3_OUTPUT_SHAPE "
        + json.dumps(json.dumps(plan["output"]["shape"], separators=(",", ":"))),
        "typedef struct { void *pointer; size_t bytes; unsigned write; } EfficientNetB3Span;",
        "typedef struct {",
        "  /* Runtime must perform CPU/AME operand visibility and trace/cycle handling. */",
        "  void (*before)(unsigned index, const char *name, const EfficientNetB3Span *spans, unsigned count);",
        "  void (*after)(unsigned index, const char *name, const EfficientNetB3Span *spans, unsigned count);",
        "} EfficientNetB3Hooks;",
        "float *efficientnet_b3_execute_graph(void *parameters, size_t parameter_bytes, void *input,",
        "    unsigned char *arena, size_t arena_bytes, const EfficientNetB3Hooks *hooks);",
        "#endif",
        "",
    ]
    (a.output / "graph_runner.h").write_text(chr(10).join(header))
    lines = [
        "/* Generated from the real-weight physical plan. No board execution implied. */",
        '#include "graph_runner.h"',
        "#if VIT_WORKSPACE_BYTES != EFFICIENTNET_B3_WORKSPACE_BYTES",
        '#error "Composite adapters and graph runner must use the same workspace size"',
        "#endif",
    ]
    used = {}
    for call in plan["calls"]:
        args = call["arguments"]
        proto = (
            "extern void _mlir_ciface_kernel_"
            + call["target"]
            + "("
            + ", ".join("MemRef%d *" % len(v["shape"]) for v in args)
            + ");"
        )
        if call["target"] in used and used[call["target"]] != proto:
            raise ValueError("incompatible prototype " + call["target"])
        used[call["target"]] = proto
    lines += list(used.values())
    lines += [
        "float *efficientnet_b3_execute_graph(void *parameters, size_t parameter_bytes, void *input,",
        "    unsigned char *arena, size_t arena_bytes, const EfficientNetB3Hooks *hooks) {",
        "  if (!parameters || !input || !arena || !hooks || !hooks->before || !hooks->after)",
        "    __builtin_trap();",
        "  if (((uintptr_t)parameters | (uintptr_t)input | (uintptr_t)arena) & 63u) __builtin_trap();",
        "  if (parameter_bytes != EFFICIENTNET_B3_PARAMETER_BYTES ||",
        "      arena_bytes != EFFICIENTNET_B3_WORKSPACE_BYTES || workspace(0) != arena) __builtin_trap();",
        "  if ((uintptr_t)arena > UINTPTR_MAX-arena_bytes ||",
        "      (uintptr_t)parameters > UINTPTR_MAX-parameter_bytes ||",
        "      (uintptr_t)input > UINTPTR_MAX-%du) __builtin_trap();"
        % tensor["input"]["bytes"],
        "  if (!((uintptr_t)parameters+parameter_bytes <= (uintptr_t)arena ||",
        "        (uintptr_t)arena+arena_bytes <= (uintptr_t)parameters)) __builtin_trap();",
        "  if (!((uintptr_t)input+%du <= (uintptr_t)arena ||"
        % tensor["input"]["bytes"],
        "        (uintptr_t)arena+arena_bytes <= (uintptr_t)input)) __builtin_trap();",
    ]

    def pointer(arg):
        value = arg["value"]
        if arg["source"] == "parameter":
            return "(unsigned char *)parameters + %du" % parameter[value]["offset"]
        storage = tensor[value]["storage"]
        return (
            "input"
            if storage["kind"] == "external_input"
            else "arena + %du" % storage["offset"]
        )

    for call in plan["calls"]:
        index = call["index"]
        name = call["logical_node"] + " / " + call["target"]
        lines += ["  { /* call %d: %s */" % (index, name)]
        for i, arg in enumerate(call["arguments"]):
            shape = arg["shape"]
            rank = len(shape)
            lines.append(
                "    MemRef%d d%d = make_%d(%s, %s);"
                % (rank, i, rank, pointer(arg), ", ".join(map(str, shape)))
            )
        lines += ["    EfficientNetB3Span spans[] = {"]
        for arg in call["arguments"]:
            spec = (
                parameter[arg["value"]]
                if arg["source"] == "parameter"
                else tensor[arg["value"]]
            )
            lines.append(
                "      {(void *)(%s), %du, %du},"
                % (pointer(arg), spec["bytes"], arg["value"] in call["writes"])
            )
        lines += [
            "    };",
            "    hooks->before(%du, %s, spans, %du);"
            % (index, json.dumps(name), len(call["arguments"])),
            "    _mlir_ciface_kernel_"
            + call["target"]
            + "("
            + ", ".join("&d%d" % i for i in range(len(call["arguments"])))
            + ");",
            "    hooks->after(%du, %s, spans, %du);"
            % (index, json.dumps(name), len(call["arguments"])),
            "  }",
        ]
    result = tensor[plan["output"]["value"]]["storage"]
    lines += ["  return (float *)(arena + %du);" % result["offset"], "}", ""]
    (a.output / "graph_runner.c").write_text(chr(10).join(lines))
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mno-relax",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-DVIT_WORKSPACE_BYTES=%du" % wbytes,
        "-I",
        str(ROOT),
        "-I",
        str(a.output),
        "-c",
        str(a.output / "graph_runner.c"),
        "-o",
        str(a.output / "graph_runner.o"),
    ]
    proc = subprocess.run([str(a.clang), *flags], capture_output=True, text=True)
    (a.output / "compile.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    manifest = {
        "status": "PASS",
        "scope": "C typed-descriptor runner object compilation only; no linked model or FPGA execution",
        "physical_calls": len(plan["calls"]),
        "unique_adapters": len(used),
        "workspace_bytes": wbytes,
        "parameter_bytes": params["bytes"],
        "parameter_sha256": params["sha256"],
        "plan_sha256": sha(a.plan_dir / "physical-plan.json"),
        "generator_sha256": sha(__file__),
        "artifacts": {
            name: sha(a.output / name)
            for name in ("graph_runner.h", "graph_runner.c", "graph_runner.o")
        },
        "runtime_requirements": [
            "Provide workspace(0)==arena and compile every composite adapter with identical VIT_WORKSPACE_BYTES.",
            "Provide before/after hooks implementing the platform cache/DMA/AME visibility protocol and completion fence.",
            "Use a whole-model board linker script with enough parameter/input/workspace RAM and correct PMP operand window.",
            "Link the 96 exact existing case adapters/objects and 84 connector adapters/objects selected by the physical plan.",
            "Compare output to W8A8 semantic golden and report error/top1/hash/cycles; original FP32 difference is separate.",
        ],
    }
    (a.output / "runner-manifest.json").write_text(
        json.dumps(manifest, indent=2) + chr(10)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

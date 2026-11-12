"""Generate board output snapshots; retain the original launch oracle on host.

This module is shared by the four text examples. Production Triton kernels,
typed adapters, input fixtures and their full numerical host oracles stay in
their model directories. The board launch stops after dispatch and capture.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ITEM_BYTES = {"f32": 4, "i32": 4, "i8": 1}
DTYPES = {"f32": "<f4", "i32": "<i4", "i8": "|i1"}


def _output_indices(case):
    path = Path(case["directory"]).parent / "tools/output_spec.py"
    spec = importlib.util.spec_from_file_location(
        "output_spec_" + path.parent.parent.name.replace("-", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.output_indices(case["name"]))


def describe_outputs(case, shapes, configuration, export_invocation=None):
    indices = _output_indices(case)
    outputs, offset = [], 0
    for invocation in range(configuration["expected_invocations"]):
        for index in indices:
            argument = case["arguments"][index]
            shape = list(shapes[index])
            if len(shape) != argument["rank"] or len(shape) > 4:
                raise ValueError("Unexpected output rank: " + case["name"])
            nbytes = math.prod(shape) * ITEM_BYTES[argument["dtype"]]
            outputs.append(
                {
                    "name": f"r{invocation}_{argument['name']}",
                    "argument": index,
                    "invocation": invocation,
                    "dtype": DTYPES[argument["dtype"]],
                    "shape": shape,
                    "nbytes": nbytes,
                    "snapshot_offset": offset,
                }
            )
            offset = (offset + nbytes + 63) & ~63
    if not outputs or len(outputs) > 16:
        raise ValueError("Scheduler output declaration limit exceeded")
    all_outputs = outputs
    if export_invocation is not None:
        if configuration["expected_invocations"] != 2 or export_invocation not in (
            0,
            1,
        ):
            raise ValueError(
                "Split capture requires the original two-invocation launcher"
            )
        outputs, offset = [], 0
        for item in all_outputs:
            if item["invocation"] == export_invocation:
                item = {**item, "snapshot_offset": offset}
                outputs.append(item)
                offset = (offset + item["nbytes"] + 63) & ~63
        configuration.update(
            all_outputs=all_outputs,
            export_invocation=export_invocation,
            validation_requires_all_invocations=True,
        )
    configuration.update(
        outputs=outputs,
        output_snapshot_bytes=offset,
        validation_mode="scheduler_download_then_original_host_launch_oracle",
        launch_source_sha256=hashlib.sha256(
            Path(case["launch_source"]).read_bytes()
        ).hexdigest(),
        host_support_sha256=hashlib.sha256(
            (Path(case["directory"]).parent / "support.c").read_bytes()
        ).hexdigest(),
    )
    return configuration


def output_header(configuration):
    entries = []
    for output in configuration["outputs"]:
        shape = output["shape"] + [0] * (4 - len(output["shape"]))
        entries.append(
            "{"
            + ", ".join(
                [
                    json.dumps(output["name"]),
                    json.dumps(output["dtype"]),
                    str(len(output["shape"])),
                    "{" + ",".join(map(str, shape)) + "}",
                    str(output["nbytes"]),
                    str(output["snapshot_offset"]),
                ]
            )
            + "}"
        )
    return (
        f"#define VIT_BOARD_EXPECTED_OUTPUTS {len(entries)}u\n"
        f"#define VIT_BOARD_EXPORT_BYTES {configuration['output_snapshot_bytes']}u\n"
        "#define VIT_BOARD_OUTPUT_TABLE {" + ", ".join(entries) + "}\n"
    )


def capture_lines(case, export_invocation=None):
    indices = _output_indices(case)
    lines = []
    for ordinal, index in enumerate(indices):
        argument = case["arguments"][index]
        size = ITEM_BYTES[argument["dtype"]]
        slot = (
            f"slot * {len(indices)}u + {ordinal}u"
            if export_invocation is None
            else f"{ordinal}u"
        )
        condition = (
            "" if export_invocation is None else f"if (slot == {export_invocation}u) "
        )
        lines.append(
            f"  {condition}vit_board_capture_output({slot}, "
            f"(const unsigned char *)a{index}->aligned + (size_t)a{index}->offset * {size}u, "
            f"vit_board_output_bytes({slot}));"
        )
    return lines


def _code_only(source):
    """Mask comments and literals without changing source positions."""
    pattern = r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda match: re.sub(r"[^\n]", " ", match[0]), source)


def board_launch(case, configuration):
    source = Path(case["launch_source"]).read_text()
    code = _code_only(source)
    launch = re.search(r"\bint\s+launch\s*\(\s*void\s*\)\s*\{", code)
    if not launch:
        raise ValueError("Missing standalone launch(void): " + case["name"])
    calls = list(
        re.finditer(
            r"\b_mlir_ciface_kernel_" + re.escape(case["name"]) + r"\s*\(",
            code[launch.end() :],
        )
    )
    if len(calls) != 1:
        raise ValueError(
            "Board launch requires one reviewed typed call site: " + case["name"]
        )
    opening = launch.end() + calls[0].end() - 1
    depth = 1
    end = opening + 1
    while depth and end < len(code):
        depth += (code[end] == "(") - (code[end] == ")")
        end += 1
    while end < len(code) and code[end].isspace():
        end += 1
    if depth or code[end : end + 1] != ";":
        raise ValueError("Unrecognized typed call: " + case["name"])
    prefix = source[: end + 1]
    scopes = code[launch.end() : end + 1].count("{") - code[
        launch.end() : end + 1
    ].count("}")
    trials = configuration["expected_invocations"]
    if scopes != (1 if trials == 2 else 0):
        raise ValueError("Unreviewed loop/branch around typed call: " + case["name"])
    if trials == 2 and not re.search(
        r"for\s*\(\s*int\s+trial\s*=\s*0\s*;\s*trial\s*<\s*2\s*;",
        code[launch.end() : opening],
    ):
        raise ValueError("Expected original two-trial loop: " + case["name"])
    if trials == 2:
        prefix = re.sub(
            r"(for\s*\(\s*int\s+trial\s*=\s*0\s*;\s*trial\s*<\s*2\s*;[^)]*\)\s*\{)",
            r"\1\n    nh_e6_cache_writes();",
            prefix,
            count=1,
        )
    return (
        "/* Generated board launch: initialization and real kernel only.\n"
        " * Each output invocation is snapshotted for scheduler download.\n"
        " * Numerical validation runs with the original launch.c on host. */\n"
        + prefix
        + "\n"
        + "  }\n" * scopes
        + "  return vit_board_finish_outputs();\n}\n"
    )

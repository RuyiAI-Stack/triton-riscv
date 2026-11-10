#!/usr/bin/env python3
"""Self-test native output replay using independent, explicitly simulated files.

Compile the generated initialization-only launcher on the host to capture its
actual fixture inputs for every invocation. NumPy references produce simulated
output BINs, then the shared reader replays them through unchanged launch.c and
requires missing/truncated/corrupt output files to fail. No FPGA is executed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(SHARED))
from cases import describe, inventory  # noqa: E402 - local modules require the example path
from export import argument_shapes  # noqa: E402 - local modules require the example path
from oracles import DTYPES, reference  # noqa: E402 - local modules require the example path
from scheduler_outputs import board_launch  # noqa: E402 - local modules require the example path


def load_replay():
    spec = importlib.util.spec_from_file_location(
        "xlnet_returned_replay", SHARED / "verify_returned_outputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_source(case, shapes, invocations):
    params = ", ".join(
        f"MemRef{arg['rank']} *a{i}" for i, arg in enumerate(case["arguments"])
    )
    lines = [
        '#include "support.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "static unsigned invocation;",
        "void nh_e6_cache_writes(void) {}",
        f"int vit_board_finish_outputs(void) {{ return invocation == {invocations} ? 0 : 1; }}",
        f"void _mlir_ciface_kernel_{case['name']}({params}) {{",
        "  char path[128]; FILE *file;",
    ]
    for i, (arg, shape) in enumerate(zip(case["arguments"], shapes)):
        nbytes = math.prod(shape) * DTYPES[arg["dtype"]].itemsize
        lines += [
            f'  snprintf(path, sizeof(path), "input_r%u_{arg["name"]}.bin", invocation);',
            '  file = fopen(path, "wb"); if (!file) exit(2);',
            f"  if (fwrite(a{i}->aligned, 1, {nbytes}, file) != {nbytes} || fclose(file)) exit(2);",
        ]
    lines += [
        "  ++invocation;",
        "}",
        "extern int launch(void);",
        "int main(void) { return launch(); }",
        "",
    ]
    return "\n".join(lines)


def check(case, kernels, destination, replay, timeout):
    directory = destination / case["name"]
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = directory / "simulated-artifacts"
    artifacts.mkdir(exist_ok=True)
    config_path = kernels / case["name"] / "board/board_config.json"
    config = json.loads(config_path.read_text())
    shapes = argument_shapes(case)
    launch = directory / "capture-launch.c"
    launch.write_text(board_launch(case, config))
    source = directory / "capture.c"
    source.write_text(capture_source(case, shapes, config["expected_invocations"]))
    header = directory / "capture-api.h"
    header.write_text(
        "void nh_e6_cache_writes(void);\nint vit_board_finish_outputs(void);\n"
    )
    executable = directory / "capture-fixture"
    command = [
        "cc",
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-I",
        str(ROOT),
        "-include",
        str(header),
        str(launch),
        str(source),
        str(ROOT / "support.c"),
        "-lm",
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    (directory / "capture-compile.stdout").write_text(compiled.stdout)
    (directory / "capture-compile.stderr").write_text(compiled.stderr)
    if compiled.returncode:
        raise RuntimeError("capture fixture compilation failed")
    captured = subprocess.run(
        [str(executable)],
        check=False,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (directory / "capture.stdout").write_text(captured.stdout)
    (directory / "capture.stderr").write_text(captured.stderr)
    if captured.returncode:
        raise RuntimeError("capture fixture execution failed")
    inputs = []
    for invocation in range(config["expected_invocations"]):
        arrays = {}
        for argument, shape in zip(case["arguments"], shapes):
            path = directory / f"input_r{invocation}_{argument['name']}.bin"
            arrays[argument["name"]] = np.fromfile(
                path, dtype=DTYPES[argument["dtype"]]
            ).reshape(shape)
            inputs.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        expected = reference(case, arrays)
        for output in config["outputs"]:
            if output["invocation"] != invocation:
                continue
            name = case["arguments"][output["argument"]]["name"]
            value = np.ascontiguousarray(
                expected[name], dtype=np.dtype(output["dtype"])
            )
            if list(value.shape) != output["shape"] or value.nbytes != output["nbytes"]:
                raise RuntimeError(
                    "reference output shape/dtype differs from export config"
                )
            (artifacts / (output["name"] + ".bin")).write_bytes(value.tobytes())
    result = replay.verify(
        ROOT,
        case["name"],
        artifacts,
        config_path,
        directory / "replay",
        timeout=timeout,
        evidence_kind="simulated_selftest",
        regressions=True,
    )
    result["fixture_input_sha256"] = inputs
    result["fixture_capture_command"] = command
    result["board_execution_verified"] = False
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--kernels", type=Path, default=ROOT / "build/scheduler-output/kernels"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/scheduler-output/replay-selftest"
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    replay = load_replay()
    records = []
    for case in [describe(name) for name in args.case] if args.case else inventory():
        try:
            record = check(
                case, args.kernels.resolve(), args.output, replay, args.timeout
            )
        except Exception as error:  # noqa: BLE001 - Record failures and continue every case.
            record = {
                "case": case["name"],
                "status": "FAIL",
                "reason": str(error),
                "evidence_kind": "simulated_selftest",
                "board_execution_verified": False,
            }
        records.append(record)
        print(record["status"], case["name"], record.get("reason", ""), flush=True)
        status = (
            "PASS"
            if all(r["status"] == "SIMULATED_SELFTEST_PASS" for r in records)
            else "FAIL"
        )
        (args.output / "report.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "scope": "native output-reader/oracle self-test over simulated NumPy files; no FPGA execution",
                    "cases": records,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if all(r["status"] == "SIMULATED_SELFTEST_PASS" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Numeric simcheck driver for BOSCAME FPGA model 'vit-base-patch16-224'.

Inputs: exported kernels and CPU oracles / reference tensors for each registered
case.

Outputs: pass/fail and max-error per operator without needing the FPGA board.

When used: after AOT export, before or instead of a board run, from the model
directory.
"""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from boscame_paths import buddy_tools
from nonlinear_simcheck import (
    audit_nonlinear_program,
    check_nonlinear_sharded,
    collect_nonlinear_case,
    nonlinear_program_mask,
    selected_nonlinear_comparison,
)
from oracles import DTYPES, compare, make_inputs, output_names, reference

from export import (
    ROOT,
    argument_shapes,
    describe,
    digest,
    inventory,
    summary_report_path,
    tool,
)

sys.path.insert(0, str(buddy_tools()))
from boscame_buffer_sim import Machine, SimError  # noqa: E402


class SimulationBlocked(RuntimeError):
    """The real pipeline or strict simulator cannot execute the requested case."""


def scenarios_for(case):
    if case["family"] in (
        "per_token_quantization",
        "quantize_transpose",
        "dequantization",
        "head_layout",
    ):
        return [
            ("normal", False, "forward"),
            ("quantized-boundaries-tail-first", True, "tail-first"),
            ("quantized-boundaries-reverse", True, "reverse"),
        ]
    if case["family"] in ("gelu", "softmax"):
        return [
            ("normal", False, "forward"),
            ("boundaries-special-reverse", True, "reverse"),
            ("boundaries-special-forward", True, "forward"),
        ]
    if case["family"] in ("gemm_i8", "matmul_i8"):
        return [
            ("normal", False, "forward"),
            ("signed-wrap-tail-first", True, "tail-first"),
            ("signed-wrap-reverse", True, "reverse"),
        ]
    return [("normal", False, "reverse")]


def check_artifacts(case, output_root):
    output = Path(output_root) / case["name"]
    result_path = output / "build-result.json"
    if not result_path.is_file():
        raise SimulationBlocked("missing build-result.json; build this case first")
    result = json.loads(result_path.read_text())
    if result.get("status") != "PASS" or result.get("stage") == "ttir":
        raise SimulationBlocked(
            f"lowering {result.get('stage', 'unknown')}: {result.get('reason', 'no emitted BOSCAME module')}"
        )
    module = output / "kernel.boscame.mlir"
    manifest_path = output / "frontend.json"
    if not module.is_file() or not manifest_path.is_file():
        raise SimulationBlocked(
            "PASS build is missing emitted module or frontend manifest"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("accumulation") != case.get("accumulation"):
        raise SimulationBlocked("stale build: destination semantics differ")
    if manifest.get("kernel_module_sha256") != digest(case["kernel_source"]):
        raise SimulationBlocked(
            "stale build: current Triton source differs from frontend manifest"
        )
    for key in (
        "name",
        "family",
        "symbol",
        "signature",
        "constexprs",
        "grid",
        "shape",
        "arguments",
    ):
        if manifest.get(key) != case[key]:
            raise SimulationBlocked(
                "stale build: current case metadata differs at " + key
            )
    expected_hash = result.get("artifacts", {}).get(module.name)
    if expected_hash != digest(module):
        raise SimulationBlocked(
            "stale build: emitted module hash differs from successful build result"
        )
    ttir = output / "kernel.ttir"
    if not ttir.is_file() or manifest.get("ttir_sha256") != digest(ttir):
        raise SimulationBlocked("stale build: frontend TTIR hash differs from manifest")
    return module


def simulate_case(
    case,
    arrays,
    output_root,
    order="reverse",
    progress=True,
    machines=None,
    selected_programs=None,
):
    """Return arrays and metrics after interpreting the requested real programs.

    ``arrays`` is a mapping from the case's argument names to NumPy arrays.
    Inputs and outputs are copied, so callers can safely retain their own data.
    The same Machine and poisoned SPM banks persist throughout this launch.
    Passing a machines dictionary also preserves them between invocations.
    Nonlinear workers may select disjoint programs; unselected output bytes
    retain their initial values and only selected elements are compared.
    Without selected_programs, the complete real grid is executed.
    """
    module = check_artifacts(case, output_root)
    if case["family"] == "patch_embed":
        raise SimulationBlocked(
            "patch kernel materializes im2col only; projection, bias and NCHW composition have not been lowered and verified"
        )
    shapes = argument_shapes(case)
    initial = {}
    for argument, shape in zip(case["arguments"], shapes):
        name = argument["name"]
        value = np.asarray(arrays[name])
        if value.shape != tuple(shape) or value.dtype != DTYPES[argument["dtype"]]:
            raise ValueError(
                f"{name}: expected {shape} {DTYPES[argument['dtype']]}, got {value.shape} {value.dtype}"
            )
        initial[name] = np.ascontiguousarray(value).copy()
    expected = reference(case, initial)
    guard = bytes([0xC7]) * 1024
    buffers = {
        f"%arg{i}": bytearray(initial[arg["name"]].tobytes() + guard)
        for i, arg in enumerate(case["arguments"])
    }
    original = {key: bytes(data) for key, data in buffers.items()}
    limits = {key: len(data) - len(guard) for key, data in buffers.items()}
    machine_key = (case["name"], digest(module))
    machine = machines.get(machine_key) if machines is not None else None
    if machine is None:
        with tempfile.TemporaryDirectory(prefix="vit-boscame-sim-") as directory:
            loops = Path(directory) / "kernel.loops.mlir"
            try:
                process = subprocess.run(
                    [
                        tool("buddy-opt"),
                        str(module),
                        "--convert-linalg-to-loops",
                        "--lower-affine",
                        "--canonicalize",
                        "-o",
                        str(loops),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except OSError as error:
                raise SimulationBlocked(
                    "cannot invoke buddy-opt: " + str(error)
                ) from error
            if process.returncode:
                raise SimulationBlocked(
                    "cannot prepare emitted module for strict simulation: "
                    + process.stdout.strip()
                )
            machine = Machine(loops, case["symbol"])
        for bank in machine.spm:
            bank[:] = bytes([0xD3]) * len(bank)
        if machines is not None:
            machines[machine_key] = machine
    access_start, dma_start = len(machine.accesses), len(machine.dma_log)
    frame_start = len(getattr(machine, "frame_accesses", []))
    ame_frame_start = len(getattr(machine, "ame_frame_accesses", []))
    frame_log_start = len(getattr(machine, "frame_log", []))
    grid = case["grid"]
    programs = list(itertools.product(*(range(extent) for extent in grid)))
    if selected_programs is not None:
        if case["family"] in ("gelu", "softmax"):
            nonlinear_program_mask(case, selected_programs)
        else:
            raise ValueError("program subsets require a GELU or Softmax case")
        programs = list(selected_programs)
    if order == "reverse":
        programs.reverse()
    elif order == "tail-first":
        programs.sort(key=lambda point: (-point[0], point[1], point[2]))
    elif order != "forward":
        raise ValueError("unknown program order: " + order)
    count = len(case["arguments"])
    nonlinear_bytes = {"%arg0": 0, "%arg1": 0}
    for launched, point in enumerate(programs, 1):
        scalars = {f"%arg{count + i}": value for i, value in enumerate([*grid, *point])}
        try:
            program_start = (
                len(machine.accesses),
                len(machine.dma_log),
                len(machine.matrix_log),
            )
            machine.launch(buffers, scalars)
        except SimError as error:
            message = str(error)
            if "unsupported" in message or "only " in message:
                raise SimulationBlocked(
                    f"strict simulator at program {point}: {message}"
                ) from error
            raise AssertionError(
                f"strict simulator at program {point}: {message}"
            ) from error
        if case["family"] in ("gelu", "softmax"):
            audit = audit_nonlinear_program(
                case, point, machine.accesses[program_start[0] :]
            )
            for name, value in audit.items():
                nonlinear_bytes[name] += value
        if progress and (
            launched == 1
            or launched == len(programs)
            or launched % max(1, len(programs) // 10) == 0
        ):
            print(
                f"  {case['name']}: {launched}/{len(programs)} programs ({order})",
                flush=True,
            )
    writable = set(output_names(case))
    actual = {}
    for i, argument in enumerate(case["arguments"]):
        key, name = f"%arg{i}", argument["name"]
        if buffers[key][-len(guard) :] != guard:
            raise AssertionError(name + ": DDR guard was modified")
        if name not in writable and bytes(buffers[key]) != original[key]:
            raise AssertionError(name + ": input buffer was modified")
        actual[name] = (
            np.frombuffer(
                buffers[key],
                dtype=DTYPES[argument["dtype"]],
                count=initial[name].size,
            )
            .copy()
            .reshape(initial[name].shape)
        )
    for name, start, end, action in machine.accesses[access_start:]:
        if name not in limits or not 0 <= start < end <= limits[name]:
            raise AssertionError(
                f"DDR access outside logical argument: {(name, start, end, action)}"
            )
        index = int(name.removeprefix("%arg"))
        if action == "write" and case["arguments"][index]["name"] not in writable:
            raise AssertionError("write to read-only input: " + name)
    if case["family"] == "matmul_i8" and any(
        name == "%arg2" and action == "read"
        for name, _, _, action in machine.accesses[access_start:]
    ):
        raise AssertionError("overwrite operation read old external C")
    private_frames = {
        frame["name"]: frame for frame in getattr(machine, "private_frames", [])
    }
    frame_events = getattr(machine, "frame_log", [])[frame_log_start:]
    acquired = [event[1] for event in frame_events if event[0] == "acquire"]
    released = [event[1] for event in frame_events if event[0] == "release"]
    if len(set(acquired)) != len(acquired) or sorted(acquired) != sorted(released):
        raise AssertionError("private DDR frames lack matching invocation releases")
    for name in acquired:
        if name not in private_frames or private_frames[name]["alive"]:
            raise AssertionError("private DDR frame remains live after invocation")
    frame_bytes = {kind: 0 for kind in ("read", "write")}
    ame_frame_bytes = dict(frame_bytes)
    for accesses, counts in (
        (getattr(machine, "frame_accesses", [])[frame_start:], frame_bytes),
        (getattr(machine, "ame_frame_accesses", [])[ame_frame_start:], ame_frame_bytes),
    ):
        for name, start, end, action in accesses:
            frame = private_frames.get(name)
            if (
                frame is None
                or name not in acquired
                or action not in counts
                or not frame["begin"] <= start < end <= frame["end"]
            ):
                raise AssertionError("private DDR access outside invocation frame")
            counts[action] += end - start
    comparison_actual, comparison_expected = actual, expected
    nonlinear_metrics = None
    if case["family"] in ("gelu", "softmax"):
        comparison_actual, comparison_expected, nonlinear_metrics = (
            selected_nonlinear_comparison(case, programs, actual, expected, initial)
        )
    max_abs = compare(case, comparison_actual, comparison_expected)
    return actual, {
        "programs": len(programs),
        "total_programs": math.prod(grid),
        "outputs": sum(value.size for value in comparison_expected.values()),
        "max_abs": max_abs,
        "guard_bytes_checked": len(guard) * len(buffers),
        "guard_errors": 0,
        "input_errors": 0,
        "readonly_bytes_checked": sum(
            initial[name].nbytes for name in initial if name not in writable
        ),
        "outside_selected_output_unchanged": True,
        "module_sha256": digest(module),
        "kernel_module_sha256": digest(case["kernel_source"]),
        "order": order,
        "ddr_accesses": len(machine.accesses) - access_start,
        "dma_transfers": len(machine.dma_log) - dma_start,
        "private_ddr": {
            "frames_acquired": len(acquired),
            "frames_released": len(released),
            "frame_bytes": frame_bytes,
            "ame_bytes": ame_frame_bytes,
            "poison": getattr(machine, "private_poison", None),
        },
        "argument_bytes": {
            argument["name"]: {
                action: sum(
                    end - start
                    for name, start, end, kind in machine.accesses[access_start:]
                    if name == f"%arg{index}" and kind == action
                )
                for action in ("read", "write")
            }
            for index, argument in enumerate(case["arguments"])
        },
        "output_sha256": {
            name: hashlib.sha256(comparison_actual[name].tobytes()).hexdigest()
            for name in comparison_expected
        },
        **(
            {
                "nonlinear_error": nonlinear_metrics,
                "nonlinear_ddr_bytes": nonlinear_bytes,
            }
            if nonlinear_metrics is not None
            else {}
        ),
        **({"selected_programs": programs} if selected_programs is not None else {}),
    }


def check_case(case, output_root):
    if case["family"] in ("gelu", "softmax"):
        return check_nonlinear_sharded(case, output_root, 1)
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": math.prod(case["grid"]),
        "outputs": 0,
        "max_abs": None,
        "reason": "",
        "module_sha256": None,
        "full_grid_programs": math.prod(case["grid"]),
    }
    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        module = check_artifacts(case, output_root)
        record["module_sha256"] = digest(module)
        record["kernel_module_sha256"] = digest(case["kernel_source"])
        scenarios = scenarios_for(case)
        record["total_programs"] *= len(scenarios)
        record["comparison_rounds"] = len(scenarios)
        runs = []
        machines = {}
        for name, stress, order in scenarios:
            arrays = make_inputs(case, argument_shapes(case), stress=stress)
            _, metrics = simulate_case(
                case, arrays, output_root, order=order, machines=machines
            )
            runs.append({"name": name, **metrics})
            record["programs"] += metrics["programs"]
            record["outputs"] += metrics["outputs"]
            record["max_abs"] = max(record["max_abs"] or 0, metrics["max_abs"])
        if len(runs) > 2 and runs[1]["output_sha256"] != runs[2]["output_sha256"]:
            raise AssertionError(
                "identical stress inputs depend on program launch order"
            )
        record.update(status="PASS", runs=runs)
    except SimulationBlocked as error:
        record["reason"] = str(error)
    except Exception as error:
        record.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    (output / "simcheck-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{record['status']}: {case['name']} ({record['programs']}/{record['total_programs']})",
        flush=True,
    )
    return record


def collect_case(case, output_root):
    """Reuse only complete simulation evidence for the current emitted module."""
    if case["family"] in ("gelu", "softmax"):
        return collect_nonlinear_case(case, output_root)
    path = Path(output_root) / case["name"] / "simcheck-result.json"
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": math.prod(case["grid"]),
        "outputs": 0,
        "max_abs": None,
        "reason": "",
        "module_sha256": None,
    }
    try:
        module = check_artifacts(case, output_root)
        record["module_sha256"] = digest(module)
        if not path.is_file():
            raise SimulationBlocked("no simcheck-result.json for this successful build")
        saved = json.loads(path.read_text())
        if saved.get("module_sha256") != digest(module):
            raise SimulationBlocked("stale simulation: emitted module hash changed")
        if saved.get("status") != "PASS":
            return saved
        scenarios = scenarios_for(case)
        rounds = len(scenarios)
        full_grid = math.prod(case["grid"])
        runs = saved.get("runs", [])
        outputs = set(output_names(case))
        per_round = sum(
            math.prod(shape)
            for arg, shape in zip(case["arguments"], argument_shapes(case))
            if arg["name"] in outputs
        )
        if (
            saved.get("name") != case["name"]
            or len(runs) != rounds
            or saved.get("programs") != full_grid * rounds
            or saved.get("total_programs") != full_grid * rounds
            or saved.get("outputs") != per_round * rounds
        ):
            raise SimulationBlocked(
                "incomplete simulation: complete grid/round/output evidence is missing"
            )
        for run in runs:
            if (
                run.get("programs") != full_grid
                or run.get("total_programs") != full_grid
                or run.get("outputs") != per_round
                or run.get("module_sha256") != digest(module)
                or set(run.get("output_sha256", {})) != outputs
            ):
                raise SimulationBlocked(
                    "incomplete simulation: a round lacks complete current-module evidence"
                )
        if rounds == 3 and (
            [(run.get("name"), run.get("order")) for run in runs]
            != [(name, order) for name, _, order in scenarios]
            or runs[1]["output_sha256"] != runs[2]["output_sha256"]
        ):
            raise SimulationBlocked("incomplete numerical/launch-order stress evidence")
        saved.update(
            kernel_module_sha256=digest(case["kernel_source"]),
            full_grid_programs=full_grid,
            comparison_rounds=rounds,
        )
        return saved
    except SimulationBlocked as error:
        record["reason"] = str(error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Combine existing results after verifying current source/module hashes and complete grid/round coverage; do not rerun simulations",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--nonlinear-workers",
        type=int,
        default=1,
        help="interpret GELU/Softmax program shards with one dirty-SPM Machine per shard across three rounds",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.all and args.cases:
        parser.error("use --all or --case, not both")
    if not args.all and not args.cases:
        parser.error("choose --all or at least one --case")
    if args.report is None:
        args.report = summary_report_path(args.output, args.cases, "simcheck")
    cases = (
        [describe(name) for name in dict.fromkeys(args.cases)]
        if args.cases
        else inventory()
    )
    if args.nonlinear_workers < 1 or (
        args.nonlinear_workers > 1
        and any(case["family"] not in ("gelu", "softmax") for case in cases)
    ):
        parser.error(
            "--nonlinear-workers requires positive workers and GELU/Softmax-only cases"
        )
    if args.collect:
        run = collect_case
    elif all(case["family"] in ("gelu", "softmax") for case in cases):

        def run(case, output):
            return check_nonlinear_sharded(case, output, args.nonlinear_workers)
    else:
        run = check_case
    results = [run(case, args.output) for case in cases]
    if args.collect:
        for result in results:
            directory = args.output / result["name"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "simcheck-result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                f"{result['status']}: {result['name']} ({result['programs']}/{result['total_programs']}; collected)"
            )
    status = (
        "FAIL"
        if any(result["status"] == "FAIL" for result in results)
        else "PARTIAL"
        if any(result["status"] == "BLOCKED" for result in results)
        else "PASS"
    )
    report = {"schema": 1, "status": status, "cases": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

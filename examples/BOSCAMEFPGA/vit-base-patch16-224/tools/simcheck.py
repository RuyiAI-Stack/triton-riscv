#!/usr/bin/env python3
"""Execute complete independent ViT launch grids in the strict BOSCAME model."""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing
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
    if case["family"] in ("gelu", "softmax"):
        return [
            ("normal", False, "forward"),
            ("boundaries-special-reverse", True, "reverse"),
            ("boundaries-special-forward", True, "forward"),
        ]
    if case["family"] in ("matmul_i8", "matmul_i8_overwrite"):
        return [
            ("normal", False, "forward"),
            ("signed-wrap-tail-first", True, "tail-first"),
            ("signed-wrap-reverse", True, "reverse"),
        ]
    if case["family"] == "attention_pv":
        return [
            ("normal", False, "forward"),
            ("ordered-blocks-tail-first", True, "tail-first"),
            ("ordered-blocks-reverse", True, "reverse"),
        ]
    return [("normal", False, "reverse")]


def pv_program_mask(case, programs):
    """Validate a nonempty disjoint subset of real PV launch programs."""
    if case["family"] != "attention_pv":
        raise ValueError("program subsets are restricted to PV validation")
    _, m, n, _ = case["shape"]
    bm, bn = (case["constexprs"][key] for key in ("BM", "BN"))
    mask = np.zeros(argument_shapes(case)[2], dtype=bool)
    if not programs:
        raise ValueError("empty PV program selection")
    for point in programs:
        if len(point) != 3 or any(
            not isinstance(value, int) or not 0 <= value < extent
            for value, extent in zip(point, case["grid"])
        ):
            raise ValueError("out-of-grid PV program: " + str(point))
        x, y, head = point
        region = (
            head,
            slice(x * bm, min((x + 1) * bm, m)),
            slice(y * bn, min((y + 1) * bn, n)),
        )
        if np.any(mask[region]):
            raise ValueError("overlapping PV programs: " + str(point))
        mask[region] = True
    return mask


def check_pv_selected_output(case, programs, actual, expected, initial):
    """Compare authorized tiles and require all other output bytes unchanged."""
    mask = pv_program_mask(case, programs)
    bits = actual.view(np.uint32)
    if not np.array_equal(bits[mask], expected.view(np.uint32)[mask]):
        raise AssertionError("PV ordered f32 output is not bit-exact")
    if not np.array_equal(bits[~mask], initial.view(np.uint32)[~mask]):
        raise AssertionError("PV program subset wrote an unauthorized C tile")
    return mask


def audit_pv_program(case, point, accesses, dma, matrix):
    """Check this launch's exact logical rows, not only backing-buffer bounds."""
    _, m, n, k = case["shape"]
    bm, bn, bk = (case["constexprs"][key] for key in ("BM", "BN", "BK"))
    x, y, head = point
    m0, n0 = x * bm, y * bn
    rows, columns = min(bm, m - m0), min(bn, n - n0)
    regions = {
        "%arg0": (m, k, m0, m0 + rows, 0, k, "read"),
        "%arg1": (k, n, 0, k, n0, n0 + columns, "read"),
        "%arg2": (m, n, m0, m0 + rows, n0, n0 + columns, "write"),
    }
    byte_counts = dict.fromkeys(regions, 0)
    intervals = {name: [] for name in regions}
    for name, start, end, action in accesses:
        height, width, r0, r1, c0, c1, expected_action = regions[name]
        first, last = start // 4, end // 4 - 1
        row, column = divmod(first - head * height * width, width)
        last_row, last_column = divmod(last - head * height * width, width)
        if (
            start % 4
            or end % 4
            or action != expected_action
            or not r0 <= row < r1
            or last_row != row
            or not c0 <= column <= last_column < c1
        ):
            raise AssertionError(
                f"PV logical row violation at {point}: {(name, start, end, action)}"
            )
        byte_counts[name] += end - start
        intervals[name].append((start, end))
    expected_bytes = {
        "%arg0": rows * k * 4,
        "%arg1": k * columns * 4,
        "%arg2": rows * columns * 4,
    }
    if byte_counts != expected_bytes:
        raise AssertionError(
            f"PV transfer coverage at {point}: {byte_counts} != {expected_bytes}"
        )
    for name, spans in intervals.items():
        ordered = sorted(spans)
        if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
            raise AssertionError(f"PV duplicate DDR access at {point}: {name}")
    # The current strided A cannot use the 64-byte-aligned DMA route. ACC32
    # copies move raw f32 bits; all arithmetic remains the ordered RVV path.
    geometry = {}
    if (k * 4) % 64:
        a_copies = [event for event in matrix if event["source"] == "%arg0"]
        if len(a_copies) != math.ceil(k / bk):
            raise AssertionError(
                f"PV A matrix-copy count at {point}: {len(a_copies)}"
            )
        for block, event in enumerate(a_copies):
            valid = min(bk, k - block * bk)
            expected = {
                # Offsets are relative to the already offset DDR subview;
                # accesses above independently validate absolute addresses.
                "source_offset": 0,
                "rows": rows,
                "row_bytes": valid * 4,
                "source_pitch": k * 4,
                "target_pitch": bk * 4,
                "instruction_bits": 32,
                "register_class": "acc",
            }
            if any(event[key] != value for key, value in expected.items()):
                raise AssertionError(
                    f"PV A matrix-copy geometry at {point}: {event}"
                )
            label = (
                ("full-M" if rows == bm else "tail-M")
                + "/"
                + ("full-K" if valid == bk else "tail-K")
            )
            geometry[label] = geometry.get(label, 0) + 1
        if any(event[0] == "%arg0" for event in dma):
            raise AssertionError("PV unaligned A incorrectly used DMA")
    b_packets = [event for event in dma if event[0] == "%arg1"]
    if len(b_packets) != k or any(
        event[4] != columns * 4 for event in b_packets
    ):
        raise AssertionError(f"PV B DMA packets at {point}: {len(b_packets)}")
    if any(event["source"] == "%arg1" for event in matrix):
        raise AssertionError(
            "PV legal B DMA unexpectedly became a matrix transfer"
        )
    return {
        "a_matrix_geometry": geometry,
        "b_dma_packets": len(b_packets),
        "ddr_bytes": byte_counts,
    }


def check_artifacts(case, output_root):
    output = Path(output_root) / case["name"]
    result_path = output / "build-result.json"
    if not result_path.is_file():
        raise SimulationBlocked(
            "missing build-result.json; build this case first"
        )
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
        raise SimulationBlocked(
            "stale build: frontend TTIR hash differs from manifest"
        )
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
    Internal PV workers may select disjoint programs; unselected output bytes
    must then retain their initial values and only selected tiles are compared.
    These subsets require assemble_pv_round's independent full-grid acceptance.
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
        if (
            value.shape != tuple(shape)
            or value.dtype != DTYPES[argument["dtype"]]
        ):
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
        with tempfile.TemporaryDirectory(
            prefix="vit-boscame-sim-"
        ) as directory:
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
    grid = case["grid"]
    programs = list(itertools.product(*(range(extent) for extent in grid)))
    if selected_programs is not None:
        if case["family"] in ("gelu", "softmax"):
            nonlinear_program_mask(case, selected_programs)
        else:
            pv_program_mask(case, selected_programs)
        programs = list(selected_programs)
    if order == "reverse":
        programs.reverse()
    elif order == "tail-first":
        programs.sort(key=lambda point: (-point[0], point[1], point[2]))
    elif order != "forward":
        raise ValueError("unknown program order: " + order)
    count = len(case["arguments"])
    pv_audit = {
        "a_matrix_geometry": {},
        "b_dma_packets": 0,
        "ddr_bytes": dict.fromkeys(("%arg0", "%arg1", "%arg2"), 0),
    }
    nonlinear_bytes = {"%arg0": 0, "%arg1": 0}
    for launched, point in enumerate(programs, 1):
        scalars = {
            f"%arg{count + i}": value for i, value in enumerate([*grid, *point])
        }
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
        if case["family"] == "attention_pv":
            access_index, dma_index, matrix_index = program_start
            audit = audit_pv_program(
                case,
                point,
                machine.accesses[access_index:],
                machine.dma_log[dma_index:],
                machine.matrix_log[matrix_index:],
            )
            pv_audit["b_dma_packets"] += audit["b_dma_packets"]
            for key in ("a_matrix_geometry", "ddr_bytes"):
                for label, value in audit[key].items():
                    pv_audit[key][label] = pv_audit[key].get(label, 0) + value
        elif case["family"] in ("gelu", "softmax"):
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
        if (
            action == "write"
            and case["arguments"][index]["name"] not in writable
        ):
            raise AssertionError("write to read-only input: " + name)
    if case["family"] == "matmul_i8_overwrite" and any(
        name == "%arg2" and action == "read"
        for name, _, _, action in machine.accesses[access_start:]
    ):
        raise AssertionError("overwrite matmul read old external C")
    comparison_actual, comparison_expected = actual, expected
    nonlinear_metrics = None
    if case["family"] == "attention_pv":
        mask = check_pv_selected_output(
            case, programs, actual["C"], expected["C"], initial["C"]
        )
        comparison_actual = {"C": actual["C"][mask]}
        comparison_expected = {"C": expected["C"][mask]}
    elif case["family"] in ("gelu", "softmax"):
        comparison_actual, comparison_expected, nonlinear_metrics = (
            selected_nonlinear_comparison(
                case, programs, actual, expected, initial
            )
        )
    max_abs = compare(case, comparison_actual, comparison_expected)
    return actual, {
        "programs": len(programs),
        "total_programs": math.prod(grid),
        "outputs": sum(value.size for value in comparison_expected.values()),
        "max_abs": max_abs,
        "module_sha256": digest(module),
        "kernel_module_sha256": digest(case["kernel_source"]),
        "order": order,
        "ddr_accesses": len(machine.accesses) - access_start,
        "dma_transfers": len(machine.dma_log) - dma_start,
        "output_sha256": {
            name: hashlib.sha256(comparison_actual[name].tobytes()).hexdigest()
            for name in comparison_expected
        },
        **(
            {
                "pv_access_audit": pv_audit,
                "bit_exact_f32": True,
                "reference_sha256": {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in comparison_expected.items()
                },
            }
            if case["family"] == "attention_pv"
            else {}
        ),
        **(
            {
                "nonlinear_error": nonlinear_metrics,
                "nonlinear_ddr_bytes": nonlinear_bytes,
            }
            if nonlinear_metrics is not None
            else {}
        ),
        **(
            {"selected_programs": programs}
            if selected_programs is not None
            else {}
        ),
    }


def pv_shards(case):
    """Keep all M tiles of one (head, N tile) on the same dirty SPM."""
    return [
        [(x, y, head) for x in range(case["grid"][0])]
        for head in range(case["grid"][2])
        for y in range(case["grid"][1])
    ]


def pv_shard_worker(case, output_root, programs):
    """Interpret all three real rounds, retaining one Machine across them."""
    machines = {}
    mask = pv_program_mask(case, programs)
    runs = []
    for name, stress, order in scenarios_for(case):
        arrays = make_inputs(case, argument_shapes(case), stress=stress)
        actual, metrics = simulate_case(
            case,
            arrays,
            output_root,
            order=order,
            progress=False,
            machines=machines,
            selected_programs=programs,
        )
        if len(machines) != 1:
            raise AssertionError("PV shard did not retain one Machine/SPM")
        runs.append(
            {
                "name": name,
                "metrics": metrics,
                "values": actual["C"][mask].copy(),
            }
        )
    return {"programs": programs, "runs": runs}


def assemble_pv_round(case, pieces, expected):
    """Reconstruct every output exclusively from disjoint executed slices."""
    shape = argument_shapes(case)[2]
    assembled = np.empty(shape, dtype=DTYPES["f32"])
    covered = np.zeros(shape, dtype=bool)
    programs = set()
    for piece in pieces:
        points = piece["programs"]
        mask = pv_program_mask(case, points)
        point_set = {tuple(point) for point in points}
        if programs & point_set or np.any(covered & mask):
            raise AssertionError(
                "PV aggregate contains duplicate program/output coverage"
            )
        metrics, values = piece["metrics"], piece["values"]
        if (
            metrics["programs"] != len(points)
            or metrics["outputs"] != np.count_nonzero(mask)
            or {tuple(point) for point in metrics["selected_programs"]}
            != point_set
            or len(metrics["selected_programs"]) != len(points)
            or values.shape != (np.count_nonzero(mask),)
            or values.dtype != DTYPES["f32"]
            or hashlib.sha256(values.tobytes()).hexdigest()
            != metrics["output_sha256"]["C"]
            or metrics.get("bit_exact_f32") is not True
            or metrics.get("reference_sha256") != metrics["output_sha256"]
        ):
            raise AssertionError(
                "PV aggregate lacks executed slice/hash evidence"
            )
        assembled[mask] = values
        covered |= mask
        programs |= point_set
    complete = set(
        itertools.product(*(range(extent) for extent in case["grid"]))
    )
    if programs != complete or not np.all(covered):
        raise AssertionError(
            "PV aggregate is missing full-grid programs/outputs"
        )
    if not np.array_equal(assembled.view(np.uint32), expected.view(np.uint32)):
        raise AssertionError("PV aggregate ordered f32 output is not bit-exact")
    return assembled


def check_pv_sharded(case, output_root, workers):
    if case["family"] != "attention_pv" or workers <= 1:
        raise ValueError("parallel shards require PV and at least two workers")
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": math.prod(case["grid"]) * 3,
        "outputs": 0,
        "max_abs": None,
        "reason": "",
        "execution": "parallel-shards",
        "order_scope": "within each fixed (head, N tile) shard; no global inter-shard order",
        "spm_lifetime": "one Machine per shard, poisoned once and retained across all selected programs and three rounds",
        "comparison_rounds": 3,
        "full_grid_programs": math.prod(case["grid"]),
    }
    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        module = check_artifacts(case, output_root)
        record["module_sha256"] = digest(module)
        record["kernel_module_sha256"] = digest(case["kernel_source"])
        shards = pv_shards(case)
        record.update(
            workers=min(workers, len(shards)), shard_count=len(shards)
        )
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=record["workers"],
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(pv_shard_worker, case, output_root, points)
                for points in shards
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"  {case['name']}: {len(results)}/{len(shards)} shards, three rounds each",
                    flush=True,
                )
        results.sort(key=lambda result: result["programs"][0])
        runs = []
        for index, (name, stress, order) in enumerate(scenarios_for(case)):
            expected = reference(
                case, make_inputs(case, argument_shapes(case), stress=stress)
            )["C"]
            pieces = [
                {"programs": result["programs"], **result["runs"][index]}
                for result in results
            ]
            assembled = assemble_pv_round(case, pieces, expected)
            audit = {
                "a_matrix_geometry": {},
                "b_dma_packets": 0,
                "ddr_bytes": dict.fromkeys(("%arg0", "%arg1", "%arg2"), 0),
            }
            for piece in pieces:
                metrics = piece["metrics"]
                if (
                    metrics["module_sha256"] != digest(module)
                    or metrics["kernel_module_sha256"]
                    != digest(case["kernel_source"])
                    or piece["name"] != name
                    or metrics["order"] != order
                ):
                    raise AssertionError(
                        "PV shard source/module/scenario differs"
                    )
                per_shard = metrics["pv_access_audit"]
                audit["b_dma_packets"] += per_shard["b_dma_packets"]
                for key in ("a_matrix_geometry", "ddr_bytes"):
                    for label, value in per_shard[key].items():
                        audit[key][label] = audit[key].get(label, 0) + value
            hashes = {"C": hashlib.sha256(assembled.tobytes()).hexdigest()}
            runs.append(
                {
                    "name": name,
                    "order": "shard-" + order,
                    "programs": math.prod(case["grid"]),
                    "total_programs": math.prod(case["grid"]),
                    "outputs": assembled.size,
                    "max_abs": 0.0,
                    "bit_exact_f32": True,
                    "module_sha256": digest(module),
                    "kernel_module_sha256": digest(case["kernel_source"]),
                    "output_sha256": hashes,
                    "reference_sha256": {
                        "C": hashlib.sha256(expected.tobytes()).hexdigest()
                    },
                    "ddr_accesses": sum(
                        piece["metrics"]["ddr_accesses"] for piece in pieces
                    ),
                    "dma_transfers": sum(
                        piece["metrics"]["dma_transfers"] for piece in pieces
                    ),
                    "pv_access_audit": audit,
                }
            )
        if runs[1]["output_sha256"] != runs[2]["output_sha256"]:
            raise AssertionError("PV shard launch order changed output")
        record.update(
            status="PASS",
            runs=runs,
            programs=sum(run["programs"] for run in runs),
            outputs=sum(run["outputs"] for run in runs),
            max_abs=0.0,
            shards=[
                {
                    "programs": result["programs"],
                    "runs": [
                        {"name": run["name"], **run["metrics"]}
                        for run in result["runs"]
                    ],
                }
                for result in results
            ],
        )
    except SimulationBlocked as error:
        record["reason"] = str(error)
    except Exception as error:
        record.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    (output / "simcheck-result.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(
        f"{record['status']}: {case['name']} ({record['programs']}/{record['total_programs']}; sharded)",
        flush=True,
    )
    return record


def validate_pv_shard_evidence(case, saved, module_hash):
    """Reject incomplete/overlapping stored shard metadata during collection."""
    shards = saved.get("shards", [])
    if len(shards) != len(pv_shards(case)) or saved.get("shard_count") != len(
        shards
    ):
        raise SimulationBlocked("incomplete PV shard count")
    references = {
        stress: reference(
            case, make_inputs(case, argument_shapes(case), stress=stress)
        )["C"]
        for stress in (False, True)
    }
    fixed_shards = {frozenset(points) for points in pv_shards(case)}
    points_seen = set()
    for shard in shards:
        points = shard.get("programs", [])
        try:
            mask = pv_program_mask(case, points)
        except ValueError as error:
            raise SimulationBlocked(
                "invalid PV shard program selection"
            ) from error
        point_set = {tuple(point) for point in points}
        if frozenset(point_set) not in fixed_shards:
            raise SimulationBlocked(
                "PV shard is not one complete fixed (head, N tile) M set"
            )
        if points_seen & point_set:
            raise SimulationBlocked("duplicate PV shard programs")
        points_seen |= point_set
        runs = shard.get("runs", [])
        if len(runs) != 3:
            raise SimulationBlocked("incomplete PV shard rounds")
        for run, (name, stress, order) in zip(runs, scenarios_for(case)):
            expected = references[stress][mask]
            expected_hash = {
                "C": hashlib.sha256(expected.tobytes()).hexdigest()
            }
            ordered_points = sorted(point_set)
            if order == "reverse":
                ordered_points.reverse()
            elif order == "tail-first":
                ordered_points.sort(
                    key=lambda point: (-point[0], point[1], point[2])
                )
            if (
                run.get("name") != name
                or run.get("order") != order
                or run.get("module_sha256") != module_hash
                or run.get("kernel_module_sha256")
                != digest(case["kernel_source"])
                or run.get("programs") != len(points)
                or run.get("outputs") != np.count_nonzero(mask)
                or len(run.get("selected_programs", [])) != len(points)
                or {tuple(point) for point in run["selected_programs"]}
                != point_set
                or [tuple(point) for point in run["selected_programs"]]
                != ordered_points
                or run.get("bit_exact_f32") is not True
                or run.get("output_sha256") != expected_hash
                or run.get("reference_sha256") != expected_hash
                or "pv_access_audit" not in run
            ):
                raise SimulationBlocked(
                    "incomplete PV shard numerical/coverage evidence"
                )
    complete = set(
        itertools.product(*(range(extent) for extent in case["grid"]))
    )
    if points_seen != complete:
        raise SimulationBlocked("PV shards do not cover the full grid")


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
        machines = {} if case["family"] == "attention_pv" else None
        for name, stress, order in scenarios:
            arrays = make_inputs(case, argument_shapes(case), stress=stress)
            _, metrics = simulate_case(
                case, arrays, output_root, order=order, machines=machines
            )
            runs.append({"name": name, **metrics})
            record["programs"] += metrics["programs"]
            record["outputs"] += metrics["outputs"]
            record["max_abs"] = max(record["max_abs"] or 0, metrics["max_abs"])
        if (
            len(runs) > 2
            and runs[1]["output_sha256"] != runs[2]["output_sha256"]
        ):
            raise AssertionError(
                "identical stress inputs depend on program launch order"
            )
        record.update(status="PASS", runs=runs)
    except SimulationBlocked as error:
        record["reason"] = str(error)
    except Exception as error:
        record.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    (output / "simcheck-result.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
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
            raise SimulationBlocked(
                "no simcheck-result.json for this successful build"
            )
        saved = json.loads(path.read_text())
        if saved.get("module_sha256") != digest(module):
            raise SimulationBlocked(
                "stale simulation: emitted module hash changed"
            )
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
            if case["family"] == "attention_pv" and (
                run.get("bit_exact_f32") is not True
                or run.get("reference_sha256") != run.get("output_sha256")
                or "pv_access_audit" not in run
            ):
                raise SimulationBlocked(
                    "incomplete PV bit-exact/transfer-geometry evidence"
                )
        order_prefix = ""
        if saved.get("execution") == "parallel-shards":
            validate_pv_shard_evidence(case, saved, digest(module))
            order_prefix = "shard-"
        if rounds == 3 and (
            [(run.get("name"), run.get("order")) for run in runs]
            != [(name, order_prefix + order) for name, _, order in scenarios]
            or runs[1]["output_sha256"] != runs[2]["output_sha256"]
        ):
            raise SimulationBlocked(
                "incomplete numerical/launch-order stress evidence"
            )
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
        "--pv-workers",
        type=int,
        default=1,
        help="explicitly interpret PV full-grid (head,N-tile) shards in separate processes; each retains dirty SPM across its three rounds",
    )
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
    if args.pv_workers < 1 or (
        args.pv_workers > 1
        and any(case["family"] != "attention_pv" for case in cases)
    ):
        parser.error("--pv-workers requires positive workers and PV-only cases")
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
    elif args.pv_workers > 1:

        def run(case, output):
            return check_pv_sharded(case, output, args.pv_workers)
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

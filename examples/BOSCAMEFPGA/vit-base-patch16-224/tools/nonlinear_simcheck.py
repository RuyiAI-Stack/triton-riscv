"""Strict emitted-IR nonlinear validation with disjoint program shards.

References only judge results. Every accepted output comes from Machine.launch;
the retained output archive lets collection repeat the numerical comparisons.
"""

import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from boscame_paths import buddy_tools, source_identity
from oracles import (
    compare,
    make_inputs,
    nonlinear_error_metrics,
    nonlinear_reference64,
    reference,
)

from export import ROOT, argument_shapes, digest


def _hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def _region(case, point):
    if (
        case["family"] not in ("gelu", "softmax")
        or len(point) != 3
        or any(
            not isinstance(v, int) or not 0 <= v < n
            for v, n in zip(point, case["grid"])
        )
    ):
        raise ValueError("invalid nonlinear program: " + str(point))
    stride = case["constexprs"][
        "BLOCK" if case["family"] == "gelu" else "LENGTH"
    ]
    start = point[0] * stride
    count = math.prod(case["shape"])
    return start, min(start + stride, count)


def nonlinear_program_mask(case, programs):
    mask = np.zeros(case["shape"], bool)
    if not programs:
        raise ValueError("empty nonlinear program selection")
    for point in programs:
        start, end = _region(case, point)
        if start >= end or np.any(mask.flat[start:end]):
            raise ValueError("empty/overlapping nonlinear program selection")
        mask.flat[start:end] = True
    return mask


def audit_nonlinear_program(case, point, accesses):
    start, end = (v * 4 for v in _region(case, point))
    intervals = {"%arg0": [], "%arg1": []}
    for name, lo, hi, action in accesses:
        if (
            name not in intervals
            or not start <= lo < hi <= end
            or lo % 4
            or hi % 4
            or action != ("read" if name == "%arg0" else "write")
        ):
            raise AssertionError(
                f"nonlinear program {point} accessed another tile: {(name, lo, hi, action)}"
            )
        intervals[name].append((lo, hi))
    for name, spans in intervals.items():
        spans.sort()
        if (
            not spans
            or spans[0][0] != start
            or spans[-1][1] != end
            or any(left[1] != right[0] for left, right in zip(spans, spans[1:]))
        ):
            raise AssertionError(
                "nonlinear missing/duplicate DDR coverage: " + name
            )
    return dict.fromkeys(intervals, end - start)


def selected_nonlinear_comparison(case, programs, actual, expected, initial):
    mask = nonlinear_program_mask(case, programs)
    np.testing.assert_array_equal(
        actual["Out"].view(np.uint32)[~mask],
        initial["Out"].view(np.uint32)[~mask],
        err_msg="nonlinear subset wrote unauthorized output",
    )
    mathematical = nonlinear_reference64(case, initial)["Out"][mask]
    metrics = nonlinear_error_metrics(actual["Out"][mask], mathematical)
    metrics["mathematical_reference_sha256"] = _hash(mathematical)
    metrics["ordered_f32_reference_sha256"] = _hash(expected["Out"][mask])
    return {"Out": actual["Out"][mask]}, {"Out": expected["Out"][mask]}, metrics


def nonlinear_shards(case, workers):
    if workers < 1:
        raise ValueError("nonlinear workers must be positive")
    points = list(itertools.product(*(range(n) for n in case["grid"])))
    width = math.ceil(len(points) / workers)
    return [
        points[start : start + width] for start in range(0, len(points), width)
    ]


def nonlinear_worker(case, output_root, programs):
    from simcheck import scenarios_for, simulate_case

    mask = nonlinear_program_mask(case, programs)
    machines, runs = {}, []
    for name, stress, order in scenarios_for(case):
        arrays = make_inputs(case, argument_shapes(case), stress=stress)
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
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
            raise AssertionError(
                "nonlinear shard did not retain one Machine/SPM"
            )
        metrics["input_sha256"] = _hash(arrays["X"])
        runs.append(
            {
                "name": name,
                "metrics": metrics,
                "values": actual["Out"][mask].copy(),
            }
        )
    return {"programs": programs, "runs": runs}


def assemble_nonlinear_round(case, pieces, expected, mathematical):
    actual = np.empty(case["shape"], np.float32)
    covered = np.zeros(case["shape"], bool)
    for piece in pieces:
        mask = nonlinear_program_mask(case, piece["programs"])
        metrics, values = piece["metrics"], piece["values"]
        if (
            np.any(covered & mask)
            or values.dtype != np.dtype("<f4")
            or values.shape != (int(np.count_nonzero(mask)),)
            or metrics["programs"] != len(piece["programs"])
            or metrics["outputs"] != values.size
            or metrics["output_sha256"] != {"Out": _hash(values)}
        ):
            raise AssertionError(
                "nonlinear aggregate lacks disjoint executed output evidence"
            )
        if {tuple(p) for p in metrics["selected_programs"]} != {
            tuple(p) for p in piece["programs"]
        } or len(metrics["selected_programs"]) != len(piece["programs"]):
            raise AssertionError(
                "nonlinear executed programs differ from selected programs"
            )
        actual[mask] = values
        covered |= mask
    if not np.all(covered):
        raise AssertionError("nonlinear aggregate is missing full-grid outputs")
    compare(case, {"Out": actual}, {"Out": expected})
    return actual, nonlinear_error_metrics(actual, mathematical)


def _sources():
    return {
        source_identity(path): digest(path)
        for path in (
            ROOT / "tools/simcheck.py",
            ROOT / "tools/nonlinear_simcheck.py",
            ROOT / "tools/oracles.py",
            buddy_tools() / "boscame_buffer_sim.py",
        )
    }


def check_nonlinear_sharded(case, output_root, workers):
    from simcheck import SimulationBlocked, check_artifacts, scenarios_for

    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    full_grid = math.prod(case["grid"])
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "reason": "",
        "programs": 0,
        "total_programs": full_grid * 3,
        "outputs": 0,
        "max_abs": None,
        "full_grid_programs": full_grid,
        "comparison_rounds": 3,
        "execution": "parallel-nonlinear-shards"
        if workers > 1
        else "serial-nonlinear-shard",
        "order_scope": "within each contiguous program shard; no global inter-shard order",
        "spm_lifetime": "one Machine per shard, poisoned once and retained across every program and all three rounds",
    }
    try:
        module = check_artifacts(case, output_root)
        record.update(
            module_sha256=digest(module),
            kernel_module_sha256=digest(case["kernel_source"]),
            simulator_sources=_sources(),
        )
        shards = nonlinear_shards(case, workers)
        record.update(
            workers=min(workers, len(shards)), shard_count=len(shards)
        )
        if workers == 1:
            results = [nonlinear_worker(case, output_root, shards[0])]
        else:
            results = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=record["workers"],
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(nonlinear_worker, case, output_root, points)
                    for points in shards
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    print(
                        f"  {case['name']}: {len(results)}/{len(shards)} shards, three rounds each",
                        flush=True,
                    )
        results.sort(key=lambda result: result["programs"][0])
        runs, archive = [], {}
        for index, (name, stress, order) in enumerate(scenarios_for(case)):
            arrays = make_inputs(case, argument_shapes(case), stress=stress)
            with np.errstate(invalid="ignore", over="ignore", under="ignore"):
                expected = reference(case, arrays)["Out"]
                mathematical = nonlinear_reference64(case, arrays)["Out"]
            pieces = [
                {"programs": result["programs"], **result["runs"][index]}
                for result in results
            ]
            actual, errors = assemble_nonlinear_round(
                case, pieces, expected, mathematical
            )
            archive["round" + str(index)] = actual
            for piece in pieces:
                metrics = piece["metrics"]
                if (
                    piece["name"] != name
                    or metrics["order"] != order
                    or metrics["module_sha256"] != digest(module)
                    or metrics["kernel_module_sha256"]
                    != digest(case["kernel_source"])
                    or metrics["input_sha256"] != _hash(arrays["X"])
                ):
                    raise AssertionError(
                        "nonlinear shard scenario/source/input differs"
                    )
            runs.append(
                {
                    "name": name,
                    "order": "shard-" + order,
                    "programs": full_grid,
                    "total_programs": full_grid,
                    "outputs": actual.size,
                    "module_sha256": digest(module),
                    "kernel_module_sha256": digest(case["kernel_source"]),
                    "input_sha256": _hash(arrays["X"]),
                    "output_sha256": {"Out": _hash(actual)},
                    "ordered_f32_reference_sha256": _hash(expected),
                    "mathematical_reference_sha256": _hash(mathematical),
                    "nonlinear_error": errors,
                    "nonlinear_ddr_bytes": {
                        key: sum(
                            p["metrics"]["nonlinear_ddr_bytes"][key]
                            for p in pieces
                        )
                        for key in ("%arg0", "%arg1")
                    },
                    "ddr_accesses": sum(
                        p["metrics"]["ddr_accesses"] for p in pieces
                    ),
                    "dma_transfers": sum(
                        p["metrics"]["dma_transfers"] for p in pieces
                    ),
                }
            )
        if runs[1]["output_sha256"] != runs[2]["output_sha256"]:
            raise AssertionError(
                "nonlinear stress output depends on launch order"
            )
        if record["simulator_sources"] != _sources():
            raise AssertionError("simulator source changed during execution")
        archive_path = output / "nonlinear-outputs.npz"
        np.savez_compressed(archive_path, **archive)
        record.update(
            status="PASS",
            programs=full_grid * 3,
            outputs=sum(r["outputs"] for r in runs),
            max_abs=max(r["nonlinear_error"]["max_abs"] for r in runs),
            runs=runs,
            output_archive=archive_path.name,
            output_archive_sha256=digest(archive_path),
            shards=[
                {
                    "programs": result["programs"],
                    "runs": [
                        {"name": r["name"], **r["metrics"]}
                        for r in result["runs"]
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
        f"{record['status']}: {case['name']} ({record['programs']}/{record['total_programs']}; nonlinear shards)",
        flush=True,
    )
    return record


def collect_nonlinear_case(case, output_root):
    from simcheck import SimulationBlocked, check_artifacts, scenarios_for

    output = Path(output_root) / case["name"]
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": math.prod(case["grid"]) * 3,
        "outputs": 0,
        "reason": "",
    }
    try:
        module = check_artifacts(case, output_root)
        saved = json.loads((output / "simcheck-result.json").read_text())
        if saved.get("status") != "PASS":
            return saved
        archive_path = output / "nonlinear-outputs.npz"
        if (
            saved.get("name") != case["name"]
            or saved.get("module_sha256") != digest(module)
            or saved.get("kernel_module_sha256")
            != digest(case["kernel_source"])
            or saved.get("simulator_sources") != _sources()
            or saved.get("output_archive") != archive_path.name
            or not archive_path.is_file()
            or saved.get("output_archive_sha256") != digest(archive_path)
        ):
            raise SimulationBlocked(
                "stale nonlinear source/module/output archive"
            )
        fixed = nonlinear_shards(case, saved["workers"])
        shards, runs = saved.get("shards", []), saved.get("runs", [])
        if (
            len(shards) != len(fixed)
            or saved.get("shard_count") != len(fixed)
            or len(runs) != 3
        ):
            raise SimulationBlocked("incomplete nonlinear shards/rounds")
        with np.load(archive_path, allow_pickle=False) as archive:
            for index, (name, stress, order) in enumerate(scenarios_for(case)):
                actual = archive["round" + str(index)]
                if actual.dtype != np.dtype("<f4") or actual.shape != tuple(
                    case["shape"]
                ):
                    raise SimulationBlocked(
                        "nonlinear output archive shape/dtype differs"
                    )
                arrays = make_inputs(case, argument_shapes(case), stress=stress)
                with np.errstate(
                    invalid="ignore", over="ignore", under="ignore"
                ):
                    expected = reference(case, arrays)["Out"]
                    mathematical = nonlinear_reference64(case, arrays)["Out"]
                compare(case, {"Out": actual}, {"Out": expected})
                errors = nonlinear_error_metrics(actual, mathematical)
                run = runs[index]
                if (
                    run.get("name") != name
                    or run.get("order") != "shard-" + order
                    or run.get("programs") != math.prod(case["grid"])
                    or run.get("total_programs") != math.prod(case["grid"])
                    or run.get("outputs") != actual.size
                    or run.get("module_sha256") != digest(module)
                    or run.get("kernel_module_sha256")
                    != digest(case["kernel_source"])
                    or run.get("input_sha256") != _hash(arrays["X"])
                    or run.get("output_sha256") != {"Out": _hash(actual)}
                    or run.get("ordered_f32_reference_sha256")
                    != _hash(expected)
                    or run.get("mathematical_reference_sha256")
                    != _hash(mathematical)
                    or run.get("nonlinear_error") != errors
                    or run.get("nonlinear_ddr_bytes")
                    != dict.fromkeys(("%arg0", "%arg1"), actual.size * 4)
                ):
                    raise SimulationBlocked(
                        "nonlinear round numerical/coverage evidence differs"
                    )
                for shard, points in zip(shards, fixed):
                    if (
                        shard.get("programs") != [list(p) for p in points]
                        or len(shard.get("runs", [])) != 3
                    ):
                        raise SimulationBlocked(
                            "nonlinear fixed shard coverage differs"
                        )
                    item = shard["runs"][index]
                    ordered = points[::-1] if order == "reverse" else points
                    mask = nonlinear_program_mask(case, points)
                    shard_errors = nonlinear_error_metrics(
                        actual[mask], mathematical[mask]
                    )
                    shard_errors.update(
                        mathematical_reference_sha256=_hash(mathematical[mask]),
                        ordered_f32_reference_sha256=_hash(expected[mask]),
                    )
                    if (
                        item.get("name") != name
                        or item.get("order") != order
                        or item.get("selected_programs")
                        != [list(p) for p in ordered]
                        or item.get("programs") != len(points)
                        or item.get("outputs") != int(np.count_nonzero(mask))
                        or item.get("output_sha256")
                        != {"Out": _hash(actual[mask])}
                        or item.get("module_sha256") != digest(module)
                        or item.get("kernel_module_sha256")
                        != digest(case["kernel_source"])
                        or item.get("input_sha256") != _hash(arrays["X"])
                        or item.get("nonlinear_error") != shard_errors
                        or item.get("nonlinear_ddr_bytes")
                        != dict.fromkeys(
                            ("%arg0", "%arg1"), int(np.count_nonzero(mask)) * 4
                        )
                    ):
                        raise SimulationBlocked(
                            "nonlinear shard executed output/order evidence differs"
                        )
        if (
            saved.get("programs") != math.prod(case["grid"]) * 3
            or saved.get("total_programs") != math.prod(case["grid"]) * 3
            or saved.get("outputs") != math.prod(case["shape"]) * 3
            or saved.get("max_abs")
            != max(run["nonlinear_error"]["max_abs"] for run in runs)
            or runs[1]["output_sha256"] != runs[2]["output_sha256"]
        ):
            raise SimulationBlocked("incomplete nonlinear full-grid totals")
        return saved
    except (
        SimulationBlocked,
        AssertionError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        record["reason"] = str(error)
    return record

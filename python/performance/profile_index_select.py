"""Capture host intervals for the CPU index_select benchmark in Perfetto format."""

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory


def _link_launches(launches, artifacts):
    # Take the IR from the object returned by that launch, including cache hits.
    # Do the hashing after capture so it is outside all timed intervals.
    by_kernel = {}
    for event, kernel in launches:
        if kernel not in by_kernel:
            if not getattr(kernel, "hash", None) or not getattr(kernel, "name", None):
                raise RuntimeError("CPU launch did not return a compiled kernel")
            fields = {"kernel_hash": kernel.hash, "kernel_name": kernel.name}
            for stage in ("ttir", "ttsharedir"):
                ir = kernel.asm.get(stage)
                if not isinstance(ir, str) or not ir.strip():
                    raise RuntimeError(f"{kernel.name}: missing compiled IR: {stage}")
                data = ir.encode("utf-8")
                digest = hashlib.sha256(data).hexdigest()
                path = f"kernels/{digest}.{stage}.mlir"
                artifacts[path] = data
                fields[f"{stage}_file"] = path
            by_kernel[kernel] = fields
        event["args"].update(by_kernel[kernel])


def capture(*, artifacts=None):
    import torch
    from test_performance_index_select import (
        INDEX_SELECT_CASES,
        index_select_row_kernel,
        make_index_select_inputs,
        run_index_select,
    )

    pid, tid = os.getpid(), threading.get_native_id()
    machine = platform.machine()
    events = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": pid,
            "tid": 0,
            "args": {"name": f"index_select ({machine})"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": pid,
            "tid": tid,
            "args": {"name": "CPU benchmark"},
        },
    ]
    origin = time.perf_counter_ns()
    launches = []

    def record(context, scope, name, func, *args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        events.append(
            {
                "ph": "X",
                "cat": "triton-riscv.host",
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": (start - origin) / 1000,
                "dur": (end - start) / 1000,
                "args": {**context, "scope": scope},
            }
        )
        if artifacts is not None and scope == "kernel_call":
            launches.append((events[-1], result))
        return result

    for rows, cols, picks in INDEX_SELECT_CASES:
        x, indices = make_index_select_inputs(rows, cols, picks)
        case = f"bench_index_select(rows={rows}, cols={cols}, picks={picks})"
        for phase, count in (("correctness", 1), ("warmup", 5), ("measure", 20)):
            for iteration in range(1, count + 1):
                context = {"case": case, "phase": phase, "iteration": iteration}
                expected = record(
                    {**context, "provider": "torch"},
                    "provider_call",
                    "torch",
                    x.index_select,
                    0,
                    indices,
                )
                triton_call = partial(record, {**context, "provider": "triton-riscv"})
                actual = triton_call(
                    "provider_call",
                    "triton-riscv",
                    run_index_select,
                    x,
                    indices,
                    allocate=partial(
                        triton_call, "output_allocation", "Allocate output", torch.empty
                    ),
                    launch=partial(
                        triton_call,
                        "kernel_call",
                        "Call index_select_row_kernel (host)",
                        index_select_row_kernel[(1,)],
                    ),
                )
                if phase == "correctness":
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if artifacts is not None:
        _link_launches(launches, artifacts)
    return {
        "traceEvents": events,
        "metadata": {
            "machine": machine,
            "clock": "host perf_counter_ns",
            "torch_threads": torch.get_num_threads(),
            "triton_openmp_threads": os.environ.get("TRITON_RISCV_OPENMP_THREADS"),
        },
    }


def _write_trace(path, trace):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, separators=(",", ":"))
        stream.write("\n")


def _write_bundle(path):
    artifacts = {}
    trace = capture(artifacts=artifacts)
    # Stage everything first. Claim the destination exclusively, even if another
    # capture created it while the benchmark was running. Publish the trace last.
    with TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp)
        (stage / "kernels").mkdir()
        for name, data in artifacts.items():
            (stage / name).write_bytes(data)
        _write_trace(stage / "trace.json", trace)
        path.mkdir()
        (stage / "kernels").rename(path / "kernels")
        (stage / "trace.json").rename(path / "trace.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dest = parser.add_mutually_exclusive_group(required=True)
    dest.add_argument("--output", type=Path, help="New trace JSON file")
    dest.add_argument(
        "--bundle", type=Path, help="New directory for trace and kernel IR"
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    path = args.bundle if args.bundle is not None else args.output
    if path.exists():
        parser.error(f"output already exists: {path}")
    if not path.parent.is_dir():
        parser.error(f"output directory does not exist: {path.parent}")
    # Match run_all.py and set thread counts before importing either runtime.
    for name in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.torch_threads)
    os.environ.setdefault("TRITON_RISCV_OPENMP_THREADS", str(args.torch_threads))
    if args.bundle is None:
        _write_trace(path, capture())
    else:
        _write_bundle(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

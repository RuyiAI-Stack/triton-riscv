"""Capture host intervals for the CPU index_select benchmark in Perfetto format."""

import argparse
import json
import os
import platform
import threading
import time
from functools import partial
from pathlib import Path


def capture():
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
    return {
        "traceEvents": events,
        "metadata": {
            "machine": machine,
            "clock": "host perf_counter_ns",
            "torch_threads": torch.get_num_threads(),
            "triton_openmp_threads": os.environ.get("TRITON_RISCV_OPENMP_THREADS"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New trace JSON file"
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output directory does not exist: {args.output.parent}")
    # Match run_all.py and set thread counts before importing either runtime.
    for name in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.torch_threads)
    os.environ.setdefault("TRITON_RISCV_OPENMP_THREADS", str(args.torch_threads))
    trace = capture()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, separators=(",", ":"))
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

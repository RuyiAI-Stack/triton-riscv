import csv
import os
import sys
import time
import numpy as np
from functools import wraps
from pathlib import Path
import triton
from triton.backends.triton_shared.driver import CPUDriver

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_PRINTED_TABLE_HEADER = False


def configure_torch_threads_from_env():
    threads = os.environ.get("TORCH_NUM_THREADS")
    if not threads:
        return
    try:
        import torch

        count = int(threads)
        torch.set_num_threads(count)
        try:
            torch.set_num_interop_threads(count)
        except RuntimeError:
            pass
    except Exception:
        return


configure_torch_threads_from_env()


def select_cpu_backend():
    triton.runtime.driver.set_active(CPUDriver())


# Unfortunately, we can't use triton.testing.perf_report and triton.testing.do_bench for CPU backend because
# they are very specific to cuda


def measure(
    repeats=20,
    warmup=5,
    percentiles=(),
    timers={"Wall": time.perf_counter, "CPU": time.process_time},
):
    """
    Decorator to benchmark a function.

    Parameters:
    - repeats (int): The number of times the function should be executed for each set of parameters.
    - percentiles (tuple): The percentiles to compute on the execution times (e.g., (50, 90, 99)).
    - timers (dict): A dictionary where keys are timer names (e.g., 'Wall', 'CPU') and values are timer functions
                     that measure elapsed time. By default:
                     * 'Wall': Uses time.perf_counter for high-resolution wall-clock time.
                     * 'CPU': Uses time.process_time for CPU time spent by the process.

    Returns:
    - A decorated function that prints:
        * Average execution time.
        * Standard deviation time.
        * Minimum and maximum times.
        * Computed percentiles for each timer.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(
                f"{func.__name__}{args} {kwargs}, warmup={warmup}, repeats={repeats}, all results in seconds"
            )
            times = {}
            for t, _ in timers.items():
                times[t] = []

            for _ in range(warmup):
                result = func(*args, **kwargs)

            for _ in range(repeats):
                starts = {}
                for t, f in timers.items():
                    starts[t] = f()

                result = func(*args, **kwargs)

                for t, f in timers.items():
                    times[t].append(f() - starts[t])

            for t, _ in timers.items():
                average_time = np.mean(times[t])
                min_time = np.min(times[t])
                max_time = np.max(times[t])
                computed_percentiles = np.percentile(times[t], percentiles)
                std_dev_time = np.std(times[t])

                print(
                    f"{t}: Avg={average_time:.6f}, min={min_time:.6f}, std={std_dev_time:.6f},",
                    end=" ",
                )
                for p, value in zip(percentiles, computed_percentiles):
                    print(f"{p}pp={value:.6f},", end=" ")
                print(f"max={max_time:.6f}")

            return result

        return wrapper

    return decorator


def _assert_close(actual, expected, *, rtol=1e-4, atol=1e-4):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for benchmark correctness checks"
        ) from exc

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return

    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(
                f"output length mismatch: actual={len(actual)}, expected={len(expected)}"
            )
        for actual_item, expected_item in zip(actual, expected):
            _assert_close(actual_item, expected_item, rtol=rtol, atol=atol)
        return

    if actual != expected:
        raise AssertionError(f"output mismatch: actual={actual}, expected={expected}")


def _assert_cpu_tree(value, provider_name):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for benchmark device checks") from exc

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise AssertionError(
                f"{provider_name} returned a non-CPU tensor: device={value.device}"
            )
        return

    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_cpu_tree(item, provider_name)
        return

    if isinstance(value, dict):
        for item in value.values():
            _assert_cpu_tree(item, provider_name)


def color(text, code):
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def format_seconds(seconds):
    if seconds is None:
        return "-"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def format_speedup(speedup):
    if speedup is None:
        return "-"
    return f"{speedup:.3f}x"


def fit(text, width):
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def status_label(status):
    padded = f"{status:<8}"
    if status == "PASS":
        return color(padded, "32;1")
    return color(padded, "31;1")


def print_table_header():
    global _PRINTED_TABLE_HEADER
    if _PRINTED_TABLE_HEADER or os.environ.get("TRITON_RISCV_BENCH_HEADER") == "0":
        return
    print(
        f"{'case':<56} {'status':<8} {'torch':>13} "
        f"{'triton-riscv':>13} {'speedup':>9} {'process cpu':>13}"
    )
    print(color("-" * 117, "2"))
    _PRINTED_TABLE_HEADER = True


def print_result_row(
    case_name,
    status,
    baseline_wall_avg,
    provider_wall_avg,
    speedup,
    provider_cpu_avg,
):
    print_table_header()
    print(
        f"{fit(case_name, 56):<56} {status_label(status)} "
        f"{format_seconds(baseline_wall_avg):>13} "
        f"{format_seconds(provider_wall_avg):>13} "
        f"{format_speedup(speedup):>9} "
        f"{format_seconds(provider_cpu_avg):>13}"
    )


def append_csv_row(row):
    csv_path = os.environ.get("TRITON_RISCV_BENCH_CSV")
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "status",
        "baseline",
        "provider",
        "baseline_wall_avg_s",
        "provider_wall_avg_s",
        "speedup",
        "baseline_cpu_avg_s",
        "provider_cpu_avg_s",
        "warmup",
        "repeats",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def measure_callable(
    name,
    func,
    *,
    repeats=20,
    warmup=5,
    percentiles=(),
    timers={"Wall": time.perf_counter, "CPU": time.process_time},
):
    times = {timer_name: [] for timer_name in timers}

    for _ in range(warmup):
        func()

    result = None
    for _ in range(repeats):
        starts = {timer_name: timer() for timer_name, timer in timers.items()}
        result = func()
        _assert_cpu_tree(result, name)
        for timer_name, timer in timers.items():
            times[timer_name].append(timer() - starts[timer_name])

    stats = {}
    for timer_name in timers:
        average_time = np.mean(times[timer_name])
        min_time = np.min(times[timer_name])
        max_time = np.max(times[timer_name])
        computed_percentiles = np.percentile(times[timer_name], percentiles)
        std_dev_time = np.std(times[timer_name])
        stats[timer_name] = {
            "avg": float(average_time),
            "min": float(min_time),
            "max": float(max_time),
            "std": float(std_dev_time),
            "percentiles": {
                str(p): float(value)
                for p, value in zip(percentiles, computed_percentiles)
            },
        }

    return {"name": name, "result": result, "stats": stats}


def _provider_name(providers, baseline):
    for name in providers:
        if name != baseline:
            return name
    return baseline


def _avg(stats, timer_name):
    timer_stats = stats.get(timer_name)
    if not timer_stats:
        return None
    return timer_stats["avg"]


def _bench_summary(
    case_name, status, baseline, provider_name, measured, warmup, repeats
):
    baseline_stats = measured[baseline]["stats"]
    provider_stats = measured[provider_name]["stats"]
    baseline_wall = _avg(baseline_stats, "Wall")
    provider_wall = _avg(provider_stats, "Wall")
    baseline_cpu = _avg(baseline_stats, "CPU")
    provider_cpu = _avg(provider_stats, "CPU")
    speedup = None
    if baseline_wall is not None and provider_wall and provider_wall > 0:
        speedup = baseline_wall / provider_wall

    print_result_row(
        case_name,
        status,
        baseline_wall,
        provider_wall,
        speedup,
        provider_cpu,
    )
    append_csv_row(
        {
            "case": case_name,
            "status": status,
            "baseline": baseline,
            "provider": provider_name,
            "baseline_wall_avg_s": (
                f"{baseline_wall:.9f}" if baseline_wall is not None else ""
            ),
            "provider_wall_avg_s": (
                f"{provider_wall:.9f}" if provider_wall is not None else ""
            ),
            "speedup": f"{speedup:.6f}" if speedup is not None else "",
            "baseline_cpu_avg_s": (
                f"{baseline_cpu:.9f}" if baseline_cpu is not None else ""
            ),
            "provider_cpu_avg_s": (
                f"{provider_cpu:.9f}" if provider_cpu is not None else ""
            ),
            "warmup": warmup,
            "repeats": repeats,
        }
    )
    return {
        "case": case_name,
        "status": status,
        "baseline": baseline,
        "provider": provider_name,
        "baseline_wall_avg_s": baseline_wall,
        "provider_wall_avg_s": provider_wall,
        "speedup": speedup,
        "baseline_cpu_avg_s": baseline_cpu,
        "provider_cpu_avg_s": provider_cpu,
    }


def compare_providers(
    case_name,
    providers,
    *,
    baseline="torch",
    check=True,
    rtol=1e-4,
    atol=1e-4,
    repeats=20,
    warmup=5,
    percentiles=(),
    timers={"Wall": time.perf_counter, "CPU": time.process_time},
):
    """
    Run correctness and performance comparison for named provider callables.

    `providers` is an ordered mapping of provider name to zero-argument callable.
    The baseline provider is used as the correctness reference.
    """
    provider_name = _provider_name(providers, baseline)

    if check:
        if baseline not in providers:
            raise ValueError(f"baseline provider '{baseline}' is not in providers")
        expected = providers[baseline]()
        _assert_cpu_tree(expected, baseline)
        for provider_name, provider in providers.items():
            if provider_name == baseline:
                continue
            actual = provider()
            _assert_cpu_tree(actual, provider_name)
            _assert_close(actual, expected, rtol=rtol, atol=atol)

    measured = {}
    for provider_name, provider in providers.items():
        measured[provider_name] = measure_callable(
            provider_name,
            provider,
            repeats=repeats,
            warmup=warmup,
            percentiles=percentiles,
            timers=timers,
        )

    return _bench_summary(
        case_name,
        "PASS",
        baseline,
        _provider_name(providers, baseline),
        measured,
        warmup,
        repeats,
    )

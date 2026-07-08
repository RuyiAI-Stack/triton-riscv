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
BUDDY_PROVIDER = "torch+buddy-mlir"
TRITON_PROVIDER = "triton-riscv"


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


def format_provider_time(seconds, status="PASS"):
    if status != "PASS":
        return "-"
    return format_seconds(seconds)


def fit(text, width):
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def status_label(status):
    padded = f"{status:^6}"
    if status == "PASS":
        return color(padded, "32;1")
    return color(padded, "31;1")


def print_table_header():
    global _PRINTED_TABLE_HEADER
    if _PRINTED_TABLE_HEADER or os.environ.get("TRITON_RISCV_BENCH_HEADER") == "0":
        return
    print(
        f"{'case':<56} {'torch(baseline)':>15} "
        f"{'torch+buddy':>13}  {'status':^6}  {'speedup':>7} "
        f"{TRITON_PROVIDER:>13}  {'status':^6}  {'speedup':>7}"
    )
    print(color("-" * 137, "2"))
    _PRINTED_TABLE_HEADER = True


def print_result_row(
    case_name,
    status,
    baseline_wall_avg,
    buddy_wall_avg,
    buddy_status,
    buddy_speedup,
    triton_wall_avg,
    triton_status,
    triton_speedup,
):
    print_table_header()
    print(
        f"{fit(case_name, 56):<56} "
        f"{format_seconds(baseline_wall_avg):>15} "
        f"{format_provider_time(buddy_wall_avg, buddy_status):>13}  "
        f"{status_label(buddy_status)}  "
        f"{format_speedup(buddy_speedup):>7} "
        f"{format_seconds(triton_wall_avg):>13} "
        f" {status_label(triton_status)}  "
        f"{format_speedup(triton_speedup):>7}"
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
        "buddy_status",
        "buddy_wall_avg_s",
        "buddy_cpu_avg_s",
        "buddy_speedup",
        "buddy_error",
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


def _truthy_env(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _candidate_buddy_python_paths():
    paths = []
    explicit = os.environ.get("BUDDY_MLIR_PYTHON_PACKAGES_DIR")
    if explicit:
        paths.append(Path(explicit))

    buddy_dir = os.environ.get("BUDDY_DIR")
    if buddy_dir:
        paths.append(Path(buddy_dir) / "build" / "python_packages")

    repo_root = Path(__file__).resolve().parents[2]
    paths.append(repo_root.parent / "buddy-mlir" / "build" / "python_packages")
    paths.append(repo_root / ".cache" / "buddy" / "python_packages")
    return paths


def _load_buddy_torch_backend():
    for path in reversed(_candidate_buddy_python_paths()):
        if path.exists():
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)

    from buddy.compiler.frontend import dynamo_compiler

    return dynamo_compiler


def _normalize_buddy_result(result):
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


def _make_buddy_provider(baseline_provider, backend):
    compiled = None

    def run_buddy():
        nonlocal compiled
        import torch

        if compiled is None:
            compiled = torch.compile(
                baseline_provider,
                backend=backend,
                fullgraph=True,
                dynamic=False,
            )
        return _normalize_buddy_result(compiled())

    return run_buddy


def _with_optional_buddy_provider(providers, baseline):
    if not _truthy_env("TRITON_RISCV_BENCH_BUDDY", default=True):
        return dict(providers)
    if baseline not in providers or BUDDY_PROVIDER in providers:
        return dict(providers)
    try:
        backend = _load_buddy_torch_backend()
    except Exception:
        return dict(providers)

    with_buddy = {}
    inserted = False
    for name, provider in providers.items():
        with_buddy[name] = provider
        if name == baseline:
            with_buddy[BUDDY_PROVIDER] = _make_buddy_provider(provider, backend)
            inserted = True
    if not inserted:
        with_buddy[BUDDY_PROVIDER] = _make_buddy_provider(providers[baseline], backend)
    return with_buddy


def _avg(stats, timer_name):
    timer_stats = stats.get(timer_name)
    if not timer_stats:
        return None
    return timer_stats["avg"]


def _bench_summary(
    case_name, status, baseline, measured, provider_errors, warmup, repeats
):
    baseline_stats = measured[baseline]["stats"]
    baseline_wall = _avg(baseline_stats, "Wall")
    baseline_cpu = _avg(baseline_stats, "CPU")
    triton = measured.get(TRITON_PROVIDER)
    triton_stats = triton["stats"] if triton else {}
    triton_wall = _avg(triton_stats, "Wall")
    triton_cpu = _avg(triton_stats, "CPU")
    triton_status = "PASS" if triton else "FAIL"
    triton_speedup = None
    if baseline_wall is not None and triton_wall and triton_wall > 0:
        triton_speedup = baseline_wall / triton_wall

    buddy = measured.get(BUDDY_PROVIDER)
    buddy_stats = buddy["stats"] if buddy else {}
    buddy_wall = _avg(buddy_stats, "Wall")
    buddy_cpu = _avg(buddy_stats, "CPU")
    buddy_error = provider_errors.get(BUDDY_PROVIDER, "")
    if buddy:
        buddy_status = "PASS"
    elif buddy_error:
        buddy_status = "FAIL"
    else:
        buddy_status = "SKIP"
    buddy_speedup = None
    if baseline_wall is not None and buddy_wall and buddy_wall > 0:
        buddy_speedup = baseline_wall / buddy_wall

    print_result_row(
        case_name,
        status,
        baseline_wall,
        buddy_wall,
        buddy_status,
        buddy_speedup,
        triton_wall,
        triton_status,
        triton_speedup,
    )
    append_csv_row(
        {
            "case": case_name,
            "status": status,
            "baseline": baseline,
            "provider": TRITON_PROVIDER,
            "baseline_wall_avg_s": (
                f"{baseline_wall:.9f}" if baseline_wall is not None else ""
            ),
            "provider_wall_avg_s": (
                f"{triton_wall:.9f}" if triton_wall is not None else ""
            ),
            "speedup": f"{triton_speedup:.6f}" if triton_speedup is not None else "",
            "baseline_cpu_avg_s": (
                f"{baseline_cpu:.9f}" if baseline_cpu is not None else ""
            ),
            "provider_cpu_avg_s": (
                f"{triton_cpu:.9f}" if triton_cpu is not None else ""
            ),
            "buddy_status": buddy_status,
            "buddy_wall_avg_s": f"{buddy_wall:.9f}" if buddy_wall is not None else "",
            "buddy_cpu_avg_s": f"{buddy_cpu:.9f}" if buddy_cpu is not None else "",
            "buddy_speedup": (
                f"{buddy_speedup:.6f}" if buddy_speedup is not None else ""
            ),
            "buddy_error": buddy_error[:500],
            "warmup": warmup,
            "repeats": repeats,
        }
    )
    return {
        "case": case_name,
        "status": status,
        "baseline": baseline,
        "provider": TRITON_PROVIDER,
        "baseline_wall_avg_s": baseline_wall,
        "provider_wall_avg_s": triton_wall,
        "speedup": triton_speedup,
        "baseline_cpu_avg_s": baseline_cpu,
        "provider_cpu_avg_s": triton_cpu,
        "buddy_status": buddy_status,
        "buddy_wall_avg_s": buddy_wall,
        "buddy_cpu_avg_s": buddy_cpu,
        "buddy_speedup": buddy_speedup,
        "buddy_error": buddy_error,
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
    providers = _with_optional_buddy_provider(providers, baseline)
    provider_errors = {}

    if check:
        if baseline not in providers:
            raise ValueError(f"baseline provider '{baseline}' is not in providers")
        expected = providers[baseline]()
        _assert_cpu_tree(expected, baseline)
        for provider_name, provider in providers.items():
            if provider_name == baseline:
                continue
            try:
                actual = provider()
            except Exception as exc:
                if provider_name == BUDDY_PROVIDER:
                    provider_errors[provider_name] = f"{type(exc).__name__}: {exc}"
                    continue
                raise
            _assert_cpu_tree(actual, provider_name)
            try:
                _assert_close(actual, expected, rtol=rtol, atol=atol)
            except Exception as exc:
                if provider_name == BUDDY_PROVIDER:
                    provider_errors[provider_name] = f"{type(exc).__name__}: {exc}"
                    continue
                raise

    measured = {}
    for provider_name, provider in providers.items():
        if provider_name in provider_errors:
            continue
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
        measured,
        provider_errors,
        warmup,
        repeats,
    )

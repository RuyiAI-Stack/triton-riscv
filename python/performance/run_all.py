import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


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


def print_table_header(total, csv_path):
    import torch

    title = f"Triton-RISCV benchmark ({total} script{'s' if total != 1 else ''})"
    print(color(title, "36;1"))
    print("  baseline     : torch")
    print("  provider     : triton-riscv")
    print("  extra        : torch+buddy-mlir")
    print("  tensor device : cpu")
    print(
        f"  torch cuda   : {'available' if torch.cuda.is_available() else 'not available'}"
    )
    print(
        f"  torch threads: intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}"
    )
    print("  output csv   : " + str(csv_path))
    print()
    print(
        f"{'case':<56} {'torch(baseline)':>15} "
        f"{'torch+buddy':>13}  {'status':^6}  {'speedup':>7} "
        f"{'triton-riscv':>13}  {'status':^6}  {'speedup':>7}"
    )
    print(color("-" * 137, "2"))
    sys.stdout.flush()


def read_csv_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def print_summary(csv_path, script_count, failures):
    rows = read_csv_rows(csv_path)
    passed = sum(1 for row in rows if row.get("status") == "PASS")
    total = len(rows)
    total_triton = 0.0
    triton_count = 0
    total_buddy = 0.0
    buddy_count = 0
    buddy_passed = 0
    for row in rows:
        value = row.get("provider_wall_avg_s", "")
        if value:
            total_triton += float(value)
            triton_count += 1
        buddy_value = row.get("buddy_wall_avg_s", "")
        if buddy_value:
            total_buddy += float(buddy_value)
            buddy_count += 1
        if row.get("buddy_status") == "PASS":
            buddy_passed += 1
    avg_triton = total_triton / triton_count if triton_count else None
    avg_buddy = total_buddy / buddy_count if buddy_count else None

    print(color("-" * 137, "2"))
    status = color(
        f"{passed}/{total} benchmark rows passed",
        "32;1" if not failures else "31;1",
    )
    print(
        f"summary: {status}; {script_count - len(failures)}/{script_count} scripts succeeded; "
        f"avg torch+buddy-mlir {format_seconds(avg_buddy)} "
        f"({buddy_passed}/{total} rows); avg triton-riscv {format_seconds(avg_triton)}"
    )
    print(f"csv: {color(str(csv_path), '36')}")
    if failures:
        print()
        print(color("failures:", "31;1"))
        for name, returncode in failures:
            print(f"  {name}: exit code {returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run all performance benchmarks.")
    parser.add_argument(
        "cases",
        nargs="*",
        help="Optional case filename stems, for example: matmul vec_add softmax.",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/performance/triton_riscv_bench.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Set PyTorch intra-op CPU thread count for benchmark subprocesses.",
    )
    args = parser.parse_args()

    import torch
    benchmark_threads = (
        args.torch_threads
        if args.torch_threads is not None
        else torch.get_num_threads()
    )
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(args.torch_threads)
        except RuntimeError:
            pass

    perf_dir = Path(__file__).resolve().parent
    repo_root = perf_dir.parents[1]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("")

    env = os.environ.copy()
    buddy_dir = Path(env.get("BUDDY_DIR", repo_root.parent / "buddy-mlir"))
    buddy_bin = buddy_dir / "build" / "bin"
    llvm_bin = buddy_dir / "llvm" / "build" / "bin"
    if buddy_bin.is_dir():
        env.setdefault("BUDDY_MLIR_BINARY_DIR", str(buddy_bin))
    if llvm_bin.is_dir():
        env.setdefault("LLVM_BINARY_DIR", str(llvm_bin))
    env["PYTHONPATH"] = (
        str(perf_dir)
        if not env.get("PYTHONPATH")
        else f"{perf_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["TRITON_RISCV_BENCH_CSV"] = str(csv_path)
    env["TRITON_RISCV_BENCH_HEADER"] = "0"
    env["OMP_NUM_THREADS"] = str(benchmark_threads)
    env["MKL_NUM_THREADS"] = str(benchmark_threads)
    env["TORCH_NUM_THREADS"] = str(benchmark_threads)
    env.setdefault("TRITON_RISCV_OPENMP_THREADS", str(benchmark_threads))

    scripts = sorted(perf_dir.glob("test_performance_*.py"))
    if args.cases:
        requested = {
            case if case.startswith("test_performance_") else f"test_performance_{case}"
            for case in args.cases
        }
        requested = {
            case if case.endswith(".py") else f"{case}.py" for case in requested
        }
        scripts = [script for script in scripts if script.name in requested]
        missing = sorted(requested - {script.name for script in scripts})
        if missing:
            print(f"Missing benchmark case(s): {', '.join(missing)}", file=sys.stderr)
            return 2

    failures = []
    print_table_header(len(scripts), csv_path)
    for script in scripts:
        result = subprocess.run([sys.executable, str(script)], env=env)
        if result.returncode != 0:
            failures.append((script.name, result.returncode))

    print_summary(csv_path, len(scripts), failures)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

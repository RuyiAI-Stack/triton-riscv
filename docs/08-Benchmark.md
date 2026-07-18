# Benchmarking Triton-RISCV

> **Advanced developer material:** this page is not part of the beginner route.
>
> **Best for:** kernel developers and compiler contributors measuring a case
> whose correctness has already been established.
>
> **Prerequisites:** the [Host CPU Quick Start](00-Getting-Started.md#15-minute-quick-start) and the
> focused correctness test for the operator being measured.
>
> **After this guide:** you can run reproducible host comparisons, save CSV
> results, and state clearly what the numbers do—and do not—measure.

The benchmarks under `python/performance/` compare correctness and host CPU
execution time across PyTorch, Triton-RISCV, and optionally Buddy's
`torch.compile` backend. They are useful for detecting regressions and
understanding compiler/runtime overhead on the current host.

PyTorch is a correctness and timing reference in these tables. It is not a
fallback executed by Triton-RISCV, and a PyTorch result does not establish that
the same operation is supported by the Triton-RISCV compiler.

They are **not** RISC-V hardware benchmarks. QEMU timing and `x86_64` host timing
must not be presented as RVV hardware performance.

## 1. Environment Setup

Activate the same Python environment used to build and run Triton-RISCV, then
source the repository environment helper:

```sh
conda activate ruyiai   # or: source /path/to/triton-riscv/.venv/bin/activate
cd /path/to/triton-riscv
source scripts/triton-riscv-env.sh
```

That command assumes the sibling source-build layout. For the fast prebuilt
layout from [Getting started](00-Getting-Started.md), activate `.venv` and
export its in-repository tool paths instead:

```sh
source .venv/bin/activate
export TRITON_PLUGIN_DIRS="$PWD"
export LLVM_SYSPATH="$PWD/.cache/llvm"
export LLVM_BINARY_DIR="$LLVM_SYSPATH/bin"
export BUDDY_MLIR_BINARY_DIR="$PWD/.cache/buddy/bin"
```

The helper sets paths for the Triton plugin, Buddy/LLVM tools, and Triton's
runtime cache directories. If you changed backend Python code, lowering passes,
or C++ sources, rebuild and clear stale kernel cache entries before measuring:

```sh
scripts/rebuild-triton-riscv.sh
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/artifacts/cache/benchmark}"
rm -rf "$TRITON_CACHE_DIR"
```

The Buddy column needs Buddy's Python packages. The benchmark helper first uses
`BUDDY_MLIR_PYTHON_PACKAGES_DIR` when set, then tries
`$BUDDY_DIR/build/python_packages`, and finally the common sibling checkout
path `../buddy-mlir/build/python_packages`.

Before timing, run the corresponding correctness test under `python/examples/`
with a fresh cache. Benchmark scripts check outputs too, but a focused pytest
failure is usually easier to diagnose.

## 2. Run All Benchmarks

Use the aggregate runner to execute every `test_performance_*.py` file:

```sh
python python/performance/run_all.py
```

For a quick installation and output-format check, start with one small family:

```sh
python python/performance/run_all.py vec_add --torch-threads 1
```

The runner discovers all benchmark scripts, adds `python/performance/` to
`PYTHONPATH`, and runs each case in a separate Python process. If any benchmark
fails, it prints a failure summary and exits with a nonzero status. It also
writes a CSV file by default:

```text
artifacts/performance/triton_riscv_bench.csv
```

To run a subset, pass filename stems:

```sh
python python/performance/run_all.py matmul vec_add softmax
```

To choose a CSV path:

```sh
python python/performance/run_all.py matmul --csv artifacts/performance/matmul.csv
```

By default PyTorch uses its normal CPU thread settings, which may use many CPU
threads through ATen, OpenMP, MKL, or oneDNN depending on the local build. To run
a single-threaded PyTorch CPU baseline:

```sh
python python/performance/run_all.py matmul --torch-threads 1
```

These names map to:

```text
matmul  -> python/performance/test_performance_matmul.py
vec_add -> python/performance/test_performance_vec_add.py
softmax -> python/performance/test_performance_softmax.py
```

## 3. Run a Single Benchmark Script

You can also run a benchmark file directly. Because `benchmark.py` is a local
module in `python/performance/`, set `PYTHONPATH` when running a script by path:

```sh
PYTHONPATH="$PWD/python/performance:$PYTHONPATH" \
python python/performance/test_performance_matmul.py
```

Do not install the unrelated PyPI package named `benchmark`. The benchmarks in
this repository import the local file:

```text
python/performance/benchmark.py
```

`scripts/triton-riscv-env.sh` sets `PYTHONSAFEPATH=1`, so relying on Python's
implicit script-directory import behavior is not portable. Use `run_all.py` or
set `PYTHONPATH` explicitly.

## 4. Output Format

Each benchmark case first checks Triton-RISCV output against a PyTorch reference
and then measures both providers.

| Column | Meaning |
| --- | --- |
| `torch(baseline)` | average PyTorch wall time |
| `torch+buddy` | average wall time for the PyTorch reference through Buddy MLIR and `torch.compile` |
| Buddy `status` | `PASS`, `SKIP`, or `FAIL` for correctness/availability |
| Buddy `speedup` | `torch / torch+buddy` |
| `triton-riscv` | average Triton-RISCV wall time |
| Triton-RISCV `status` | correctness/compilation status for the backend |
| Triton-RISCV `speedup` | `torch / triton-riscv` |

A speedup above `1.0x` means that provider was faster than the PyTorch baseline
for that case; below `1.0x` means it was slower. Compare status before speedup.
A fast result with failed correctness is not a valid measurement.

The CSV stores raw second values for the table columns, plus `buddy_status`,
`buddy_speedup`, `buddy_error`, `warmup`, and `repeats`. It also retains
process CPU timing columns for detailed analysis. The default timing policy is
still 5 warmup executions and 20 measured repeats per provider. Disable the
Buddy column with:

```sh
TRITON_RISCV_BENCH_BUDDY=0 python python/performance/run_all.py matmul
```

The benchmark helper also checks that all tensor outputs are CPU tensors. If a
PyTorch reference or Triton-RISCV provider returns a CUDA tensor, the benchmark
fails instead of reporting a timing row.

## 5. What Is Being Measured

The `python/performance/` benchmarks run the Triton-RISCV CPU backend in the
current Python process and compare it with PyTorch CPU operations. They are
useful for:

- catching correctness regressions against PyTorch references;
- comparing host-side execution time for supported lowering patterns;
- measuring changes in allocation, copy, and CPU-backend overhead;
- quickly checking whether a pass or backend change affects existing kernels.

They are not final RISC-V hardware performance numbers. QEMU is useful for
correctness and RVV code-generation checks, but host-side Python benchmark
results do not predict exact RVV speedups. Final performance conclusions require
measurement on target RISC-V hardware.

Large gaps against PyTorch are expected at this stage. PyTorch CPU kernels are
heavily optimized and may use vendor math libraries and many host CPU threads.
Triton-RISCV's CPU backend currently measures the repository's lowering/runtime
path rather than a tuned CPU math library implementation.

For ELF/QEMU execution, use the workflow in
[`docs/06-RISCV-QEMU.md`](06-RISCV-QEMU.md).

## Fair Comparison Checklist

Before comparing two commits or publishing a number:

- use the same machine, power mode, CPU affinity, and background-load policy;
- record host architecture, CPU model, Python, PyTorch, Triton-RISCV, Buddy, and
  LLVM versions;
- use the same inputs and validate output before timing;
- keep PyTorch thread settings identical, and report them;
- warm both providers and exclude compilation unless compile latency is the
  quantity being measured;
- use enough repeats and report variation, not only one average;
- clear or isolate Triton caches when measuring compilation changes;
- inspect IR/assembly when attributing a speedup to a compiler optimization;
- avoid extrapolating host or QEMU results to real RISC-V hardware.

For compiler optimization work, pair runtime numbers with static evidence such
as allocation count, copied bytes, vector loads/stores, scalar fallback loops,
and final assembly size.

## 6. Benchmark Coverage

All files matching `python/performance/test_performance_*.py` use the common
comparison helper:

```python
benchmark.compare_providers(
    "bench_name(...)",
    {
        "torch": lambda: torch_reference(...),
        "triton-riscv": lambda: triton_riscv_implementation(...),
    },
)
```

`compare_providers` automatically adds `torch+buddy-mlir` from the `torch`
provider when Buddy is enabled, so benchmark files do not need a third callable
for the common case.

The helper checks correctness once before timing. Floating-point cases can pass
case-specific tolerances through `rtol` and `atol`.

When adding a new benchmark:

1. Write the Triton-RISCV implementation as a callable returning its output.
2. Write an equivalent PyTorch reference returning the same output structure.
3. Use `benchmark.compare_providers`.
4. Keep inputs fixed across providers so correctness and timing are comparable.
5. Clone tensors for providers that mutate inputs in place.

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'benchmark'`

Use the aggregate runner:

```sh
python python/performance/run_all.py matmul
```

Or set `PYTHONPATH` for direct script execution:

```sh
PYTHONPATH="$PWD/python/performance:$PYTHONPATH" \
python python/performance/test_performance_matmul.py
```

### `Unable to locate 'buddy-opt'`

Source the environment helper before running benchmarks:

```sh
source scripts/triton-riscv-env.sh
```

If your Buddy build is not in the default location, set `BUDDY_DIR` or
`BUDDY_MLIR_BINARY_DIR` before sourcing the helper.

### Stale Backend Changes Are Not Reflected

Rebuild the backend and remove cached kernels:

```sh
scripts/rebuild-triton-riscv.sh
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/artifacts/cache/benchmark}"
rm -rf "$TRITON_CACHE_DIR"
```

### A Benchmark Is Very Slow

`run_all.py` uses each script's default sizes, and some cases intentionally use
large tensors. Run a subset while iterating:

```sh
python python/performance/run_all.py vec_add matmul
```

For local debugging, temporarily reduce the sizes in the target benchmark file
or call its `bench_*` function from a short Python snippet with smaller inputs.

[Previous: Optimization opportunities](07-Optimization.md) · [Repository README](../README.md)

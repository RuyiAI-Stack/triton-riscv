import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel

triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def softmax_cpu_kernel(
    output_ptr,
    input_ptr,
    INPUT_ROW_STRIDE: tl.constexpr,
    OUTPUT_ROW_STRIDE: tl.constexpr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
):
    # Work directly on the input/output rows. Tensor materialization of the
    # conventional whole-row formulation creates several full-row stack
    # buffers on CPU. This three-pass form computes exp only once, stages the
    # numerator in the final output, and lets LLVM vectorize every inner loop.
    for row in tl.range(0, N_ROWS):
        input_base = row * INPUT_ROW_STRIDE
        output_base = row * OUTPUT_ROW_STRIDE

        maximum = -float("inf")
        for col in tl.range(0, N_COLS):
            maximum = tl.maximum(maximum, tl.load(input_ptr + input_base + col))

        denominator = 0.0
        for col in tl.range(0, N_COLS):
            numerator = tl.exp(tl.load(input_ptr + input_base + col) - maximum)
            denominator += numerator
            tl.store(output_ptr + output_base + col, numerator)

        for col in tl.range(0, N_COLS):
            output = tl.load(output_ptr + output_base + col)
            tl.store(output_ptr + output_base + col, output / denominator)


def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    key = (n_rows, n_cols, x.stride(0), y.stride(0), x.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            softmax_cpu_kernel,
            (1,),
            y,
            x,
            INPUT_ROW_STRIDE=x.stride(0),
            OUTPUT_ROW_STRIDE=y.stride(0),
            N_ROWS=n_rows,
            N_COLS=n_cols,
            allow_fp_reassoc=True,
        )
        _prepared_kernels[key] = runner
    runner(y, x)
    return y


def bench_softmax(size):
    torch.manual_seed(0)
    x = torch.randn(size, size, device="cpu")
    benchmark.compare_providers(
        f"bench_softmax(size={size})",
        {
            "torch": lambda: torch.softmax(x, axis=1),
            "triton-riscv": lambda: softmax(x),
        },
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for X in [2**i for i in range(10, 14, 1)]:
        bench_softmax(X)

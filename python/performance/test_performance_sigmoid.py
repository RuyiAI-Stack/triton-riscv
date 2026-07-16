import torch
import triton
import triton.language as tl

import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def exp_approx_negative(value):
    """Vector-friendly exp approximation for values in [-88, 0]."""
    value = tl.maximum(value, -88.0)
    exponent = tl.floor(value * 1.4426950408889634 + 0.5)
    reduced = value - exponent * 0.693359375
    reduced = reduced - exponent * -2.12194440e-4

    polynomial = 1.9875691500e-4
    polynomial = polynomial * reduced + 1.3981999507e-3
    polynomial = polynomial * reduced + 8.3334519073e-3
    polynomial = polynomial * reduced + 4.1665795894e-2
    polynomial = polynomial * reduced + 1.6666665459e-1
    polynomial = polynomial * reduced + 5.0000001201e-1
    polynomial = polynomial * reduced * reduced + reduced + 1.0

    exponent_bits = (exponent.to(tl.int32) + 127) << 23
    power_of_two = exponent_bits.to(tl.float32, bitcast=True)
    return polynomial * power_of_two


@triton.jit
def sigmoid_exp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask)
    result = 1.0 / (1.0 + tl.exp(-value))
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def sigmoid_polynomial_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask)
    exp_negative_abs = exp_approx_negative(-tl.abs(value))
    denominator = 1.0 + exp_negative_abs
    result = tl.where(
        value >= 0.0,
        1.0 / denominator,
        exp_negative_abs / denominator,
    )
    tl.store(output_ptr + offsets, result, mask=mask)


def sigmoid(x):
    if x.ndim != 1 or not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("sigmoid CPU fast path expects contiguous 1D float32")
    output = torch.empty_like(x)
    size = x.numel()
    use_polynomial = size > 2**18
    key = (size, x.dtype, use_polynomial)
    runner = _prepared_kernels.get(key)
    if runner is None:
        kernel = (
            sigmoid_polynomial_kernel if use_polynomial else sigmoid_exp_kernel
        )
        runner = prepare_cpu_kernel(
            kernel,
            (triton.cdiv(size, 1024),),
            x,
            output,
            size,
            BLOCK_SIZE=1024,
        )
        _prepared_kernels[key] = runner
    runner(x, output, size)
    return output


def bench_sigmoid(size):
    torch.manual_seed(0)
    x = torch.randn((size,), device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_sigmoid(size={size})",
        {
            "torch": lambda: torch.sigmoid(x),
            "triton-riscv": lambda: sigmoid(x),
        },
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**18, 2**20, 2**22]:
        bench_sigmoid(size)

import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def load_2d_tensor_col_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
):
    for offset in tl.range(0, N_ELEMENTS):
        tl.store(output_ptr + offset, tl.load(input_ptr + offset))


def run_load_2d_tensor_col(input_tensor):
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    runner = _prepared_kernels.get(n_elements)
    if runner is None:
        runner = prepare_cpu_kernel(
            load_2d_tensor_col_kernel,
            (1,),
            input_tensor,
            output,
            N_ELEMENTS=n_elements,
        )
        _prepared_kernels[n_elements] = runner
    runner(input_tensor, output)
    return output


def bench_load_2d_tensor_col(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    benchmark.compare_providers(
        f"bench_load_2d_tensor_col(rows={rows}, cols={cols})",
        {
            "torch": lambda: input_tensor.clone(),
            "triton-riscv": lambda: run_load_2d_tensor_col(input_tensor),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    bench_load_2d_tensor_col(8 * 32, 4 * 32)

import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def addptr_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    # The original two-elements-per-program grid performs millions of CPU
    # kernel calls. Preserve the pointer-add semantics in one structured loop,
    # which LLVM recognizes as a contiguous copy.
    for block_start in tl.range(0, N_ELEMENTS, 2):
        in1 = input_ptr + block_start
        in2 = in1 + 1
        out1 = output_ptr + block_start
        out2 = out1 + 1
        tl.store(out1, tl.load(in1))
        tl.store(out2, tl.load(in2))
    if HAS_TAIL:
        tail = N_ELEMENTS
        tl.store(output_ptr + tail, tl.load(input_ptr + tail))


def run_addptr(x):
    assert x.ndim == 1
    output = torch.empty_like(x)
    n_elements = x.numel()
    addptr_kernel[(1,)](
        x,
        output,
        N_ELEMENTS=n_elements - n_elements % 2,
        HAS_TAIL=n_elements % 2 != 0,
    )
    return output


def bench_addptr(size):
    x = torch.arange(size, device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_addptr(size={size})",
        {
            "torch": lambda: x.clone(),
            "triton-riscv": lambda: run_addptr(x),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [1024000, 2048000, 4096000, 8192000]:
        bench_addptr(size)

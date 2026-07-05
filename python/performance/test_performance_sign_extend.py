import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def sign_extend_kernel(offset_ptr, input_ptr, output_ptr, input_size):
    pid = tl.program_id(0)
    offset = tl.load(offset_ptr + pid).to(tl.int64)
    offsets = offset + tl.arange(0, 4)
    values = tl.load(input_ptr + offsets, mask=offsets < input_size, other=11)
    tl.store(output_ptr + pid * 4 + tl.arange(0, 4), values)


def run_sign_extend(x, offsets):
    assert x.ndim == 1
    assert offsets.ndim == 1
    assert offsets.numel() > 0
    assert torch.all(offsets >= 0).item()
    output = torch.empty((offsets.numel() * 4,), device=x.device, dtype=x.dtype)
    sign_extend_kernel[(offsets.numel(),)](offsets, x, output, x.numel())
    return output


def bench_sign_extend(size):
    x = torch.arange(size + 4, device="cpu", dtype=torch.int32)
    offsets = torch.arange(1, size + 1, device="cpu", dtype=torch.int32)
    gather_offsets = offsets[:, None].to(torch.int64) + torch.arange(4, device="cpu")
    benchmark.compare_providers(
        f"bench_sign_extend(size={size})",
        {
            "torch": lambda: torch.where(
                gather_offsets < x.numel(),
                x[gather_offsets.clamp(max=x.numel() - 1)],
                torch.full_like(gather_offsets, 11, dtype=x.dtype),
            ).reshape(-1),
            "triton-riscv": lambda: run_sign_extend(x, offsets),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [64 * 32, 256 * 32, 1024 * 32]:
        bench_sign_extend(size)

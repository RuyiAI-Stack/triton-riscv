import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def unstructured_mask_2d_kernel(
    input_ptr,
    output_ptr,
    mask_m_ptr,
    mask_n_ptr,
    rows,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = tl.load(mask_m_ptr + offs_m, mask=offs_m < rows, other=0) != 0
    mask_n = tl.load(mask_n_ptr + offs_n, mask=offs_n < cols, other=0) != 0
    input_ptrs = input_ptr + offs_m[:, None] * cols + offs_n[None, :]
    values = tl.load(
        input_ptrs,
        mask=(mask_m[:, None]) & (offs_n[None, :] < cols),
        other=-2.0,
    )
    output_ptrs = output_ptr + offs_m[:, None] * cols + offs_n[None, :]
    tl.store(
        output_ptrs,
        values,
        mask=(offs_m[:, None] < rows) & (mask_n[None, :]),
    )


def run_unstructured_mask(rows, cols):
    input_tensor = torch.arange(
        2, 2 + rows * cols, device="cpu", dtype=torch.float32
    ).reshape(rows, cols)
    output = torch.full((rows, cols), -1.0, device="cpu", dtype=torch.float32)
    mask_m = (torch.arange(rows, device="cpu") % 2 == 0).to(torch.int8)
    mask_n = torch.arange(cols, device="cpu") % 2 == 1
    if cols > 4:
        mask_n[4] = True
    if cols > 5:
        mask_n[5] = False
    block_m = triton.next_power_of_2(rows)
    block_n = triton.next_power_of_2(cols)
    unstructured_mask_2d_kernel[(1,)](
        input_tensor,
        output,
        mask_m,
        mask_n.to(torch.int8),
        rows,
        cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return output


def bench_unstructured_mask(rows, cols):
    def torch_reference():
        input_tensor = torch.arange(
            2, 2 + rows * cols, device="cpu", dtype=torch.float32
        ).reshape(rows, cols)
        output = torch.full((rows, cols), -1.0, device="cpu", dtype=torch.float32)
        mask_m = torch.arange(rows, device="cpu") % 2 == 0
        col_ids = torch.arange(cols, device="cpu")
        mask_n = ((col_ids % 2 == 1) | (col_ids == 4)) & (col_ids != 5)
        values = torch.where(
            mask_m[:, None], input_tensor, torch.full_like(input_tensor, -2.0)
        )
        return torch.where(mask_n[None, :], values, output)

    benchmark.compare_providers(
        f"bench_unstructured_mask(rows={rows}, cols={cols})",
        {
            "torch": torch_reference,
            "triton-riscv": lambda: run_unstructured_mask(rows, cols),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(4 * 32, 6 * 32), (8 * 32, 8 * 32), (16 * 32, 16 * 32)]:
        bench_unstructured_mask(rows, cols)

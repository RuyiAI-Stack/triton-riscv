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
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    for row in tl.range(0, ROWS):
        row_enabled = tl.load(mask_m_ptr + row) != 0
        for col in tl.range(0, COLS):
            col_enabled = tl.load(mask_n_ptr + col) != 0
            value = tl.load(input_ptr + row * COLS + col)
            value = tl.where(row_enabled, value, -2.0)
            tl.store(
                output_ptr + row * COLS + col,
                tl.where(col_enabled, value, -1.0),
            )


def run_unstructured_mask(input_tensor, mask_m, mask_n):
    rows, cols = input_tensor.shape
    output = torch.empty_like(input_tensor)
    unstructured_mask_2d_kernel[(1,)](
        input_tensor,
        output,
        mask_m,
        mask_n,
        ROWS=rows,
        COLS=cols,
    )
    return output


def bench_unstructured_mask(rows, cols):
    input_tensor = torch.arange(
        2, 2 + rows * cols, device="cpu", dtype=torch.float32
    ).reshape(rows, cols)
    mask_m = (torch.arange(rows, device="cpu") % 2 == 0).to(torch.int8)
    col_ids = torch.arange(cols, device="cpu")
    mask_n = (((col_ids % 2 == 1) | (col_ids == 4)) & (col_ids != 5)).to(torch.int8)

    def torch_reference():
        values = torch.where(
            mask_m.bool()[:, None],
            input_tensor,
            torch.full_like(input_tensor, -2.0),
        )
        return torch.where(
            mask_n.bool()[None, :], values, torch.full_like(input_tensor, -1.0)
        )

    benchmark.compare_providers(
        f"bench_unstructured_mask(rows={rows}, cols={cols})",
        {
            "torch": torch_reference,
            "triton-riscv": lambda: run_unstructured_mask(input_tensor, mask_m, mask_n),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(4 * 32, 6 * 32), (8 * 32, 8 * 32), (16 * 32, 16 * 32)]:
        bench_unstructured_mask(rows, cols)

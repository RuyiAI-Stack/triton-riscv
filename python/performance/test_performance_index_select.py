import benchmark
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

INDEX_SELECT_CASES = ((256, 128, 128), (512, 256, 256), (1024, 512, 512))


@triton.jit
def index_select_row_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    STRIDE_I: tl.constexpr,
    STRIDE_M: tl.constexpr,
    OUTPUT_STRIDE_M: tl.constexpr,
    N_PICKS: tl.constexpr,
    N_COLS: tl.constexpr,
):
    # Loading the whole picks x columns tensor creates large staging buffers in
    # the CPU lowering. A structured gather with a contiguous inner loop lets
    # LLVM vectorize each selected row directly into the destination.
    for pick in tl.range(0, N_PICKS):
        source_row = tl.load(indices_ptr + pick * STRIDE_I)
        for col in tl.range(0, N_COLS):
            value = tl.load(input_ptr + source_row * STRIDE_M + col)
            tl.store(output_ptr + pick * OUTPUT_STRIDE_M + col, value)


def run_index_select(input_tensor, indices, *, allocate=None, launch=None):
    picks = indices.numel()
    cols = input_tensor.shape[1]
    if allocate is None:
        allocate = torch.empty
    output = allocate((picks, cols), device="cpu", dtype=torch.float32)
    if launch is None:
        launch = index_select_row_kernel[(1,)]
    launch(
        input_tensor,
        output,
        indices,
        STRIDE_I=indices.stride(0),
        STRIDE_M=input_tensor.stride(0),
        OUTPUT_STRIDE_M=output.stride(0),
        N_PICKS=picks,
        N_COLS=cols,
    )
    return output


def make_index_select_inputs(rows, cols, picks):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    if picks <= 1:
        indices = torch.zeros((picks,), device="cpu", dtype=torch.int64)
    else:
        step = max((rows - 1) // (picks - 1), 1)
        indices = (torch.arange(picks, device="cpu") * step).to(torch.int64)
    return input_tensor, indices


def bench_index_select(rows, cols, picks):
    input_tensor, indices = make_index_select_inputs(rows, cols, picks)
    benchmark.compare_providers(
        f"bench_index_select(rows={rows}, cols={cols}, picks={picks})",
        {
            "torch": lambda: input_tensor.index_select(0, indices),
            "triton-riscv": lambda: run_index_select(input_tensor, indices),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols, picks in INDEX_SELECT_CASES:
        bench_index_select(rows, cols, picks)

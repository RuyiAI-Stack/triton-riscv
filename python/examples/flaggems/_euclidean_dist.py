import torch
import triton
import triton.language as tl


@triton.jit
def _euclidean_dist_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    N,
    M,
    D,
    stride_x1,
    stride_x2,
    stride_out,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    x1_row_ptr = x1_ptr + pid_n * stride_x1
    x2_row_ptr = x2_ptr + pid_m * stride_x2
    output_ptr_out = output_ptr + pid_n * stride_out + pid_m

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        x1_vals = tl.load(x1_row_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x2_vals = tl.load(x2_row_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        diff = x1_vals - x2_vals
        acc += diff * diff

    sq_dist = tl.sum(acc, axis=0)
    dist = tl.sqrt(sq_dist)
    tl.store(output_ptr_out, dist)


def _euclidean_dist(x1, x2):
    assert x1.ndim == 2, "x1 must be a 2D tensor"
    assert x2.ndim == 2, "x2 must be a 2D tensor"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same number of columns"

    N, D = x1.shape
    M = x2.shape[0]

    x1 = x1.contiguous()
    x2 = x2.contiguous()

    output = torch.empty((N, M), dtype=x1.dtype, device=x1.device)

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (N, M)
    _euclidean_dist_kernel[grid](
        x1,
        x2,
        output,
        N,
        M,
        D,
        x1.stride(0),
        x2.stride(0),
        output.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return output

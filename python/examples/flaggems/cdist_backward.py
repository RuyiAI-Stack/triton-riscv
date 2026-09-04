import torch
import triton
import triton.language as tl


@triton.jit
def _fill_zero_kernel(out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    tl.store(out_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)


@triton.jit
def _cdist_backward_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    cdist_ptr,
    grad_x1_ptr,
    batch_size,
    n1,
    n2,
    dim,
    p,
    BLOCK_N2: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Get program ids
    pid_b = tl.program_id(0)
    pid_n1 = tl.program_id(1)
    pid_dim = tl.program_id(2)

    # Offsets for dimension
    off_dim = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_dim = off_dim < dim

    # This kernel processes one (batch, n1_idx, dim_range) at a time
    # For each dim element, we sum over all n2 elements

    # n1_idx for this program
    n1_idx = pid_n1

    # Accumulator for gradient
    grad_x1_acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    # Iterate over n2 dimension (reduction)
    for n2_start in range(0, n2, BLOCK_N2):
        off_n2 = n2_start + tl.arange(0, BLOCK_N2)
        mask_n2 = off_n2 < n2

        # Load x1 for this batch, n1_idx, and dim elements
        # x1[b, n1_idx, d] for all d in off_dim
        x1_offset = pid_b * n1 * dim + n1_idx * dim + off_dim
        x1 = tl.load(x1_ptr + x1_offset, mask=mask_dim, other=0.0).to(tl.float32)

        # Load x2 for this batch, all n2 elements, and dim elements
        # x2[b, n2_j, d] for all n2_j in off_n2, all d in off_dim
        # x2_offset[b, n2_j, d] = b * n2 * dim + n2_j * dim + d
        x2_offset_base = pid_b * n2 * dim
        x2 = tl.load(
            x2_ptr + x2_offset_base + off_n2[:, None] * dim + off_dim[None, :],
            mask=mask_n2[:, None] & mask_dim[None, :],
            other=0.0,
        ).to(tl.float32)

        # Load grad for this batch, n1_idx, and all n2 elements
        # grad[b, n1_idx, n2_j] for all n2_j in off_n2
        grad_offset = pid_b * n1 * n2 + n1_idx * n2 + off_n2
        grad = tl.load(grad_ptr + grad_offset, mask=mask_n2, other=0.0).to(tl.float32)

        # Load cdist for this batch, n1_idx, and all n2 elements
        # cdist[b, n1_idx, n2_j] for all n2_j in off_n2
        cdist = tl.load(cdist_ptr + grad_offset, mask=mask_n2, other=1.0).to(tl.float32)

        # Compute diff: x1[n1_idx, d] - x2[n2_j, d]
        # x1 is [BLOCK_DIM], x2 is [BLOCK_N2, BLOCK_DIM]
        # Need to broadcast x1 across n2 dimension
        diff = x1[None, :] - x2  # [BLOCK_N2, BLOCK_DIM]

        # General p-norm derivative
        eps = 1e-12
        if p == 2.0:
            contrib = grad[:, None] * diff / (cdist[:, None] + eps)
        elif p == 1.0:
            sign_diff = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
            contrib = grad[:, None] * sign_diff
        elif p == float("inf"):
            # Approximation for p=inf backward: not typically requested for cdist,
            # but PyTorch handles it via max mask.
            # We'll raise error for p=inf in the python wrapper, but let's implement
            # standard general p here.
            contrib = tl.zeros_like(diff)
        else:
            abs_diff = tl.abs(diff)
            cdist_pow = tl.exp((p - 1.0) * tl.log(cdist[:, None] + eps))
            diff_pow = tl.exp((p - 1.0) * tl.log(abs_diff + eps))
            sign_diff = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
            contrib = grad[:, None] * sign_diff * diff_pow / (cdist_pow + eps)

        # Sum over n2 dimension
        grad_x1_acc += tl.sum(contrib, axis=0)

    # Store result
    store_offset = pid_b * n1 * dim + n1_idx * dim + off_dim
    tl.store(grad_x1_ptr + store_offset, grad_x1_acc, mask=mask_dim)


def _cdist_backward(grad, x1, x2, p, cdist):
    assert x1.device == x2.device == grad.device == cdist.device
    assert x1.shape[-1] == x2.shape[-1]  # feature dim must match
    assert x1.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Unsupported dtype: {x1.dtype}"
    if float(p) == float("inf"):
        raise NotImplementedError(
            "cdist backward for p=inf is not natively implemented yet."
        )
    if float(p) == 0.0:
        grad_x1 = torch.empty_like(x1)
        block_size = 256
        grid = (triton.cdiv(grad_x1.numel(), block_size),)
        _fill_zero_kernel[grid](grad_x1, grad_x1.numel(), BLOCK_SIZE=block_size)
        return grad_x1

    if x1.dim() == 2:
        batch_size = 1
        n1, dim = x1.shape
        n2 = x2.shape[0]
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        grad = grad.unsqueeze(0)
        cdist = cdist.unsqueeze(0)
    else:
        assert x1.dim() == 3 and x2.dim() == 3
        batch_size, n1, dim = x1.shape
        _, n2, _ = x2.shape

    # Ensure contiguity
    grad = grad.contiguous()
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    cdist = cdist.contiguous()

    grad_x1 = torch.empty_like(x1)

    # For float16/bfloat16, use float32 accumulation
    if x1.dtype in (torch.float16, torch.bfloat16):
        grad_x1_fp32 = torch.empty_like(x1, dtype=torch.float32)
    else:
        grad_x1_fp32 = grad_x1

    BLOCK_N2 = 64
    BLOCK_DIM = 64

    grid = (
        batch_size,
        n1,  # One program per (batch, n1) pair
        triton.cdiv(dim, BLOCK_DIM),
    )

    _cdist_backward_kernel[grid](
        grad,
        x1,
        x2,
        cdist,
        grad_x1_fp32,
        batch_size,
        n1,
        n2,
        dim,
        float(p),
        BLOCK_N2=BLOCK_N2,
        BLOCK_DIM=BLOCK_DIM,
    )

    if x1.dtype in (torch.float16, torch.bfloat16):
        grad_x1 = grad_x1_fp32.to(x1.dtype)
    else:
        grad_x1 = grad_x1_fp32

    if grad_x1.shape[0] == 1:
        grad_x1 = grad_x1.squeeze(0)
    return grad_x1

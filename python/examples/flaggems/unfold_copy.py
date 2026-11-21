import torch
import triton
import triton.language as tl

# unfold_copy: aten::unfold as a value copy. Only three input layouts have
# Triton kernels (2D dim=1, 3D dim=2, 3D dim=1); anything else raises
# NotImplementedError (upstream limitation, kept). ``dimension`` is taken
# modulo the rank without a range check (upstream behavior), unlike aten
# which raises for out-of-range dims.


@triton.jit
def _unfold_copy_kernel_2d(
    input_ptr, output_ptr, B, D, L, size, step, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # decode: output (B, L, size) row-major; input (B, D) assumed contiguous
    k = offs % size
    tmp = offs // size
    l_idx = tmp % L
    b_idx = tmp // L
    pos = l_idx * step + k
    input_idx = b_idx * D + pos
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)


@triton.jit
def _unfold_copy_kernel_3d_dim2(
    input_ptr, output_ptr, B, D, C, L, size, step, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # output (B, D, L, size): input stride is (D*C, C, 1), unfold applies to C
    k = offs % size
    tmp = offs // size
    l_idx = tmp % L
    tmp2 = tmp // L
    d_idx = tmp2 % D
    b_idx = tmp2 // D
    pos = l_idx * step + k
    input_idx = b_idx * D * C + d_idx * C + pos
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)


@triton.jit
def _unfold_copy_kernel_3d_dim1(
    input_ptr, output_ptr, B, D, C, L, size, step, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # output (B, L, C, size)
    k = offs % size
    tmp = offs // size
    c_idx = tmp % C
    tmp2 = tmp // C
    l_idx = tmp2 % L
    b_idx = tmp2 // L
    pos = l_idx * step + k
    input_idx = b_idx * D * C + pos * C + c_idx
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)


def unfold_copy(
    input: torch.Tensor, dimension: int, size: int, step: int
) -> torch.Tensor:
    """Value copy of ``input.unfold(dimension, size, step)``; only 2D/3D layouts supported."""
    if step <= 0:
        raise ValueError("step must be > 0")

    ndim = input.ndim
    d = dimension % ndim
    D = input.shape[d]

    if size > D:
        raise ValueError("size must be <= dimension size")

    L = (D - size) // step + 1

    output_shape = input.shape[:d] + (L,) + input.shape[d + 1 :] + (size,)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    numel_out = output.numel()
    BLOCK = 128
    grid = (triton.cdiv(numel_out, BLOCK),)

    if ndim == 2 and d == 1:
        B = input.shape[0]
        D = input.shape[1]
        _unfold_copy_kernel_2d[grid](
            input, output, B, D, L, size, step, numel_out, BLOCK=BLOCK
        )
    elif ndim == 3 and d == 2:
        B = input.shape[0]
        D = input.shape[1]
        C = input.shape[2]
        _unfold_copy_kernel_3d_dim2[grid](
            input, output, B, D, C, L, size, step, numel_out, BLOCK=BLOCK
        )
    elif ndim == 3 and d == 1:
        B = input.shape[0]
        D = input.shape[1]
        C = input.shape[2]
        _unfold_copy_kernel_3d_dim1[grid](
            input, output, B, D, C, L, size, step, numel_out, BLOCK=BLOCK
        )
    else:
        raise NotImplementedError(
            f"unfold_copy is only implemented for 2D tensors with dim=1 "
            f"and 3D tensors with dim=1 or dim=2. Got ndim={ndim}, dim={d}"
        )

    return output

import torch
import triton
import triton.language as tl


@triton.jit
def rot90_kernel_2d(
    in_ptr,
    out_ptr,
    n_elements,
    M,
    N,
    inner,
    k_norm,
    BLOCK_SIZE: tl.constexpr,
):
    """
    rot90 kernel for rotating a tensor by 90 degrees in the plane [0, 1].

    Input shape: [M, N, D2, D3, ...]
    Output shape for k=1,3: [N, M, D2, D3, ...]
    Output shape for k=0,2: [M, N, D2, D3, ...]

    Formulas (verified):
    - k=0 (identity): out[i,j] = in[i,j] -> in_dim0=out_dim0, in_dim1=out_dim1
    - k=1 (90° clockwise): out[i,j] = in[j, N-1-i]
      -> in_dim0=out_dim1, in_dim1=N-1-out_dim0
    - k=2 (180°): out[i,j] = in[M-1-i, N-1-j]
      -> in_dim0=M-1-out_dim0, in_dim1=N-1-out_dim1
    - k=3 (270° clockwise / 90° CCW): out[i,j] = in[M-1-j, i]
      -> in_dim0=M-1-out_dim1, in_dim1=out_dim0
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m_minus_1 = M - 1
    n_minus_1 = N - 1
    idx_2d = offsets // inner
    tail = offsets % inner

    if k_norm == 0:
        out_dim0 = idx_2d // N
        out_dim1 = idx_2d % N

        in_dim0 = out_dim0
        in_dim1 = out_dim1
    elif k_norm == 1:
        out_dim0 = idx_2d // M
        out_dim1 = idx_2d % M

        in_dim0 = out_dim1
        in_dim1 = n_minus_1 - out_dim0
    elif k_norm == 2:
        out_dim0 = idx_2d // N
        out_dim1 = idx_2d % N

        in_dim0 = m_minus_1 - out_dim0
        in_dim1 = n_minus_1 - out_dim1
    else:
        out_dim0 = idx_2d // M
        out_dim1 = idx_2d % M

        in_dim0 = m_minus_1 - out_dim1
        in_dim1 = out_dim0

    in_offset = (in_dim0 * N + in_dim1) * inner + tail

    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def rot90_2d(inp, k, dims, out):
    """Handle the case when dims = [0, 1] using optimized Triton kernel."""
    M = inp.shape[dims[0]]
    N = inp.shape[dims[1]]
    n_elements = out.numel()
    if n_elements == 0:
        return

    # Normalize k to 0, 1, 2, 3
    k_norm = ((k % 4) + 4) % 4

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    BLOCK_SIZE = 1024

    rot90_kernel_2d[grid](
        inp,
        out,
        n_elements,
        M,
        N,
        inp.numel() // (M * N) if M * N != 0 else 0,
        k_norm,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def rot90(input, k=1, dims=[0, 1]):
    """Rotate an n-D tensor by 90 degrees in the plane specified by dims.

    Args:
        input: the input tensor
        k: number of times to rotate (default: 1)
        dims: axis to rotate (default: [0, 1])

    Returns:
        Rotated tensor
    """
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    if len(dims) != 2:
        raise RuntimeError(
            f"expected total rotation dims == 2, but got dims = {len(dims)}"
        )

    dim0, dim1 = dims[0], dims[1]
    if dim0 == dim1:
        raise RuntimeError(
            f"expected rotation dims to be different, but got dim0 = {dim0} and dim1 = {dim1}"
        )

    dim0 = dim0 if dim0 >= 0 else dim0 + x.ndim
    if dim0 >= x.ndim or dim0 < -x.ndim:
        raise RuntimeError(f"Rotation dim0 out of range, dim0 = {dim0}")
    dim1 = dim1 if dim1 >= 0 else dim1 + x.ndim
    if dim1 >= x.ndim or dim1 < -x.ndim:
        raise RuntimeError(f"Rotation dim1 out of range, dim1 = {dim1}")

    if dim0 == dim1:
        raise RuntimeError(
            f"expected rotation dims to be different, but got dim0 = {dim0} and dim1 = {dim1}"
        )

    k_norm = ((k % 4) + 4) % 4

    if k_norm in (0, 2):
        out_shape = list(x.shape)
    else:
        out_shape = list(x.shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    if dim0 == 0 and dim1 == 1:
        x_rot = x
        out_rot = out
    else:
        ndim = x.ndim
        perm = [dim0, dim1]
        for i in range(ndim):
            if i != dim0 and i != dim1:
                perm.append(i)

        inv = [0] * ndim
        inv[dim0] = 0
        inv[dim1] = 1
        idx = 2
        for i in range(ndim):
            if i != dim0 and i != dim1:
                inv[i] = idx
                idx += 1

        x_rot = x.permute(perm).contiguous()
        if k_norm in (1, 3):
            out_shape_front = list(x_rot.shape)
            out_shape_front[0], out_shape_front[1] = (
                out_shape_front[1],
                out_shape_front[0],
            )
            out_rot = torch.empty(out_shape_front, device=x.device, dtype=x.dtype)
        else:
            out_rot = torch.empty_like(x_rot)

    rot90_2d(x_rot, k_norm, (0, 1), out_rot)

    if dim0 == 0 and dim1 == 1:
        return out_rot
    out.copy_(out_rot.permute(inv))
    return out

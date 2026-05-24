import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["alpha", "beta"])
def addmv_kernel(
    A,
    B,
    Inp,
    Out,
    N,
    M,
    alpha,
    beta,
    stride_an,
    stride_am,
    stride_bm,
    stride_in,
    stride_outn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)[:, None]
    Inp_ptrs = Inp + offset_n * stride_in
    inp = tl.load(Inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    Out_ptrs = Out + offset_n * stride_outn
    out_block = acc * alpha + inp * beta
    tl.store(Out_ptrs, out_block, mask=n_mask)


def addmv(self, mat, vec, *, beta=1, alpha=1):
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert (
        self.shape == () or self.numel() == 1 or self.shape == (mat.shape[0],)
    ), "Incompatible self shape"
    N, M = mat.shape
    out = torch.empty((N,), device=mat.device, dtype=mat.dtype)
    self = self.broadcast_to(out.shape)

    BLOCK_N = 32
    BLOCK_M = 32
    grid = (triton.cdiv(N, BLOCK_N),)

    addmv_kernel[grid](
        mat,
        vec,
        self,
        out,
        N,
        M,
        alpha,
        beta,
        mat.stride(0),
        mat.stride(1),
        vec.stride(0),
        self.stride(0),
        out.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )
    return out


def addmv_out(self, mat, vec, *, beta=1, alpha=1, out=None):
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert (
        self.shape == () or self.numel() == 1 or self.shape == (mat.shape[0],)
    ), "Incompatible self shape"
    N, M = mat.shape
    if out is None:
        out = torch.empty((N,), device=mat.device, dtype=mat.dtype)
    else:
        assert out.shape == (N,), "Incompatible output shape"

    self = self.broadcast_to(out.shape)

    BLOCK_N = 32
    BLOCK_M = 32
    grid = (triton.cdiv(N, BLOCK_N),)

    addmv_kernel[grid](
        mat,
        vec,
        self,
        out,
        N,
        M,
        alpha,
        beta,
        mat.stride(0),
        mat.stride(1),
        vec.stride(0),
        self.stride(0),
        out.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )
    return out

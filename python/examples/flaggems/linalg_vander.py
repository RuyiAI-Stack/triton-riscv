import torch
import triton
import triton.language as tl


@triton.jit
def _vander_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row).to(tl.float32)
    power = cols.to(tl.int32)
    ax = tl.abs(x)
    mag = tl.where(
        power == 0,
        1.0,
        tl.where(ax == 0, 0.0, tl.exp(power.to(tl.float32) * tl.log(ax))),
    )
    sign = tl.where((x < 0) & ((power % 2) == 1), -1.0, 1.0)
    tl.store(out_ptr + row * n_cols + cols, mag * sign, mask=mask)


def linalg_vander(x, N=None):
    if N is None:
        N = x.shape[-1]
    flat = x.reshape(-1).contiguous()
    out = torch.empty(flat.shape[0], N, dtype=torch.float32, device=x.device)
    block = 1
    while block < N:
        block *= 2
    if flat.numel():
        _vander_kernel[(flat.shape[0],)](flat, out, flat.shape[0], N, BLOCK=block)
    return out.reshape(x.shape + (N,))


__all__ = ["linalg_vander"]

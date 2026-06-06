import torch
import triton
import triton.language as tl

from .rand import philox_backend_seed_offset


@triton.jit
def multinomial_with_replacement(
    cdf_ptr,
    out_ptr,
    K,
    N,
    philox_seed,
    philox_offset,
    NBLOCK: tl.constexpr = 128,
):
    y_off = tl.program_id(1) * N
    n = tl.program_id(0) * NBLOCK + tl.arange(0, NBLOCK)

    # Generate a single uniform random number via philox
    seed = philox_seed.to(tl.int64)
    po = philox_offset.to(tl.int64)
    c0 = (po & 0xFFFFFFFF).to(tl.uint32) + y_off + n
    c1 = ((po >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(seed, c0, c1, _O, _O)
    rv = r0.to(tl.int32, bitcast=True)
    rv = tl.where(rv < 0, -rv - 1, rv).to(tl.float32) * 4.6566127342e-10

    rv += 0.0001
    rv = tl.where(rv > 0.9999, 0.9999, rv)

    cdf_ptr += tl.program_id(1) * K
    start = tl.zeros((NBLOCK,), dtype=tl.int32)
    end = tl.zeros((NBLOCK,), dtype=tl.int32) + K - 1
    steps = tl.math.log2(K.to(tl.float32)).to(tl.int32) + 1
    for _ in range(steps):
        mid = start + (end - start) // 2
        x = tl.load(cdf_ptr + mid, mask=n < N)
        start = tl.where(x < rv, mid + 1, start)
        end = tl.where(x < rv, end, mid)

    start = tl.where(start >= K, K - 1, start)

    tl.store(out_ptr + y_off + n, start, mask=n < N)


def cumsum_normalized(prob, dim=-1):
    cumsum = torch.cumsum(prob, dim=dim)
    last_val = cumsum.select(dim, -1).unsqueeze(dim)
    return cumsum / last_val


def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    assert prob.dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.float64,
    )
    assert 0 < prob.dim() <= 2, "prob_dist must be 1 or 2 dim"
    n_categories = prob.size(-1)
    assert n_categories <= (1 << 24), "number of categories cannot exceed 2^24"
    assert with_replacement or n_samples <= n_categories, (
        "cannot sample n_samples > prob.size(-1) samples without replacement."
    )

    if (not with_replacement) or n_samples == 1:
        q = torch.empty_like(prob).exponential_(1.0, generator=gen)
        s = torch.div(prob, q, out=q)
        if n_samples == 1:
            return torch.argmax(s, dim=-1, keepdim=True).to(torch.int64)
        else:
            _, indices = torch.topk(s, n_samples, dim=-1)
            return indices.to(torch.int64)

    cum_prob = cumsum_normalized(prob, dim=-1)

    if cum_prob.dim() == 1:
        n_dist = 1
        out = torch.empty((n_samples,), device=prob.device, dtype=torch.int64)
    else:
        n_dist = cum_prob.size(0)
        out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int64)

    increment = n_dist * n_samples
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)

    def grid(META):
        return (triton.cdiv(n_samples, META["NBLOCK"]), n_dist)

    multinomial_with_replacement[grid](
        cum_prob, out, n_categories, n_samples, philox_seed, philox_offset
    )
    return out

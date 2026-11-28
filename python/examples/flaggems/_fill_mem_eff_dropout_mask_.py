import torch
import triton
import triton.language as tl

# _fill_mem_eff_dropout_mask_: fills a contiguous 4-D float32 tensor with
# uniform values in [0, 1) drawn from a Philox4x32-10 stream identified by
# (seed, offset), mirroring the curand counter layout the upstream kernel
# reproduces: counter.x = offset // 4 + block_index, one philox block per
# four consecutive floats.
#
# The dropout probability is accepted for schema compatibility but does not
# participate in the computation (mirroring upstream, where the kernel only
# fills the uniform stream). Only a contiguous 4-D float32 buffer is
# accepted (upstream asserts). The stream follows the documented layout;
# bit-exact equivalence with curand itself is not claimed by this port.


# do_not_specialize keeps base/key0/key1 runtime scalars: a seed whose low
# word is exactly 1 would otherwise be constexpr-specialized (the ==1 trap)
# and break the .to(tl.uint32) key setup.
@triton.jit(do_not_specialize=["base", "key0", "key1"])
def _fill_mem_eff_dropout_mask_kernel(
    out_ptr, n_elements, base, key0, key1, BLOCK: tl.constexpr
):
    # base = offset // 4 (host-computed); one philox block per 4 floats.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offsets < n_elements
    block = offsets // 4
    lane = offsets - block * 4

    c0 = (base + block).to(tl.uint32)
    c1 = tl.zeros([BLOCK], dtype=tl.uint32)
    c2 = tl.zeros([BLOCK], dtype=tl.uint32)
    c3 = tl.zeros([BLOCK], dtype=tl.uint32)
    k0 = tl.zeros([BLOCK], dtype=tl.uint32) + key0.to(tl.uint32)
    k1 = tl.zeros([BLOCK], dtype=tl.uint32) + key1.to(tl.uint32)

    for r in tl.static_range(10):
        p0 = c0.to(tl.uint64) * 0xD2511F53
        hi0 = (p0 >> 32).to(tl.uint32)
        lo0 = (p0 & 0xFFFFFFFF).to(tl.uint32)
        p1 = c2.to(tl.uint64) * 0xCD9E8D57
        hi1 = (p1 >> 32).to(tl.uint32)
        lo1 = (p1 & 0xFFFFFFFF).to(tl.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if r < 9:
            k0 = k0 + 0x9E3779B9
            k1 = k1 + 0xBB67AE85

    bits = tl.where(lane == 0, c0, tl.where(lane == 1, c1, tl.where(lane == 2, c2, c3)))
    u = bits.to(tl.float64) / 4294967296.0
    # Clamp below 1.0: the largest counter values round up to 1.0f on the
    # final fp32 store (2^-32 < half an ulp of 1.0), which would leave the
    # promised [0, 1) range.
    res = tl.minimum(u.to(tl.float32), 0.99999994)
    tl.store(out_ptr + offsets, res, mask=m)


def _fill_mem_eff_dropout_mask_(self, dropout_p, seed, offset):
    """Fill self with the Philox(seed, offset) uniform stream; returned as self.

    dropout_p does not participate in the computation (upstream fills the
    plain U[0,1) stream; the thresholding happens in the memory-efficient
    attention kernels that consume this mask).
    """
    assert self.is_contiguous(), "self must be contiguous"
    assert self.dim() == 4, "_fill_mem_eff_dropout_mask_ expects a 4D tensor"
    assert self.dtype == torch.float32, "self must be a float32 tensor"

    n = self.numel()
    if n:
        base = int(offset) // 4
        key0 = int(seed) & 0xFFFFFFFF
        key1 = (int(seed) >> 32) & 0xFFFFFFFF
        _fill_mem_eff_dropout_mask_kernel[(triton.cdiv(n, 1024),)](
            self, n, base, key0, key1, BLOCK=1024
        )
    return self

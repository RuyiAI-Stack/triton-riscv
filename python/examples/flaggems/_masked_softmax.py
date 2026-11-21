import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    K,
    inner,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    outer_idx = pid // inner
    inner_idx = pid % inner
    base = outer_idx * K * inner + inner_idx

    k_offsets = tl.arange(0, BLOCK_K)
    valid = k_offsets < K

    x = tl.load(input_ptr + base + k_offsets * inner, mask=valid, other=0.0)
    # True marks a masked-out lane; padded lanes count as masked so they
    # never contribute to the reduction.
    m = tl.load(mask_ptr + base + k_offsets * inner, mask=valid, other=True)

    x_f32 = x.to(tl.float32)
    neg_inf = float("-inf")
    x_masked = tl.where(m, neg_inf, x_f32)
    # Overflow guard: subtract the row max of the masked row before exp.
    # Masked lanes exp() a neutral 0.0 and are zeroed right after (exp of
    # -inf - a -inf max would be NaN and poison the sum), so a fully-masked
    # row sums to 0 and divides 0/0 = NaN. This matches aten._masked_softmax,
    # which does NOT special-case fully-masked rows.
    x_max = tl.max(x_masked, axis=0)
    x_sub = tl.where(m, 0.0, x_f32 - x_max)
    x_exp = tl.exp(x_sub)
    x_exp = tl.where(m, 0.0, x_exp)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    tl.store(
        output_ptr + base + k_offsets * inner,
        out.to(output_ptr.dtype.element_ty),
        mask=valid,
    )


def _masked_softmax(self, mask, dim=None, mask_type=None):
    # Softmax over `dim` with bool-masked positions removed: out ==
    # softmax(x.masked_fill(mask, -inf), dim); masked positions come out
    # exactly 0.0 and fully-masked rows come out NaN (the aten contract).
    # mask_type selects the broadcast: 0 right-aligns the mask over the
    # trailing dims (source mask), 1 reshapes a 2-D [B,K] mask across the
    # middle dims (padding mask), 2 is an elementwise same-shape mask.
    assert mask.dtype == torch.bool, "Mask should be a boolean tensor"
    assert mask_type in (0, 1, 2), "mask_type should be 0, 1 or 2"
    assert self.ndim >= 1, "Input tensor must have at least 1 dimension"

    input_ndim = self.ndim
    if dim is None:
        dim = input_ndim - 1
    if dim < 0:
        dim += input_ndim

    self = self.contiguous()
    K = self.shape[dim]

    outer = 1
    for i in range(dim):
        outer *= self.shape[i]
    inner = 1
    for i in range(dim + 1, input_ndim):
        inner *= self.shape[i]

    M = outer * inner
    output = torch.empty_like(self)
    if M == 0 or K == 0:
        return output

    # Materialize the mask to the input shape, then run the generic strided
    # softmax kernel over `dim`. (Upstream also keeps last-dim 2-D-mask fast
    # paths that index the mask directly; on this CPU backend the full-shape
    # bool materialization is cheap, so the single kernel covers all cases.)
    if mask_type == 2:
        mask_contig = mask.expand(self.shape).contiguous()
    elif mask_type == 0:
        new_shape = [1] * (input_ndim - mask.ndim) + list(mask.shape)
        mask_contig = mask.reshape(new_shape).broadcast_to(self.shape).contiguous()
    else:  # mask_type == 1
        new_shape = (
            [mask.shape[0]]
            + [1] * (dim - 1)
            + [mask.shape[1]]
            + [1] * (input_ndim - dim - 1)
        )
        mask_contig = mask.reshape(new_shape).broadcast_to(self.shape).contiguous()

    BLOCK_K = triton.next_power_of_2(K)
    grid = (M,)
    _masked_softmax_kernel[grid](
        self,
        mask_contig,
        output,
        K,
        inner,
        BLOCK_K=BLOCK_K,
    )
    return output

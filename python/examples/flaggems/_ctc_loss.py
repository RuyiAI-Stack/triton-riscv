import torch
import triton
import triton.language as tl

# _ctc_loss (aten::_ctc_loss): the CTC forward returning
# (neg_log_likelihood (N,), log_alpha (N, T, 2S+1)).
# The DP runs in fp64 inside the kernels (log-domain forward algorithm over
# the extended blank-interleaved label lattice); probed contracts: blank out
# of range -> RuntimeError, empty log_probs -> RuntimeError, targets narrower
# than max(target_lengths) -> RuntimeError, target_lengths > input_lengths
# gives inf losses (zeroed by zero_infinity=True).
#
# One kernel launch per timestep (init / step) plus a finalize kernel; the
# host only drives the timestep loop.


@triton.jit
def _ctc_init_kernel(
    lp_ptr,
    tg_ptr,
    tl_ptr,
    alpha_ptr,
    n_lanes,
    states,
    seq,
    c,
    blank,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lmask = lane < n_lanes
    n = lane // states
    s = lane - n * states
    valid = lmask

    blank_lp = tl.load(lp_ptr + n * c + blank, mask=valid, other=float("-inf")).to(
        tl.float64
    )
    first = tl.load(tg_ptr + n * seq, mask=valid, other=0).to(tl.int64)
    first_lp = tl.load(lp_ptr + n * c + first, mask=valid, other=float("-inf")).to(
        tl.float64
    )

    # The lattice spans the full padded extent: target_lengths only select
    # the final states (mirroring aten's log_alpha pattern).
    res = tl.where(s == 0, blank_lp, float("-inf"))
    res = tl.where(s == 1, first_lp, res)
    tl.store(alpha_ptr + n * states + s, res, mask=lmask)


@triton.jit
def _ctc_loss_kernel(
    lp_ptr,
    tg_ptr,
    tl_ptr,
    prev_ptr,
    cur_ptr,
    n_lanes,
    t,
    n_batch,
    c,
    seq,
    states,
    blank,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lmask = lane < n_lanes
    n = lane // states
    s = lane - n * states
    valid = lmask

    # label(s): even -> blank; odd -> targets[n, (s-1)//2].
    odd = (s % 2) == 1
    lab_idx = (s - 1) // 2
    lab = tl.where(
        odd,
        tl.load(tg_ptr + n * seq + lab_idx, mask=valid & odd, other=0).to(tl.int64),
        blank,
    )
    lab2 = tl.where(
        (s % 2) == 1,
        tl.load(tg_ptr + n * seq + (s - 3) // 2, mask=valid & (s >= 3), other=-1).to(
            tl.int64
        ),
        blank,
    )
    # The skip transition s-2 -> s only exists when the two labels differ.
    allow2 = valid & (s >= 2) & (lab != lab2)

    prev = prev_ptr + n * states
    a0 = tl.load(prev + s, mask=valid, other=float("-inf")).to(tl.float64)
    a1 = tl.load(prev + s - 1, mask=valid & (s >= 1), other=float("-inf")).to(
        tl.float64
    )
    a2 = tl.load(prev + s - 2, mask=allow2, other=float("-inf")).to(tl.float64)

    m = tl.maximum(tl.maximum(a0, a1), a2)
    se = tl.exp(a0 - m) + tl.exp(a1 - m) + tl.exp(a2 - m)
    r = tl.where(m == float("-inf"), float("-inf"), m + tl.log(se))

    lp = tl.load(
        lp_ptr + (t * n_batch + n) * c + lab, mask=valid, other=float("-inf")
    ).to(tl.float64)
    res = tl.where(valid, r + lp, float("-inf"))
    tl.store(cur_ptr + n * states + s, res, mask=lmask)


def _ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False
):
    """CTC forward returning (neg_log_likelihood, log_alpha)."""
    lp = log_probs.contiguous()
    if lp.dim() != 3 or lp.numel() == 0:
        raise RuntimeError("log_probs tensor must not be empty")
    t_max, n, c = lp.shape
    if blank < 0 or blank >= c:
        raise RuntimeError("blank must be in label range")
    il = torch.as_tensor(input_lengths, dtype=torch.long, device=lp.device).reshape(-1)
    tl_ = torch.as_tensor(target_lengths, dtype=torch.long, device=lp.device).reshape(
        -1
    )
    if targets.dim() == 1:
        # Flat concatenated targets: rebuild the padded (N, S) layout.
        seq = int(tl_.max().item()) if tl_.numel() else 0
        padded = torch.zeros(n, seq, dtype=torch.long, device=lp.device)
        offset = 0
        for i in range(n):
            length = int(tl_[i].item())
            padded[i, :length] = targets[offset : offset + length]
            offset += length
        targets = padded
    else:
        targets = targets.contiguous()
        seq = targets.shape[1]
    if seq < int(tl_.max().item()):
        raise RuntimeError(
            f"Expected tensor to have size at least {int(tl_.max().item())} at "
            f"dimension 1, but got size {seq} for argument #2 'targets'"
        )

    # aten sizes the state axis by the batch max target length (probed:
    # S=2, tl=[1] yields a (N, T, 3) log_alpha), not by the padded extent.
    seq_eff = int(tl_.max().item()) if tl_.numel() else 0
    states = 2 * seq_eff + 1
    # Internal layout is (T, N, states): each timestep plane is contiguous,
    # so the kernels can index a plane as a dense (N, states) buffer.
    alpha = torch.full(
        (t_max, n, states), float("-inf"), dtype=torch.float64, device=lp.device
    )
    lanes = n * states
    grid = (triton.cdiv(lanes, 256),)
    _ctc_init_kernel[grid](
        lp, targets, tl_, alpha, lanes, states, seq, c, blank, BLOCK=256
    )
    for t in range(1, t_max):
        _ctc_loss_kernel[grid](
            lp,
            targets,
            tl_,
            alpha[t - 1],
            alpha[t],
            lanes,
            t,
            n,
            c,
            seq,
            states,
            blank,
            BLOCK=256,
        )
    # Final-state extraction on the host: loss_n = -logsumexp over the
    # sample's two final states (2*tl_n and 2*tl_n - 1) at t = il_n - 1.
    rows = torch.arange(n, device=lp.device)
    plane = alpha[il - 1, rows, :]
    a_final = plane[rows, 2 * tl_]
    a_prev = plane[rows, torch.clamp(2 * tl_ - 1, min=0)].where(
        tl_ >= 1, torch.full_like(tl_, 0, dtype=alpha.dtype).fill_(float("-inf"))
    )
    nll = -torch.logsumexp(torch.stack([a_final, a_prev], dim=1), dim=1)
    if zero_infinity:
        nll = torch.where(torch.isinf(nll), torch.zeros_like(nll), nll)
    # aten exposes log_alpha as (N, T, states).
    return nll.to(lp.dtype), alpha.permute(1, 0, 2).contiguous().to(lp.dtype)

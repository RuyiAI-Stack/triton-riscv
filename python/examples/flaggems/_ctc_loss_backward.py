import torch
import triton
import triton.language as tl

# _ctc_loss_backward (aten::_ctc_loss_backward): consumes the forward's
# (neg_log_likelihood, log_alpha) and returns grad_log_probs (T, N, C).
# The beta recursion runs backwards over the same blank-interleaved lattice:
#   beta[il-1, s] = 0 on the two final states (2*tl and 2*tl-1),
#   beta[t, s]    = logsumexp over s' in {s, s+1, s+2} (the s+2 skip only
#                   when label(s+2) != label(s)) of beta[t+1, s'] +
#                   lp[t+1, label(s')],
#   grad[t, n, c] = g[n] * (softmax(lp)[t,n,c] - P(z_t = c)) with
#   P(z_t = c)    = exp(logsumexp_{s: label(s)=c}(alpha[t,s] + beta[t,s])
#                     + nll[n]).
# The softmax term mirrors the probed aten/autograd convention (the
# log_softmax jacobian is folded in; rows sum to zero).
#
# The beta sweep runs on the host in vectorized torch (a T-loop of
# (N, states, 3) logsumexp reductions): chained dependent kernel launches
# exposed an execution-order race on this CPU backend, and the sweep is
# only O(T * N * states). The O(T * N * C * states) combine runs in the
# Triton kernel below. All math is fp64.

_NEG_INF = float("-inf")


@triton.jit
def _ctc_loss_backward_kernel(
    alpha_ptr,
    beta_ptr,
    lab_ptr,
    g_ptr,
    lp_ptr,
    nll_ptr,
    il_ptr,
    out_ptr,
    n_lanes,
    n_batch,
    c,
    states,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lmask = lane < n_lanes
    cc = lane % c
    tn = lane // c
    n = tn % n_batch
    t = tn // n_batch
    base = (t * n_batch + n) * states

    # aten's _ctc_loss leaves unreachable lattice slots filled with
    # uninitialized memory (probed: alpha hashes differ across processes,
    # magnitudes > 1e35); genuine log-alpha values live above ~-1e4 for any
    # sane input, so anything beyond +-1e29 is sentinel garbage and is
    # treated as -inf here.
    m = tl.zeros([BLOCK], dtype=tl.float64) + float("-inf")
    for s in range(states):
        lab = tl.load(lab_ptr + n * states + s, mask=lmask, other=-1).to(tl.int64)
        match = lmask & (lab == cc)
        a = tl.load(alpha_ptr + base + s, mask=match, other=float("-inf")).to(
            tl.float64
        )
        # a == a filters NaN garbage; the range check filters the huge
        # sentinel magnitudes.
        a_sane = (a == a) & (a < 1e29) & (a > -1e29)
        a = tl.where(a_sane, a, float("-inf"))
        b = tl.load(beta_ptr + base + s, mask=match, other=float("-inf")).to(tl.float64)
        m = tl.maximum(m, tl.where(match, a + b, float("-inf")))
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    for s in range(states):
        lab = tl.load(lab_ptr + n * states + s, mask=lmask, other=-1).to(tl.int64)
        match = lmask & (lab == cc)
        a = tl.load(alpha_ptr + base + s, mask=match, other=float("-inf")).to(
            tl.float64
        )
        # a == a filters NaN garbage; the range check filters the huge
        # sentinel magnitudes.
        a_sane = (a == a) & (a < 1e29) & (a > -1e29)
        a = tl.where(a_sane, a, float("-inf"))
        b = tl.load(beta_ptr + base + s, mask=match, other=float("-inf")).to(tl.float64)
        acc += tl.where(match & (m > float("-inf")), tl.exp((a + b) - m), 0.0)
    g = tl.load(g_ptr + n, mask=lmask, other=0.0).to(tl.float64)
    nll_n = tl.load(nll_ptr + n, mask=lmask, other=0.0).to(tl.float64)
    # alpha + beta is the unnormalized log-joint: rescale by exp(nll) so the
    # row sums over classes hit 1 (P(z_t = c) = exp(lse + nll)).
    p = tl.where(m == float("-inf"), 0.0, tl.exp(m + tl.log(acc) + nll_n))
    sm = tl.exp(
        tl.load(lp_ptr + (t * n_batch + n) * c + cc, mask=lmask, other=0.0).to(
            tl.float64
        )
    )
    # Rows beyond the sample's input length do not influence the loss:
    # autograd (and aten's raw backward) return exactly zero there.
    iln = tl.load(il_ptr + n, mask=lmask, other=0).to(tl.int64)
    res = tl.where(t < iln, g * (sm - p), 0.0)
    tl.store(out_ptr + lane, res.to(out_ptr.dtype.element_ty), mask=lmask)


def _ctc_loss_backward(
    grad_out,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank=0,
):
    """CTC backward returning grad_log_probs (T, N, C)."""
    lp = log_probs.contiguous()
    t_max, n, c = lp.shape
    il = torch.as_tensor(input_lengths, dtype=torch.long, device=lp.device).reshape(-1)
    tl_ = torch.as_tensor(target_lengths, dtype=torch.long, device=lp.device).reshape(
        -1
    )
    if targets.dim() == 1:
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
    states = log_alpha.shape[-1]

    # Host-prebuilt lattice tables: labels (N, states) and skip flags.
    labels = torch.zeros(n, states, dtype=torch.long, device=lp.device)
    labels[:, 0::2] = blank
    labels[:, 1::2] = targets[:, : states // 2]
    skip = torch.zeros(n, 1, states, dtype=torch.bool, device=lp.device)
    if states > 2:
        skip[:, 0, 1 : states - 2] = labels[:, 3:states] != labels[:, 1 : states - 2]

    # Internally the planes are (T, N, states); every producer (this
    # forward and aten) hands us (N, T, states).
    alpha_tns = log_alpha.transpose(0, 1).contiguous().to(torch.float64)
    lp64 = lp.to(torch.float64)

    # Host beta sweep (vectorized over (N, states) per timestep).
    beta = torch.full(
        (t_max, n, states), _NEG_INF, dtype=torch.float64, device=lp.device
    )
    for i in range(n):
        beta[int(il[i]) - 1, i, 2 * int(tl_[i])] = 0.0
        if int(tl_[i]) >= 1:
            beta[int(il[i]) - 1, i, 2 * int(tl_[i]) - 1] = 0.0
    lp_state = lp64.gather(
        2, labels.expand(t_max, n, states)
    )  # (T, N, states): lp[t, n, label(s)]
    for t in range(int(il.max().item()) - 2, -1, -1):
        nxt = beta[t + 1]  # (N, states)
        # transitions s -> {s, s+1, s+2 (skip only)}
        t0 = nxt + lp_state[t + 1]
        t1 = torch.full_like(t0, _NEG_INF)
        t1[:, :-1] = nxt[:, 1:] + lp_state[t + 1][:, 1:]
        t2 = torch.full_like(t0, _NEG_INF)
        t2[:, :-2] = torch.where(
            skip[:, 0, : states - 2], nxt[:, 2:] + lp_state[t + 1][:, 2:], t2[:, :-2]
        )
        stacked = torch.stack([t0, t1, t2], dim=-1)
        # Samples whose own input ends at t + 1 keep their seed plane (the
        # per-sample loop bound): the sweep must not recurse past it.
        new_beta = torch.logsumexp(stacked, dim=-1)
        is_seed_plane = (il - 1 == t).unsqueeze(1)
        beta[t] = torch.where(is_seed_plane, beta[t], new_beta)

    g = grad_out.reshape(-1).to(torch.float64)
    nll64 = neg_log_likelihood.contiguous().to(torch.float64)
    out = torch.empty(t_max, n, c, dtype=lp.dtype, device=lp.device)
    combine_lanes = t_max * n * c
    _ctc_loss_backward_kernel[(triton.cdiv(combine_lanes, 256),)](
        alpha_tns,
        beta,
        labels,
        g,
        lp,
        nll64,
        il,
        out,
        combine_lanes,
        n,
        c,
        states,
        BLOCK=256,
    )
    return out

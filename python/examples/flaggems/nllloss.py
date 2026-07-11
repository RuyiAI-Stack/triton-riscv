import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0).to(tl.int32)
    ignore_mask = (
        (tgt != ignore_index) & (tgt >= 0) & (tgt < C) & mask_n
    )

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(
            tl.float32
        )

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.store(out_ptr, total_out)
        tl.store(out_ptr + 1, total_wgt)
        tl.store(out_ptr + 3, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.store(out_ptr, total_out)


@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N * C
    offsets_n = offsets // C
    offsets_c = offsets % C

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask, other=0).to(tl.int32)
    ignore_mask = (
        (tgt != ignore_index) & (tgt >= 0) & (tgt < C) & mask
    )

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(
            tl.float32
        )

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    selected = ignore_mask & (offsets_c == tgt)
    inp_grad = tl.where(selected, -1 * out_grad * wgt_tgt / total_w, 0)
    tl.store(inp_grad_ptr + offsets, inp_grad, mask=mask)


@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0).to(tl.int32)
    ignore_mask = (
        (tgt != ignore_index) & (tgt >= 0) & (tgt < C) & mask_block
    )

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(
            tl.float32
        )

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        out_ptrs = out_ptr + offset_n * D + offset_d
        tl.store(out_ptrs, out, mask=mask_block)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.store(out_ptr, total_out)
        tl.store(out_ptr + 1, total_wgt)
        tl.store(out_ptr + 3, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.store(out_ptr, total_out)


@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    inp_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    FUSE_LOG_SOFTMAX: tl.constexpr = True,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0).to(tl.int32)
    ignore_mask = (
        (tgt != ignore_index) & (tgt >= 0) & (tgt < C) & mask_block
    )

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(
            tl.float32
        )

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(
            tl.float32
        )
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    if FUSE_LOG_SOFTMAX:
        for channel in range(0, C):
            offsets = offset_n * C * D + channel * D + offset_d
            log_probability = tl.load(
                inp_ptr + offsets, mask=mask_block, other=0.0
            ).to(tl.float32)
            channel_grad = inp_grad * (
                (tgt == channel).to(tl.float32) - tl.exp(log_probability)
            )
            tl.store(inp_grad_ptr + offsets, channel_grad, mask=mask_block)
    else:
        inp_grad_ptrs = inp_grad_ptr + offset_n * C * D + tgt * D + offset_d
        tl.store(inp_grad_ptrs, inp_grad, mask=mask_block)


def nll_loss_forward(
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
):
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    if reduction == 0:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = torch.zeros(
            [4],
            dtype=torch.float32,
            device=self.device,
        )
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    BLOCK_N = 128 if reduction == 0 else triton.next_power_of_2(N)
    grid = (triton.cdiv(N, BLOCK_N),)
    nll_loss_forward_kernel[grid](
        self,
        target,
        weight,
        out,
        ignore_index,
        N,
        C,
        reduction=reduction,
        BLOCK_N=BLOCK_N,
    )

    if reduction == 0:
        output = out
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = out.to(self.dtype)
        output = out[3]
        total_weight = out[1]
    else:
        output = out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    BLOCK_N = 128
    grid = (triton.cdiv(N * C, BLOCK_N),)

    nll_loss_backward_kernel[grid](
        grad_output,
        target,
        weight,
        grad_input,
        ignore_index,
        total_weight,
        N,
        C,
        reduction=reduction,
        BLOCK_N=BLOCK_N,
    )

    return grad_input


def nll_loss2d_forward(
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
):
    assert self.ndim == 4, "Invalid input ndim"
    shape = list(target.shape)
    N, C, D1, D2 = self.shape
    assert shape == [N, D1, D2], "Invalid target size"
    D = D1 * D2
    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    if reduction == 0:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = torch.zeros(
            [4],
            dtype=torch.float32,
            device=self.device,
        )
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    BLOCK_ND = (
        128 if reduction == 0 else triton.next_power_of_2(N * D)
    )
    grid = (triton.cdiv(N * D, BLOCK_ND),)
    nll_loss2d_forward_kernel[grid](
        self,
        target,
        weight,
        out,
        ignore_index,
        N,
        C,
        D,
        reduction=reduction,
        BLOCK_ND=BLOCK_ND,
    )

    if reduction == 0:
        output = out
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = out.to(self.dtype)
        output = out[3]
        total_weight = out[1]
    else:
        output = out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss2d_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    N, C, D1, D2 = self.shape
    D = D1 * D2
    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    BLOCK_ND = 128
    grid = (triton.cdiv(N * D, BLOCK_ND),)

    nll_loss2d_backward_kernel[grid](
        grad_output,
        target,
        weight,
        grad_input,
        self,
        ignore_index,
        total_weight,
        N,
        C,
        D,
        reduction=reduction,
        FUSE_LOG_SOFTMAX=True,
        BLOCK_ND=BLOCK_ND,
    )

    return grad_input


def nllloss(input, target, weight=None, reduction=1, ignore_index=-100):
    if input.ndim != 4:
        out, _ = nll_loss_forward(
            input, target, weight, reduction, ignore_index
        )
    else:
        out, _ = nll_loss2d_forward(
            input, target, weight, reduction, ignore_index
        )
    return out

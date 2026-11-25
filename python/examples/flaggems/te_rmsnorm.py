# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Ported from FlagOpen/FlagGems 599c5ba05c86, src/flag_gems/ops/te_rmsnorm.py.

"""
RMSNorm forward and backward implementation aligned with TransformerEngine.

TransformerEngine rmsnorm_fwd signature:
    rmsnorm_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)

TransformerEngine rmsnorm_bwd signature:
    rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)

Forward supports:
    - zero_centered_gamma: If true, applies (weight + 1) instead of weight
    - Pre-allocated output tensor (ln_out)
    - Output dtype conversion (otype)
    - Returns: (output, None, rsigma) where rsigma = 1/sqrt(mean(x^2) + eps)

Backward returns:
    (dx, dgamma) where:
        - dx: gradient w.r.t. input, shape same as x
        - dgamma: gradient w.r.t. weight, shape same as gamma
"""

from ._cpu_port_utils import fixed_config, get_tuned_config

import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["eps"])
def rmsnorm_fwd_kernel(
    out_ptr,
    rsigma_ptr,
    in_ptr,
    weight_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    N,
    eps,
    zero_centered_gamma: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward kernel for small N (fits in single block)."""
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty
    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + cols * x_stride_c, mask=mask, other=0.0).to(cdtype)
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    rsigma = 1.0 / tl.sqrt(var + eps)
    x_norm = x * rsigma
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)
    if zero_centered_gamma:
        w = w + 1.0
    y = (x_norm * w).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(rsigma_ptr + pid, rsigma)


@triton.jit(do_not_specialize=["eps"])
def rmsnorm_fwd_loop_kernel(
    out_ptr,
    rsigma_ptr,
    in_ptr,
    weight_ptr,
    N,
    eps,
    zero_centered_gamma: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """RMSNorm forward kernel for large N (requires loop)."""
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty
    pid = tl.program_id(0)
    in_ptr += pid * N
    out_ptr += pid * N
    sum_sq = tl.zeros([TILE_N], dtype=cdtype)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(cdtype)
        sum_sq += x * x
    var = tl.sum(sum_sq, axis=0) / N
    rsigma = 1.0 / tl.sqrt(var + eps)
    tl.store(rsigma_ptr + pid, rsigma)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)
        if zero_centered_gamma:
            w = w + 1.0
        x_norm = x * rsigma
        y = (x_norm * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + cols, y, mask=mask)


def te_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor = None,
    quantizer=None,
    otype: torch.dtype = None,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """
    RMSNorm forward pass aligned with TransformerEngine's rmsnorm_fwd.

    Args:
        input: Input tensor of shape (..., N)
        weight: Weight tensor (gamma) of shape (N,)
        eps: Epsilon for numerical stability
        ln_out: Pre-allocated output tensor (optional)
        quantizer: FP8 quantizer (currently not supported, placeholder for API compatibility)
        otype: Output dtype (optional, defaults to input dtype)
        sm_margin: SM margin (currently ignored, placeholder for API compatibility)
        zero_centered_gamma: If True, use (weight + 1) instead of weight

    Returns:
        Tuple of (output, None, rsigma):
            - output: Normalized tensor, same shape as input
            - None: Placeholder for amax (FP8 compatibility)
            - rsigma: 1/sqrt(mean(x^2) + eps), shape (...,) without last dim
    """
    logger.debug("GEMS TE_RMSNORM_FWD")
    if quantizer is not None:
        logger.warning("FP8 quantizer is not yet supported, ignoring")
    original_shape = input.shape
    N = original_shape[-1]
    input_2d = input.view(-1, N)
    M = input_2d.shape[0]
    assert weight.shape[0] == N, (
        f"Weight shape {weight.shape} doesn't match input last dim {N}"
    )
    if otype is None:
        otype = input.dtype
    if ln_out is not None:
        out = ln_out.view(-1, N)
        assert out.shape == (M, N), f"ln_out shape mismatch: {ln_out.shape}"
    else:
        out = torch.empty((M, N), dtype=otype, device=input.device)
    rsigma = torch.empty(M, dtype=torch.float32, device=input.device)
    x_stride_r, x_stride_c = input_2d.stride()
    y_stride_r, y_stride_c = out.stride()
    MAX_FUSED_SIZE = 65536 // input.element_size()
    if N <= MAX_FUSED_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(N)
        rmsnorm_fwd_kernel[M,](
            out,
            rsigma,
            input_2d,
            weight,
            y_stride_r,
            y_stride_c,
            x_stride_r,
            x_stride_c,
            N,
            eps,
            zero_centered_gamma,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        TILE_N = 1024
        rmsnorm_fwd_loop_kernel[M,](
            out, rsigma, input_2d, weight, N, eps, zero_centered_gamma, TILE_N=TILE_N
        )
    out = out.view(*original_shape[:-1], N)
    if ln_out is not None:
        out = ln_out
    return (out, None, rsigma)


@fixed_config(configs=get_tuned_config("rmsnorm_bwd_dx"), key=["M", "N"])
@triton.jit
def rmsnorm_bwd_dx_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    weight_ptr,
    rsigma_ptr,
    M,
    N,
    zero_centered_gamma: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute dx for RMSNorm backward using 2D tiling.
    Each program handles BLOCK_ROW_SIZE rows.
    """
    pid = tl.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty
    dz_ptr = dz_ptr + pid * N
    x_ptr = x_ptr + pid * N
    dx_ptr = dx_ptr + pid * N
    rsigma = tl.load(rsigma_ptr + pid, mask=row_mask).to(cdtype)
    c1_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)
        if zero_centered_gamma:
            w = w + 1.0
        x_hat = x * rsigma
        c1_acc += tl.where(mask, x_hat * dz * w, 0.0)
    c1 = tl.sum(c1_acc, axis=1)[:, None] / N
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)
        if zero_centered_gamma:
            w = w + 1.0
        x_hat = x * rsigma
        dx = rsigma * (dz * w - x_hat * c1)
        tl.store(dx_ptr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@fixed_config(configs=get_tuned_config("rmsnorm_bwd_dgamma"), key=["M", "N"])
@triton.jit
def rmsnorm_bwd_dgamma_kernel(
    dgamma_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute dgamma for RMSNorm backward using 2D tiling.
    Simple version: each program handles BLOCK_COL_SIZE columns, iterating over all rows.
    """
    pid = tl.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty
    dz_col_ptr = dz_ptr + pid[None, :]
    x_col_ptr = x_ptr + pid[None, :]
    dgamma_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]
        x = tl.load(x_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        rsigma = tl.load(rsigma_ptr + rows, mask=row_mask).to(cdtype)
        x_hat = x * rsigma
        dgamma_acc += tl.where(mask, dz * x_hat, 0.0)
    dgamma = tl.sum(dgamma_acc, axis=0)
    tl.store(dgamma_ptr + pid, dgamma.to(dgamma_ptr.dtype.element_ty), mask=col_mask)


@fixed_config(configs=get_tuned_config("rmsnorm_bwd_dgamma"), key=["M", "N"])
@triton.jit
def rmsnorm_bwd_dgamma_parallel_kernel(
    dgamma_partial_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    stride_partial,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute partial dgamma using 2D grid parallelization.
    Grid: (num_col_groups, num_row_groups)
    Each program handles BLOCK_COL_SIZE columns and iterates over its assigned row range.
    """
    col_group_id = tl.program_id(0)
    row_group_id = tl.program_id(1)
    col_start = col_group_id * BLOCK_COL_SIZE
    cols = col_start + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty
    dz_col_ptr = dz_ptr + cols[None, :]
    x_col_ptr = x_ptr + cols[None, :]
    rows_per_group = tl.cdiv(M, tl.num_programs(1))
    row_start = row_group_id * rows_per_group
    row_end = tl.minimum(row_start + rows_per_group, M)
    dgamma_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)
    for off in range(row_start, row_end, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < row_end
        mask = row_mask & col_mask[None, :]
        x = tl.load(x_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        rsigma = tl.load(rsigma_ptr + rows, mask=row_mask).to(cdtype)
        x_hat = x * rsigma
        dgamma_acc += tl.where(mask, dz * x_hat, 0.0)
    dgamma_partial = tl.sum(dgamma_acc, axis=0)
    tl.store(
        dgamma_partial_ptr + row_group_id * stride_partial + cols,
        dgamma_partial.to(dgamma_partial_ptr.dtype.element_ty),
        mask=col_mask,
    )


@triton.jit
def rmsnorm_bwd_dgamma_reduce_kernel(
    dgamma_ptr,
    dgamma_partial_ptr,
    num_row_groups,
    N,
    stride_partial,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Reduce partial dgamma across all row groups.
    """
    pid = tl.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dgamma_acc = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    for row_group in range(num_row_groups):
        partial = tl.load(
            dgamma_partial_ptr + row_group * stride_partial + pid,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        dgamma_acc += partial
    tl.store(
        dgamma_ptr + pid, dgamma_acc.to(dgamma_ptr.dtype.element_ty), mask=col_mask
    )


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """
    RMSNorm backward pass.

    Args:
        dz: gradient of output, shape (*, N)
        x: input tensor from forward pass, shape (*, N)
        rsigma: 1/sqrt(variance + eps) from forward pass, shape (*,)
        gamma: weight tensor, shape (N,)
        sm_margin: SM margin (unused, for API compatibility)
        zero_centered_gamma: if True, gamma is centered around 0

    Returns:
        dx: gradient w.r.t. input, shape (*, N)
        dgamma: gradient w.r.t. weight, shape (N,)
    """
    logger.debug("GEMS TE_RMSNORM_BWD")
    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.view(-1, N)
    dz_2d = dz.view(-1, N)
    M = x_2d.shape[0]
    x_2d = x_2d.contiguous()
    dz_2d = dz_2d.contiguous()
    rsigma = rsigma.contiguous()
    dx = torch.empty_like(x_2d)
    dgamma = torch.zeros(N, dtype=gamma.dtype, device=gamma.device)

    def grid_dx(meta):
        return (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]),)

    rmsnorm_bwd_dx_kernel[grid_dx](
        dx, dz_2d, x_2d, gamma, rsigma, M, N, zero_centered_gamma
    )
    M_THRESHOLD = 2048
    if M <= M_THRESHOLD:

        def grid_dgamma(meta):
            return (triton.cdiv(N, meta["BLOCK_COL_SIZE"]),)

        rmsnorm_bwd_dgamma_kernel[grid_dgamma](dgamma, dz_2d, x_2d, rsigma, M, N)
    else:
        num_row_groups = min(triton.cdiv(M, 128), 64)
        num_row_groups = max(num_row_groups, 1)
        dgamma_partial = torch.zeros(
            (num_row_groups, N), dtype=torch.float32, device=x.device
        )
        stride_partial = N

        def grid_dgamma(meta):
            return (
                triton.cdiv(N, meta["BLOCK_COL_SIZE"]),
                num_row_groups,
            )

        rmsnorm_bwd_dgamma_parallel_kernel[grid_dgamma](
            dgamma_partial, dz_2d, x_2d, rsigma, M, N, stride_partial
        )
        REDUCE_BLOCK_SIZE = 256
        grid_reduce = (triton.cdiv(N, REDUCE_BLOCK_SIZE),)
        rmsnorm_bwd_dgamma_reduce_kernel[grid_reduce](
            dgamma,
            dgamma_partial,
            num_row_groups,
            N,
            stride_partial,
            BLOCK_COL_SIZE=REDUCE_BLOCK_SIZE,
        )
    dx = dx.view(*original_shape)
    return (dx, dgamma)

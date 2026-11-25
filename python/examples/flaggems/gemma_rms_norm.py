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

# Ported from FlagOpen/FlagGems 599c5ba05c86, src/flag_gems/ops/gemma_rms_norm.py.

import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["eps"])
def gemma_rms_norm_kernel(
    out_ptr,
    INV_RMS,
    in_ptr,
    w_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty
    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    y = (x * rrms * (1.0 + w)).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


def gemma_rms_norm(input, weight, eps=1e-6):
    """Normalize the final dimension and apply Gemma's ``1 + weight`` scale."""
    if input.ndim == 0 or input.shape[-1] != weight.numel():
        raise ValueError("weight must match the final input dimension")
    x = input.contiguous()
    n = x.shape[-1]
    out = torch.empty_like(x)
    rows = x.numel() // n if n else 0
    inv = torch.empty(rows, dtype=torch.float32, device=x.device)
    if rows:
        gemma_rms_norm_kernel[(rows,)](
            out,
            inv,
            x,
            weight.contiguous(),
            n,
            1,
            n,
            1,
            n,
            eps,
            BLOCK_SIZE=triton.next_power_of_2(n),
        )
    return out

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

# Ported from FlagOpen/FlagGems 599c5ba05c86, src/flag_gems/ops/upsample_linear1d_backward.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_backward(
    G,
    OUT,
    N,
    IW: tl.constexpr,
    OW: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    o = i % OW
    real = o * SCALE if ALIGN else tl.maximum((o + 0.5) * SCALE - 0.5, 0.0)
    left = tl.minimum(tl.floor(real).to(tl.int64), IW - 1)
    right = tl.minimum(left + 1, IW - 1)
    frac = real - left
    base = (i // OW) * IW
    value = tl.load(G + i, mask, other=0)
    tl.atomic_add(OUT + base + left, value * (1 - frac), mask)
    tl.atomic_add(OUT + base + right, value * frac, mask)


def upsample_linear1d_backward(
    grad_output, output_size, input_size, align_corners, scale_factors=None
):
    iw = input_size[-1]
    ow = grad_output.shape[-1]
    if output_size is not None and tuple(output_size) != (ow,):
        raise ValueError("gradient size mismatch")
    supplied = (
        scale_factors[0] if isinstance(scale_factors, (tuple, list)) else scale_factors
    )
    scale = (
        (iw - 1) / (ow - 1)
        if align_corners and ow > 1
        else (
            0.0
            if align_corners
            else 1.0 / supplied
            if supplied is not None and supplied > 0
            else iw / ow
        )
    )
    dtype = (
        torch.float32
        if grad_output.dtype in (torch.float16, torch.bfloat16)
        else grad_output.dtype
    )
    out = torch.zeros(input_size, dtype=dtype, device=grad_output.device)
    n = grad_output.numel()
    if n:
        _linear_backward[(triton.cdiv(n, 256),)](
            grad_output.contiguous(),
            out,
            n,
            iw,
            ow,
            scale,
            bool(align_corners),
            BLOCK=256,
        )
    return out.to(grad_output.dtype)

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

# Ported from FlagOpen/FlagGems 599c5ba05c86, src/flag_gems/ops/upsample_lanczos2d_aa.py.

from typing import Optional, Sequence, Tuple
import torch
from ._resize_aa_cpu import resize_aa


def _upsample_lanczos2d_aa(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    return resize_aa(
        input, output_size, align_corners, scales_h, scales_w, lanczos=True
    )


def _upsample_lanczos2d_aa_out(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor on {input.device}, but got {out.device}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype}"
        )
    result = _upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _upsample_lanczos2d_aa_vec(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    if (output_size is None) == (scale_factors is None):
        raise RuntimeError("Must specify exactly one of output_size and scale_factors")
    if output_size is not None:
        return _upsample_lanczos2d_aa(input, tuple(output_size), align_corners)
    if len(scale_factors) != 2:
        raise RuntimeError("scale_factors must have two elements")
    output_size = (
        int(input.shape[-2] * scale_factors[0]),
        int(input.shape[-1] * scale_factors[1]),
    )
    return _upsample_lanczos2d_aa(
        input,
        output_size,
        align_corners,
        float(scale_factors[0]),
        float(scale_factors[1]),
    )

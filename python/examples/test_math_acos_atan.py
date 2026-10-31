"""Regression tests for the Triton 3.8.0 frontend compatibility patch.

The 0003-triton-compatibility patch adds the ``acos``/``atan``/``atan2``
builders to the TritonOpBuilder bindings and the corresponding builtins to
``triton.language.math``. FlagGems exercises them only in a suite that the
common run excludes, so this file keeps a direct end-to-end check in the
common example suite.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def math_acos_atan_kernel(
    x_ptr,
    y_ptr,
    acos_out,
    atan_out,
    atan2_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(acos_out + offsets, tl.math.acos(x), mask=mask)
    tl.store(atan_out + offsets, tl.math.atan(x), mask=mask)
    tl.store(atan2_out + offsets, tl.math.atan2(x, y), mask=mask)


def _run_case(dtype):
    n = 1024
    # Keep |x| < 0.9: acos has an unbounded derivative at +-1, which would
    # make tight fp32 tolerances meaningless.
    x = (torch.rand(n, dtype=torch.float32) * 1.8 - 0.9).to(dtype)
    y = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0).to(dtype)
    acos_out = torch.empty_like(x)
    atan_out = torch.empty_like(x)
    atan2_out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    math_acos_atan_kernel[grid](
        x, y, acos_out, atan_out, atan2_out, n, BLOCK_SIZE=256
    )
    torch.testing.assert_close(acos_out, torch.acos(x), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(atan_out, torch.atan(x), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        atan2_out, torch.atan2(x, y), rtol=1e-5, atol=1e-6
    )


def test_math_acos_atan_atan2_fp32():
    _run_case(torch.float32)


def test_math_acos_atan_atan2_fp64():
    _run_case(torch.float64)

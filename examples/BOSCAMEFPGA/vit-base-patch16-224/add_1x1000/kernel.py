"""Standalone Triton source for add_1x1000."""

import triton
import triton.language as tl


@triton.jit
def add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    tl.store(
        Out + index,
        tl.load(X + index, valid, other=0.0)
        + tl.load(Y + index, valid, other=0.0),
        valid,
    )

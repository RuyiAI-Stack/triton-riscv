"""Standalone Triton source for softmax_12x197x197."""

import triton
import triton.language as tl


@triton.jit
def softmax(
    X, Out, ROWS: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < LENGTH
    x = tl.load(X + row * LENGTH + column, valid, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponential = tl.exp(x - maximum)
    total = tl.sum(exponential, axis=0)
    tl.store(Out + row * LENGTH + column, exponential / total, valid)

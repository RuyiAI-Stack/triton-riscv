"""Standalone Triton source for patch_embed_196x768."""

import triton
import triton.language as tl


@triton.jit
def patch_embed(
    X,
    Weight,
    Bias,
    Out,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    PATCH: tl.constexpr,
    HIDDEN: tl.constexpr,
    GRID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize the convolution patches as a row-major im2col matrix.

    The projection is a ``[196, 768] @ [768, 768]^T`` operation.  The old
    kernel performed that reduction while loading each output channel, which
    made the weight and result accesses strided by the opposite matrix
    dimension.  This stage only walks the input image and writes one
    contiguous ``[patch, channel*PATCH*PATCH + dy*PATCH + dx]`` row.  The
    graph-facing C adapter quantizes activations and weights, calls signed i8
    matmul, dequantizes, adds bias and converts the result to NCHW.

    ``Weight`` and ``Bias`` remain in the signature for ABI compatibility with
    the graph replacement.  They are intentionally unused here.
    """
    patch = tl.program_id(0)
    row = patch // GRID
    col = patch % GRID
    for channel in range(CHANNELS):
        for dy in range(PATCH):
            for dx in range(PATCH):
                input_index = (
                    channel * HEIGHT * WIDTH
                    + (row * PATCH + dy) * WIDTH
                    + col * PATCH
                    + dx
                )
                column = channel * PATCH * PATCH + dy * PATCH + dx
                # Every row is contiguous in the destination even though the
                # source is traversed through the NCHW image layout.
                tl.store(
                    Out + patch * HIDDEN + column, tl.load(X + input_index)
                )

"""Standalone Triton kernel for FPGA case 'embedding_128x768' (model 'electra-base-discriminator').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/electra-base-discriminator/embedding_128x768
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def embedding(
    IDs,
    Types,
    Word,
    Position,
    Segment,
    Out,
    SEQ: tl.constexpr,
    WIDTH: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(IDs + row)
    segment = tl.load(Types + row)
    # IDs must be in [0,VOCAB), Types in [0,2); the host tokenizer validates
    # them. Only the static channel tail participates in the DMA contract.
    good = col < WIDTH
    word = tl.load(Word + token * WIDTH + col, good, other=0.0)
    position = tl.load(Position + row * WIDTH + col, col < WIDTH, other=0.0)
    typ = tl.load(Segment + segment * WIDTH + col, good, other=0.0)
    tl.store(Out + row * WIDTH + col, (word + typ) + position, col < WIDTH)


KERNEL_CONFIG = {
    "family": "embedding",
    "kernel": "embedding",
    "arguments": [
        {"name": "IDs", "rank": 1, "dtype": "i32"},
        {"name": "Types", "rank": 1, "dtype": "i32"},
        {"name": "Word", "rank": 2, "dtype": "f32"},
        {"name": "Position", "rank": 2, "dtype": "f32"},
        {"name": "Segment", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "argument_shapes": [[128], [128], [30522, 768], [512, 768], [2, 768], [128, 768]],
    "shape": [128, 768],
    "constexprs": {"SEQ": 128, "WIDTH": 768, "VOCAB": 30522, "BLOCK": 1024},
    "grid": [128, 1, 1],
    "roles": ["model"],
    "metadata": {"model": "google/electra-base-discriminator"},
}

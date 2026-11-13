"""Standalone Triton kernel for FPGA case 'embedding_lookup_128x384' (model 'gte-small').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/gte-small/embedding_lookup_128x384
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def embedding_lookup(
    Ids,
    Table,
    Out,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(Ids + row)
    valid = col < WIDTH
    value = tl.load(Table + token * WIDTH + col, valid, other=0.0)
    tl.store(Out + row * WIDTH + col, value, col < WIDTH)


KERNEL_CONFIG = {
    "family": "embedding_lookup",
    "kernel": "embedding_lookup",
    "arguments": [
        {"name": "Ids", "rank": 1, "dtype": "i32"},
        {"name": "Table", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [128],
    "argument_shapes": [[128], [30522, 384], [128, 384]],
    "constexprs": {"ROWS": 128, "WIDTH": 384, "VOCAB": 30522, "BLOCK": 512},
    "grid": [128, 1, 1],
    "roles": ["word_embeddings"],
    "outputs": [2],
    "checked_elements": 49152,
    "metadata": {
        "model": "thenlper/gte-small",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}

"""``_standard_gamma`` port (Gamma sampler, Marsaglia-Tsang + boost).

Port notes (triton-riscv, vs the upstream flag_gems file):
- ``philox_backend_seed_offset`` comes from the repo's ``rand`` module
  (cross-file reuse precedent: concatenate -> .cat); the standard
  uint32->uniform transform is inlined.
- @libentry and the torch device context manager are stripped; the lambda
  grid is a plain tuple. Kernel arithmetic is upstream's verbatim.
"""

import logging

import torch
import triton
import triton.language as tl

from .exponential_ import uint_to_uniform_float
from .rand import philox_backend_seed_offset

logger = logging.getLogger(__name__)

# Each lane draws up to MAX_ITERS Marsaglia-Tsang proposals; every proposal
# consumes one philox call (4 uint32 -> 4 uniforms) and one extra call is
# reserved for the alpha<1 boost, so a lane reserves (MAX_ITERS + 1) * 4
# random words. Marsaglia-Tsang accepts with probability >~0.95 per proposal,
# so 32 proposals make an unfilled lane astronomically unlikely.
MAX_ITERS = 32
PHILOX_STRIDE = (MAX_ITERS + 1) * 4
BLOCK_SIZE = 1024


@triton.jit
def _boost_uniform(seed, base_counter, alpha_f32, MAX_ITERS: tl.constexpr):
    # For alpha < 1 use the boost trick: if X ~ Gamma(alpha+1) then
    # X * U^(1/alpha) ~ Gamma(alpha). Draw the single boost uniform from a
    # dedicated counter slot (past every proposal slot) so it never correlates
    # with the proposal randoms.
    counter = base_counter + MAX_ITERS * 4
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = c0 * 0
    r0, _, _, _ = tl.philox(seed, c0, c1, z, z)
    u = tl.maximum(uint_to_uniform_float(r0), 1e-7)
    return tl.where(alpha_f32 < 1.0, tl.exp(tl.log(u) / alpha_f32), 1.0)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def standard_gamma_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    philox_seed,
    philox_offset,
    TINY,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    alpha = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    alpha_f32 = alpha.to(tl.float32)

    # alpha < 1 is handled through Gamma(alpha + 1) plus the boost factor.
    alpha_adj = tl.where(alpha_f32 < 1.0, alpha_f32 + 1.0, alpha_f32)
    d = alpha_adj - 0.3333333333333333
    c = 1.0 / tl.sqrt(9.0 * d)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    base_counter = philox_offset + offsets.to(tl.int64) * STRIDE

    boost = _boost_uniform(philox_seed, base_counter, alpha_f32, MAX_ITERS)

    # Marsaglia-Tsang acceptance-rejection. Each iteration draws fresh randoms
    # for the still-unaccepted lanes; accepted lanes freeze their sample. This
    # avoids the bias of substituting a fallback value on rejection.
    accepted = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    result_f32 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for it in range(0, MAX_ITERS):
        counter = base_counter + it * 4
        c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        z0 = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z0, z0)
        u2 = tl.maximum(uint_to_uniform_float(r1), 1e-7)
        u3 = uint_to_uniform_float(r2)
        u4 = tl.maximum(uint_to_uniform_float(r3), 1e-7)

        # Standard normal via Box-Muller.
        z = tl.sqrt(-2.0 * tl.log(u2)) * tl.cos(2.0 * 3.141592653589793 * u3)
        v = 1.0 + c * z
        v3 = v * v * v

        accept = (v > 0.0) & (tl.log(u4) < (0.5 * z * z + d - d * v3 + d * tl.log(v3)))
        take = accept & (accepted == 0)
        result_f32 = tl.where(take, d * v3, result_f32)
        accepted = accepted | accept

    result_f32 = result_f32 * boost
    # ATen clamps the draw to the smallest positive value of the output dtype
    # so tiny-alpha samples never underflow to exactly zero.
    result_f32 = tl.maximum(result_f32, TINY)

    result = result_f32.to(alpha.dtype)
    tl.store(output_ptr + offsets, result, mask=mask)


def standard_gamma(input, generator=None):
    logger.debug("GEMS _STANDARD_GAMMA")

    if not torch.is_floating_point(input):
        raise RuntimeError(f"\"standard_gamma\" not implemented for '{input.dtype}'")
    # The Marsaglia-Tsang kernel samples entirely in float32, so it cannot
    # preserve the float64 smallest-positive value (~2.2e-308) that ATen
    # clamps to for tiny alpha -- passing it as an fp32 scalar underflows to
    # zero. Reject float64 explicitly rather than silently degrading
    # precision.
    if input.dtype == torch.float64:
        raise RuntimeError("flag_gems standard_gamma does not support float64 inputs")

    input = input.contiguous()
    output = torch.empty_like(input)

    n_elements = input.numel()
    if n_elements == 0:
        return output

    tiny = torch.finfo(input.dtype).tiny

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    increment = triton.cdiv(n_elements * PHILOX_STRIDE, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    standard_gamma_kernel[grid](
        input,
        output,
        n_elements,
        philox_seed,
        philox_offset,
        tiny,
        BLOCK_SIZE=BLOCK_SIZE,
        STRIDE=PHILOX_STRIDE,
        MAX_ITERS=MAX_ITERS,
    )

    return output

"""CPU launch configuration for the B12--B14 standalone examples.

No CUDA runtime or FlagGems dispatcher is imported. Configurations are fixed,
so kernels with in-place outputs are launched exactly once. Output resets that
were part of the upstream autotuner contract are retained.
"""

import enum
import os
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


class _FixedKernel:
    def __init__(self, fn, cfg, reset_to_zero):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.cfg = cfg
        self.reset_to_zero = tuple(reset_to_zero)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(self, *args, **kwargs):
        kwargs.setdefault("warmup", False)
        for name, value in self.cfg.kwargs.items():
            kwargs.setdefault(name, value)
        arguments = dict(zip(self.arg_names, args)) | kwargs
        if not kwargs.get("warmup", False):
            for name in self.reset_to_zero:
                arguments[name].zero_()
            if self.cfg.pre_hook is not None:
                self.cfg.pre_hook(arguments)
        return self.fn.run(*args, **kwargs)


def fixed_config(configs, key=(), reset_to_zero=None, **options):
    if not configs:
        raise ValueError("A concrete CPU launch configuration is required")
    # restore_value only protects inputs during repeated autotuning trials.
    # There are no trials here: the selected kernel executes once.
    if options.get("pre_hook") or options.get("post_hook"):
        raise ValueError(
            "This operator needs an explicit CPU adaptation of its tuning hooks"
        )
    cfg = configs[0]
    return lambda fn: _FixedKernel(fn, cfg, reset_to_zero or ())


def get_tuned_config(name):
    return [triton.Config(_CPU_CONFIGS[name], num_warps=1)]


def get_heuristic_config(name):
    if name == "gather_block_quantized":
        return {
            "BLOCK_SIZE_N": lambda a: min(256, triton.next_power_of_2(max(1, a["N"])))
        }
    if name == "upsample_nearest3d":
        return {
            "SAME_D": lambda a: a["OD"] == a["ID"],
            "SAME_H": lambda a: a["OH"] == a["IH"],
            "SAME_W": lambda a: a["OW"] == a["IW"],
            "USE_INT32_IDX": lambda a: (
                a["N"] * a["C"] * a["OD"] * a["OH"] * a["OW"] <= 2**31 - 1
            ),
        }
    if name != "batch_norm":
        raise KeyError(name)
    return {
        "BLOCK_M": lambda args: min(32, triton.next_power_of_2(args["batch_dim"])),
        "BLOCK_N": lambda args: min(256, triton.next_power_of_2(args["spatial_dim"])),
    }


def get_device_properties(index=None):
    # Used only for scheduling persistent programs; CPU cores replace GPU SMs.
    return SimpleNamespace(multi_processor_count=os.cpu_count() or 1)


def get_device_capability():
    # CUDA half-precision atomic fast paths do not apply to the CPU backend.
    return (0, 0)


def bracket_next_power_of_2(n, lower, upper):
    return min(max(triton.next_power_of_2(n), lower), upper)


def c_contiguous_stride(shape):
    strides, stride = [], 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(size, 1)
    return tuple(reversed(strides))


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dims = [d % inp.ndim for d in dims]
    outer = [d for d in range(inp.ndim) if d not in dims]
    inner = sorted(dims, key=lambda d: inp.stride(d), reverse=True)
    return inp.permute(outer + inner).contiguous()


class MemOverlap(enum.Enum):
    No = 0
    Yes = 1
    TooHard = 2


def has_internal_overlapping(x):
    return MemOverlap(int(torch._debug_has_internal_overlap(x)))


@triton.jit
def round_even(x):
    low = tl.floor(x)
    frac = x - low
    odd = (low.to(tl.int64) & 1) != 0
    rounded = low + ((frac > 0.5) | ((frac == 0.5) & odd)).to(x.dtype)
    return tl.where(tl.abs(x) >= 4503599627370496.0, x, rounded)


_CPU_CONFIGS = {
    "batch_norm": {},
    "vector_norm": {"BLOCK_M": 1, "BLOCK_N": 1024},
    "naive_reduction": {"BLOCK_M": 1, "BLOCK_N": 1024},
    "trapezoid_kernel": {"BLOCK_M": 1, "BLOCK_N": 256},
    "rms_norm_loop": {"TILE_N": 1024},
    "rmsnorm_bwd_dx": {"BLOCK_ROW_SIZE": 1, "BLOCK_COL_SIZE": 256},
    "rmsnorm_bwd_dgamma": {"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 64},
    "cudnn_grid_sampler": {"BLOCK": 128},
    "grid_sampler_2d": {"BLOCK_SIZE": 256},
    "index_select_backward": {"BLOCK": 128},
    "upsample_bilinear2d": {"BLOCK_X": 512, "BLOCK_Y": 2},
    "upsample_bilinear2d_aa": {"BLOCK_X": 512, "BLOCK_Y": 2},
    "adaptive_max_pool2d": {"BLOCK_H": 16, "BLOCK_W": 16},
    "adaptive_max_pool2d_backward": {"BLOCK_SIZE": 256},
    "adaptive_max_pool3d": {"OUT_PER_BLOCK": 256, "CHAN_PER_BLOCK": 1},
    "adaptive_max_pool3d_backward": {"BLOCK_SIZE": 16},
    "fractional_max_pool2d_forward": {"BLOCK_H": 8, "BLOCK_W": 8},
    "fractional_max_pool2d_backward": {"BLOCK_H": 8, "BLOCK_W": 8},
    "im2col": {"BLOCK_M": 64, "BLOCK_N": 128},
    "max_pool1d_with_indices": {"BLOCK_L": 128},
    "max_unpool2d": {"BLOCK_H": 16, "BLOCK_W": 16},
    "quantized_max_pool1d": {"BLOCK_L": 64},
    "quantized_max_pool2d": {"BLOCK_H": 16, "BLOCK_W": 16},
    "quantized_max_pool3d": {"BLOCK_H": 16, "BLOCK_W": 16},
    "_adaptive_avg_pool2d_backward": {"BLOCK_H": 16, "BLOCK_W": 16},
    "conv2d_backward_weight": {"BLOCK_CI_HK_WK": 32, "BLOCK_CO": 32, "BLOCK_NO": 32},
}

# The kernels below are adapted from FlagOS Contributors, Apache-2.0.
# Upstream revision: 599c5ba05c86. See LICENSE.FlagGems.


@triton.jit
def get_dtype_max(dtype: tl.constexpr):
    """get a value which is greater that all other values of that dtype"""
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float("inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2 ** (width - 1) - 1
        return value
    if dtype_.is_int_unsigned():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2**width - 1
        return value


@triton.jit
def get_dtype_min(dtype):
    """get a value which is less that all other values of that dtype"""
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float("-inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = -1 * 2 ** (width - 1)
        return value
    if dtype_.is_int_unsigned():
        value: tl.constexpr = 0
        return value


@triton.jit
def _seg_scan_kernel(sorted_keys, gstart, slots, total, n, BLOCK: tl.constexpr):
    """Single-program walk over the sorted keys.

    Emits the start offset of every group into ``gstart`` (groups are
    contiguous runs of equal keys), the compact group id of every sorted
    position into ``slots``, and the number of groups into ``total``.
    """
    run = 0
    for start in range(0, n, BLOCK):
        p = start + tl.arange(0, BLOCK)
        m = p < n
        key = tl.load(sorted_keys + p, mask=m, other=-1)
        prev = tl.load(sorted_keys + p - 1, mask=m & (p > 0), other=-1)
        first = (p == 0) | (key != prev)
        f32 = first.to(tl.int32)
        cs = tl.cumsum(f32, 0)
        slot = run + cs - 1
        tl.store(gstart + slot, p, mask=first & m)
        tl.store(slots + p, slot, mask=m)
        run += tl.sum(f32, axis=0)
    tl.store(total, run)


@triton.jit
def pow_cpu(x, y):
    # Real power expressed in portable MLIR math operations.
    magnitude = tl.exp(tl.log(tl.abs(x)) * y)
    integral = y == tl.floor(y)
    odd = y - 2.0 * tl.floor(y * 0.5) != 0
    signed = tl.where((x < 0) & odd, -magnitude, magnitude)
    value = tl.where((x < 0) & ~integral, float("nan"), signed)
    return tl.where(y == 0, 1.0, value)

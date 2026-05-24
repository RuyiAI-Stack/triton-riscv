from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._is_all_true import _is_all_true
from ._safe_softmax import _safe_softmax
from ._upsample_nearest_exact1d import _upsample_nearest_exact1d
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .act_quant import act_quant_triton
from .add import add, add_
from .addcdiv import addcdiv, addcdiv_out
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from .addmv import addmv, addmv_out
from .addr import addr
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
from .amax import amax
from .aminmax import aminmax
from .angle import angle
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arcsinh import arcsinh, arcsinh_out
from .arcsinh_ import arcsinh_
from .arctanh_ import arctanh_
from .argmax import argmax
from .argmin import argmin
from .asinh import asinh, asinh_out
from .asinh_ import asinh_
from .assert_async import _assert_async
from .atan import atan, atan_
from .atan2 import atan2, atan2_out
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .avg_pool3d import avg_pool3d, avg_pool3d_backward
from .baddbmm import baddbmm, baddbmm_out
from .batch_norm import batch_norm, batch_norm_backward
from .bernoulli_ import bernoulli_
from .bincount import bincount
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_left_shift import bitwise_left_shift
from .bitwise_not import bitwise_not, bitwise_not_
from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .bitwise_right_shift import bitwise_right_shift
from .bmm import bmm, bmm_out
from .cat import cat, cat_out
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .clamp import (
    clamp,
    clamp_,
    clamp_min,
    clamp_min_,
    clamp_tensor,
    clamp_tensor_,
)
from .clip import clip, clip_
from .col2im import col2im
from .concatenate import concatenate
from .conj_physical import conj_physical
from .contiguous import contiguous
from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d
from .conv_depthwise2d import _conv_depthwise2d
from .copy import copy, copy_
from .copysign import copysign, copysign_out
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out
from .count_nonzero import count_nonzero
from .cummax import cummax
from .cummin import cummin
from .cumprod import cumprod, cumprod_
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diff import diff
from .digamma_ import digamma_
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from .dot import dot
from .dropout import dropout, dropout_backward
from .elu import elu, elu_, elu_backward
from .embedding import embedding, embedding_backward
from .embedding_dense_backward import embedding_dense_backward
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .expm1 import expm1, expm1_, expm1_out
from .exponential_ import exponential_
from .eye import eye
from .eye_m import eye_m
from .feature_dropout import feature_dropout, feature_dropout_
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .flip import flip
from .floor_ import floor_
from .fmin import fmin, fmin_out
from .fmod import fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from .fp8_matmul import fp8_matmul
from .full import full
from .full_like import full_like
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .get_paged_mqa_logits_metadata import get_paged_mqa_logits_metadata
from .get_scheduler_metadata import get_scheduler_metadata
from .glu import glu, glu_backward
from .greater import (
    greater,
    greater_out,
    greater_scalar,
    greater_scalar_out,
)
from .grid_sample import grid_sample
from .group_gemm import group_mm
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .hadamard_transform import hadamard_transform
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_
from .histc import histc
from .hstack import hstack
from .hypot import hypot, hypot_out
from .i0 import i0, i0_out
from .i0_ import i0_
from .index import index
from .index_add import index_add, index_add_
from .index_put import (
    _index_put_impl_,
    index_put,
    index_put_,
)
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .isneginf import isneginf, isneginf_out
from .kron import kron
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linspace import linspace
from .log import log
from .log1p_ import log1p_
from .log10 import log10, log10_, log10_out
from .log_sigmoid import log_sigmoid
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .logaddexp import logaddexp, logaddexp_out
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logit import logit, logit_out
from .logit_ import logit_
from .logspace import logspace
from .logsumexp import logsumexp
from .lt import lt, lt_scalar
from .margin_ranking_loss import margin_ranking_loss
from .masked_fill import masked_fill, masked_fill_
from .masked_scatter import masked_scatter, masked_scatter_
from .masked_select import masked_select
from .max import max, max_dim
from .max_pool2d_with_indices import (
    max_pool2d_backward,
    max_pool2d_with_indices,
)
from .max_pool3d_with_indices import (
    max_pool3d_backward,
    max_pool3d_with_indices,
)
from .maximum import maximum
from .mean import mean, mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, mm_out
from .mse_loss import mse_loss
from .mul import mul, mul_
from .multinomial import multinomial
from .mv import mv
from .nan_to_num import nan_to_num
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .new_full import new_full
from .nll_loss_nd import nll_loss_nd_backward, nll_loss_nd_forward
from .nllloss import (
    nll_loss2d_backward,
    nll_loss2d_forward,
    nll_loss_backward,
    nll_loss_forward,
)
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .pad import constant_pad_nd, pad
from .per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from .pixel_shuffle import pixel_shuffle
from .pixel_unshuffle import pixel_unshuffle, pixel_unshuffle_out
from .polar import polar
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prelu import prelu
from .prod import prod, prod_dim
from .quantile import quantile
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .reflection_pad1d import reflection_pad1d, reflection_pad1d_out
from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out
from .relu import relu, relu_
from .relu6 import relu6
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .replication_pad1d import replication_pad1d, replication_pad1d_out
from .replication_pad3d import replication_pad3d
from .resolve_conj import resolve_conj
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .roll import roll
from .round import round, round_, round_out
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub_scalar, rsub_tensor
from .scaled_softmax import scaled_softmax_backward, scaled_softmax_forward
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .scatter_reduce_ import scatter_reduce_
from .select_backward import select_backward
from .select_scatter import select_scatter
from .selu import selu
from .selu_ import selu_
from .sgn_ import sgn_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .signbit import signbit, signbit_out
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .sinh_ import sinh_
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .smooth_l1_loss import (
    smooth_l1_loss,
    smooth_l1_loss_backward,
    smooth_l1_loss_out,
)
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .softmax import (
    softmax,
    softmax_backward,
    softmax_backward_out,
    softmax_out,
)
from .softplus import softplus
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .sqrt import sqrt, sqrt_
from .square import square, square_, square_out
from .stack import stack
from .std import std
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .threshold import threshold, threshold_backward
from .tile import tile
from .to import to_copy
from .topk import topk
from .trace import trace
from .tril import tril, tril_, tril_out
from .triu import triu, triu_
from .unfold_backward import unfold_backward
from .uniform import uniform_
from .unique import _unique2
from .unique_consecutive import unique_consecutive
from .upsample_bicubic2d import upsample_bicubic2d
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest3d import upsample_nearest3d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vdot import vdot
from .vector_norm import vector_norm
from .vstack import vstack
from .w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
from .weightnorm import (
    weight_norm_interface,
    weight_norm_interface_backward,
)
from .where import (
    where_scalar_other,
    where_scalar_self,
    where_self,
    where_self_out,
)
from .zero import zero, zero_out
from .zeros import zero_, zeros
from .zeros_like import zeros_like


__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "ScaleDotProductAttention",
    "_assert_async",
    "_conv_depthwise2d",
    "_functional_sym_constrain_range_for_size",
    "_index_put_impl_",
    "_is_all_true",
    "_safe_softmax",
    "_unique2",
    "_upsample_bicubic2d_aa",
    "_upsample_bicubic2d_aa_backward",
    "_upsample_nearest_exact1d",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "act_quant_triton",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_out",
    "addcmul",
    "addcmul_out",
    "addmm",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv",
    "addmv_out",
    "addr",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "amax",
    "aminmax",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arctanh_",
    "argmax",
    "argmin",
    "asinh",
    "asinh_",
    "asinh_out",
    "atan",
    "atan2",
    "atan2_out",
    "atan_",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_out",
    "batch_norm",
    "batch_norm_backward",
    "bernoulli_",
    "bincount",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "bitwise_right_shift",
    "bmm",
    "bmm_out",
    "cat",
    "cat_out",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "col2im",
    "concatenate",
    "conj_physical",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "copy",
    "copy_",
    "copysign",
    "copysign_out",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "cummax",
    "cummin",
    "cumprod",
    "cumprod_",
    "cumsum",
    "cumsum_out",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "diff",
    "digamma_",
    "div_mode",
    "div_mode_",
    "dot",
    "dropout",
    "dropout_backward",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_backward",
    "embedding_dense_backward",
    "eq",
    "eq_scalar",
    "equal",
    "erf",
    "erf_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "exp_out",
    "expm1",
    "expm1_",
    "expm1_out",
    "exponential_",
    "eye",
    "eye_m",
    "feature_dropout",
    "feature_dropout_",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_opt_func",
    "flip",
    "floor_",
    "floor_divide",
    "floor_divide_",
    "fmin",
    "fmin_out",
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "fp8_matmul",
    "full",
    "full_like",
    "gather",
    "gather_backward",
    "gcd",
    "gcd_out",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "get_paged_mqa_logits_metadata",
    "get_scheduler_metadata",
    "glu",
    "glu_backward",
    "greater",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "grid_sample",
    "group_mm",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar",
    "hadamard_transform",
    "hardsigmoid",
    "hardsigmoid_out",
    "hardswish_",
    "histc",
    "hstack",
    "hypot",
    "hypot_out",
    "i0",
    "i0_",
    "i0_out",
    "index",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_select",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isneginf_out",
    "kron",
    "layer_norm",
    "layer_norm_backward",
    "le",
    "le_scalar",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_out",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linspace",
    "log",
    "log1p_",
    "log10",
    "log10_",
    "log10_out",
    "log_sigmoid",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "logaddexp",
    "logaddexp_out",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logit",
    "logit_",
    "logit_out",
    "logspace",
    "logsumexp",
    "lt",
    "lt_scalar",
    "margin_ranking_loss",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_",
    "masked_select",
    "max",
    "max_dim",
    "max_pool2d_backward",
    "max_pool2d_with_indices",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "maximum",
    "mean",
    "mean_dim",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "mm_out",
    "mse_loss",
    "mul",
    "mul_",
    "multinomial",
    "mv",
    "nan_to_num",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "new_full",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nll_loss_backward",
    "nll_loss_forward",
    "nll_loss_nd_backward",
    "nll_loss_nd_forward",
    "nonzero",
    "nonzero_numpy",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "one_hot",
    "ones",
    "ones_like",
    "pad",
    "per_token_group_quant_fp8",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pixel_unshuffle_out",
    "polar",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "prod",
    "prod_dim",
    "quantile",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "reflection_pad1d",
    "reflection_pad1d_out",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "relu",
    "relu6",
    "relu_",
    "remainder",
    "remainder_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "replication_pad1d",
    "replication_pad1d_out",
    "replication_pad3d",
    "resolve_conj",
    "resolve_neg",
    "rms_norm",
    "rms_norm_backward",
    "rms_norm_forward",
    "roll",
    "round",
    "round_",
    "round_out",
    "rrelu_with_noise_backward",
    "rsqrt",
    "rsqrt_",
    "rsub_scalar",
    "rsub_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_softmax_backward",
    "scaled_softmax_forward",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce_",
    "select_backward",
    "select_scatter",
    "selu",
    "selu_",
    "sgn_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "signbit",
    "signbit_out",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "sinh_",
    "slice_backward",
    "slice_scatter",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "softmax",
    "softmax_backward",
    "softmax_backward_out",
    "softmax_out",
    "softplus",
    "softshrink",
    "softshrink_out",
    "sort",
    "sort_stable",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "stack",
    "std",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "t_copy",
    "t_copy_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "threshold",
    "threshold_backward",
    "tile",
    "to_copy",
    "topk",
    "trace",
    "tril",
    "tril_",
    "tril_out",
    "triu",
    "triu_",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "unfold_backward",
    "uniform_",
    "unique_consecutive",
    "upsample_bicubic2d",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vdot",
    "vector_norm",
    "vstack",
    "w8a8_block_fp8_matmul",
    "weight_norm_interface",
    "weight_norm_interface_backward",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]

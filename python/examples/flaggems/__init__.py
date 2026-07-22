from .__ilshift__ import __ilshift__
from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)
from ._euclidean_dist import _euclidean_dist
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._fused_adam import _fused_adam, _fused_adam_
from ._is_all_true import _is_all_true
from ._jagged_to_padded_dense_forward import (
    _jagged_to_padded_dense_forward,
)
from ._prelu_kernel_backward import _prelu_kernel_backward
from ._resize_output import _resize_output
from ._safe_softmax import _safe_softmax
from ._sparse_semi_structured_mm import _sparse_semi_structured_mm
from ._unsafe_masked_index import _unsafe_masked_index
from ._upsample_bilinear2d_aa import _upsample_bilinear2d_aa
from ._upsample_nearest_exact1d import _upsample_nearest_exact1d
from ._upsample_nearest_exact2d_backward import (
    _upsample_nearest_exact2d_backward,
)
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .adaptive_avg_pool2d import adaptive_avg_pool2d
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .add import add, add_
from .addcdiv import addcdiv, addcdiv_out
from .addcdiv_ import addcdiv_
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from .addmm_ import addmm_
from .addmv import addmv, addmv_out
from .addr import addr
from .affine_grid_generator import affine_grid_generator
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
from .alpha_dropout import alpha_dropout
from .amax import amax
from .aminmax import aminmax
from .angle import angle
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arcsin import arcsin, arcsin_, arcsin_out
from .arcsinh import arcsinh, arcsinh_out
from .arcsinh_ import arcsinh_
from .arctanh_ import arctanh_
from .argmax import argmax
from .argmin import argmin
from .argsort import argsort
from .as_strided_copy import as_strided_copy, as_strided_copy_out
from .asin import asin, asin_
from .asinh import asinh, asinh_out
from .asinh_ import asinh_
from .assert_async import _assert_async
from .atan import atan, atan_
from .atan2 import atan2, atan2_out
from .atanh import atanh, atanh_
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
from .bernoulli import bernoulli
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
from .broadcast_to import broadcast_to
from .cat import cat, cat_out
from .cauchy import cauchy, cauchy_
from .cdist_backward import _cdist_backward
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .channel_shuffle import channel_shuffle
from .clamp import (
    clamp,
    clamp_,
    clamp_min,
    clamp_min_,
    clamp_tensor,
    clamp_tensor_,
)
from .clamp_max import clamp_max, clamp_max_
from .clip import clip, clip_
from .col2im import col2im
from .concatenate import concatenate
from .conj_physical import conj_physical
from .contiguous import contiguous
from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d
from .conv_depthwise2d import _conv_depthwise2d
from .conv_transpose1d import conv_transpose1d
from .conv_transpose2d import conv_transpose2d
from .copy import copy, copy_
from .copysign import copysign, copysign_out
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out
from .count_nonzero import count_nonzero
from .ctc_loss import ctc_loss
from .cudnn_convolution import cudnn_convolution
from .cummax import cummax
from .cummin import cummin
from .cumprod import cumprod, cumprod_
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .deg2rad import deg2rad
from .dequantize import dequantize
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diagonal_copy import diagonal_copy
from .diff import diff
from .digamma_ import digamma_
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from .dot import dot
from .dropout import dropout, dropout_backward
from .elu import elu, elu_, elu_backward
from .embedding import embedding, embedding_backward
from .embedding_dense_backward import embedding_dense_backward
from .empty import empty
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .erfinv_ import erfinv, erfinv_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .expm1 import expm1, expm1_, expm1_out
from .exponential_ import exponential_
from .eye import eye
from .eye_m import eye_m
from .feature_dropout import feature_dropout, feature_dropout_
from .fft import fft
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .fix import fix
from .flash_attention_backward import (
    efficient_attention_backward,
    flash_attention_backward,
    scaled_dot_product_cudnn_attention_backward,
    scaled_dot_product_efficient_attention_backward,
    scaled_dot_product_flash_attention_backward,
)
from .flip import flip
from .floor import floor, floor_out
from .floor_ import floor_
from .fmin import fmin, fmin_out
from .fmod import (
    fmod_scalar,
    fmod_scalar_,
    fmod_tensor,
    fmod_tensor_,
)
from .fmod_ import fmod_
from .fp8_matmul import fp8_matmul
from .fp8_mqa_logits import fp8_mqa_logits
from .fp8_paged_mqa_logits import fp8_paged_mqa_logits
from .full import full
from .full_like import full_like
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .geometric import geometric, geometric_
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
from .hadamard_transform import (
    hadamard_transform,
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_
from .histc import histc
from .hstack import hstack
from .hypot import hypot, hypot_out
from .i0 import i0, i0_out
from .i0_ import i0_
from .im2col import im2col
from .index import index
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_reduce import index_reduce_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .isneginf import isneginf, isneginf_out
from .kron import kron
from .kthvalue import kthvalue
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import (
    lerp_scalar,
    lerp_scalar_,
    lerp_tensor,
    lerp_tensor_,
)
from .lgamma_ import lgamma, lgamma_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linear import linear
from .linspace import linspace
from .log import log
from .log1p import log1p, log1p_out
from .log1p_ import log1p_
from .log10 import log10, log10_, log10_out
from .log_normal_ import log_normal_
from .log_sigmoid import log_sigmoid
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .logaddexp import logaddexp, logaddexp_out
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not, logical_not_
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logical_xor_ import logical_xor_
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
from .median import (
    median,
    median_dim,
    median_dim_values,
    median_out,
)
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, mm_out, router_gemm
from .mode import mode
from .mse_loss import mse_loss
from .mul import mul, mul_
from .multinomial import multinomial
from .mv import mv
from .nan_to_num import nan_to_num
from .nanmedian import (
    nanmedian,
    nanmedian_dim,
    nanmedian_dim_values,
    nanmedian_out,
)
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .negative import negative
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
from .not_equal import not_equal, not_equal_scalar
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
from .poisson import poisson
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
from .rad2deg import rad2deg, rad2deg_
from .rand import rand
from .rand_like import rand_like
from .randint import randint
from .randint_like import randint_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .reflection_pad1d import (
    reflection_pad1d,
    reflection_pad1d_out,
)
from .reflection_pad1d_backward import reflection_pad1d_backward
from .reflection_pad2d import (
    reflection_pad2d,
    reflection_pad2d_out,
)
from .reflection_pad3d_backward import reflection_pad3d_backward
from .relu import relu, relu_
from .relu6 import relu6
from .remainder import remainder, remainder_
from .renorm import renorm, renorm_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .replication_pad1d import (
    replication_pad1d,
    replication_pad1d_out,
)
from .replication_pad3d import replication_pad3d
from .resolve_conj import resolve_conj
from .resolve_neg import resolve_neg
from .rms_norm import (
    rms_norm,
    rms_norm_backward,
    rms_norm_forward,
)
from .roll import roll
from .rot90 import rot90
from .round import round, round_, round_out
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .rrelu_with_noise_functional import (
    rrelu_with_noise_functional,
)
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub_scalar, rsub_tensor
from .scaled_grouped_mm import scaled_grouped_mm
from .scaled_mm import scaled_mm, scaled_mm_out
from .scaled_softmax import (
    scaled_softmax_backward,
    scaled_softmax_forward,
)
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .scatter_reduce import (
    scatter_reduce,
    scatter_reduce_,
    scatter_reduce_out,
)
from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
from .segment_reduce import (
    _segment_reduce_backward,
    _segment_reduce_backward_out,
    segment_reduce,
    segment_reduce_out,
)
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
from .soft_margin_loss import (
    soft_margin_loss,
    soft_margin_loss_out,
)
from .soft_margin_loss_backward import soft_margin_loss_backward
from .softmax import (
    softmax,
    softmax_backward,
    softmax_backward_out,
    softmax_out,
)
from .softplus import softplus
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_gammainc import (
    special_gammainc,
    special_gammainc_out,
)
from .special_hermite_polynomial_h import special_hermite_polynomial_h
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .special_log_softmax import special_log_softmax
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)
from .split_with_sizes_copy import split_with_sizes_copy
from .sqrt import sqrt, sqrt_
from .square import square, square_, square_out
from .stack import stack
from .std import std
from .sub import sub, sub_
from .subtract_ import subtract_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .svd import svd
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .tensor_split import tensor_split
from .threshold import threshold, threshold_backward
from .threshold_ import threshold_
from .tile import tile
from .to import to_copy
from .topk import topk
from .trace import trace
from .tril import tril, tril_, tril_out
from .triu import triu, triu_
from .trunc_ import trunc, trunc_
from .unbind_copy import unbind_copy
from .unfold_backward import unfold_backward
from .unfold_copy import unfold_copy
from .uniform import uniform_
from .unique import _unique2
from .unique_consecutive import unique_consecutive
from .unique_dim import unique_dim
from .upsample_bicubic2d import upsample_bicubic2d
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward
from .upsample_linear1d import upsample_linear1d
from .upsample_linear1d_backward import upsample_linear1d_backward
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest3d import upsample_nearest3d
from .upsample_trilinear3d import upsample_trilinear3d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vdot import vdot
from .vector_norm import vector_norm
from .view_copy import view_copy
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
    "__ilshift__",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_assert_async",
    "_cdist_backward",
    "_conv_depthwise2d",
    "_euclidean_dist",
    "_functional_sym_constrain_range_for_size",
    "_fused_adam",
    "_fused_adam_",
    "_index_put_impl_",
    "_is_all_true",
    "_jagged_to_padded_dense_forward",
    "_prelu_kernel_backward",
    "_resize_output",
    "_safe_softmax",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "_sparse_semi_structured_mm",
    "_unique2",
    "_unsafe_masked_index",
    "_upsample_bicubic2d_aa",
    "_upsample_bicubic2d_aa_backward",
    "_upsample_bilinear2d_aa",
    "_upsample_nearest_exact1d",
    "_upsample_nearest_exact2d_backward",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "adaptive_avg_pool2d",
    "adaptive_max_pool3d_backward",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcdiv_out",
    "addcmul",
    "addcmul_out",
    "addmm",
    "addmm_",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv",
    "addmv_out",
    "addr",
    "affine_grid_generator",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "alpha_dropout",
    "amax",
    "aminmax",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arcsin",
    "arcsin_",
    "arcsin_out",
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "as_strided_copy",
    "as_strided_copy_out",
    "asin",
    "asin_",
    "asinh",
    "asinh_",
    "asinh_out",
    "atan",
    "atan2",
    "atan2_out",
    "atan_",
    "atanh",
    "atanh_",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_out",
    "batch_norm",
    "batch_norm_backward",
    "bernoulli",
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
    "broadcast_to",
    "cat",
    "cat_out",
    "cauchy",
    "cauchy_",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "channel_shuffle",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
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
    "conv_transpose1d",
    "conv_transpose2d",
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
    "ctc_loss",
    "cudnn_convolution",
    "cummax",
    "cummin",
    "cumprod",
    "cumprod_",
    "cumsum",
    "cumsum_out",
    "deg2rad",
    "dequantize",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "diagonal_copy",
    "diff",
    "digamma_",
    "div_mode",
    "div_mode_",
    "dot",
    "dropout",
    "dropout_backward",
    "efficient_attention_backward",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_backward",
    "embedding_dense_backward",
    "empty",
    "eq",
    "eq_scalar",
    "equal",
    "erf",
    "erf_",
    "erfinv",
    "erfinv_",
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
    "fft",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "fix",
    "flash_attention_backward",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_opt_func",
    "flip",
    "floor",
    "floor_",
    "floor_divide",
    "floor_divide_",
    "floor_out",
    "fmin",
    "fmin_out",
    "fmod_",
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "fp8_matmul",
    "fp8_mqa_logits",
    "fp8_paged_mqa_logits",
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
    "geometric",
    "geometric_",
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
    "hadamard_transform_12N",
    "hadamard_transform_20N",
    "hadamard_transform_28N",
    "hadamard_transform_40N",
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
    "im2col",
    "index",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_reduce_",
    "index_select",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isneginf_out",
    "kron",
    "kthvalue",
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
    "lgamma",
    "lgamma_",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linear",
    "linspace",
    "log",
    "log1p",
    "log1p_",
    "log1p_out",
    "log10",
    "log10_",
    "log10_out",
    "log_normal_",
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
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
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
    "median",
    "median_dim",
    "median_dim_values",
    "median_out",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "mm_out",
    "mode",
    "mse_loss",
    "mul",
    "mul_",
    "multinomial",
    "mv",
    "nan_to_num",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "negative",
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
    "not_equal",
    "not_equal_scalar",
    "one_hot",
    "ones",
    "ones_like",
    "pad",
    "per_token_group_quant_fp8",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pixel_unshuffle_out",
    "poisson",
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
    "rad2deg",
    "rad2deg_",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "reflection_pad1d",
    "reflection_pad1d_backward",
    "reflection_pad1d_out",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "reflection_pad3d_backward",
    "relu",
    "relu6",
    "relu_",
    "remainder",
    "remainder_",
    "renorm",
    "renorm_",
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
    "rot90",
    "round",
    "round_",
    "round_out",
    "router_gemm",
    "rrelu_with_noise_backward",
    "rrelu_with_noise_functional",
    "rsqrt",
    "rsqrt_",
    "rsub_scalar",
    "rsub_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_cudnn_attention_backward",
    "scaled_dot_product_efficient_attention_backward",
    "scaled_dot_product_flash_attention_backward",
    "scaled_grouped_mm",
    "scaled_mm",
    "scaled_mm_out",
    "scaled_softmax_backward",
    "scaled_softmax_forward",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "searchsorted",
    "searchsorted_out",
    "searchsorted_scalar",
    "searchsorted_scalar_out",
    "segment_reduce",
    "segment_reduce_out",
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
    "soft_margin_loss_backward",
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
    "special_chebyshev_polynomial_v",
    "special_gammainc",
    "special_gammainc_out",
    "special_hermite_polynomial_h",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "special_log_softmax",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "split_with_sizes_copy",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "stack",
    "std",
    "sub",
    "sub_",
    "subtract_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "svd",
    "t_copy",
    "t_copy_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "tensor_split",
    "threshold",
    "threshold_",
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
    "trunc",
    "trunc_",
    "unbind_copy",
    "unfold_backward",
    "unfold_copy",
    "uniform_",
    "unique_consecutive",
    "unique_dim",
    "upsample_bicubic2d",
    "upsample_linear1d",
    "upsample_linear1d_backward",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_trilinear3d",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vdot",
    "vector_norm",
    "view_copy",
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

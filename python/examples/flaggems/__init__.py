from ._aminmax import _aminmax, _aminmax_out
from ._batch_norm_impl_index import batch_norm_impl_index as _batch_norm_impl_index
from ._batch_norm_impl_index_backward import _batch_norm_impl_index_backward
from ._batch_norm_no_update import _batch_norm_no_update
from ._batch_norm_with_update import _batch_norm_with_update
from ._batch_norm_with_update_functional import _batch_norm_with_update_functional
from ._cslt_sparse_mm import _cslt_sparse_mm
from ._cummin_helper import _cummin_helper
from ._dyn_quant_matmul_4bit import _dyn_quant_matmul_4bit
from ._fake_quantize_learnable_per_channel_affine import (
    _fake_quantize_learnable_per_channel_affine,
)
from ._fake_quantize_learnable_per_channel_affine_backward import (
    _fake_quantize_learnable_per_channel_affine_backward,
)
from ._fake_quantize_learnable_per_tensor_affine import (
    _fake_quantize_learnable_per_tensor_affine,
)
from ._fake_quantize_learnable_per_tensor_affine_backward import (
    _fake_quantize_learnable_per_tensor_affine_backward,
)
from ._fake_quantize_per_tensor_affine_cachemask_tensor_qparams import (
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams,
)
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._fused_rms_norm import (
    _fused_rms_norm,
    _fused_rms_norm_backward,
    _fused_rms_norm_forward,
)
from ._histogramdd_from_bin_cts import (
    _histogramdd_from_bin_cts,
    _histogramdd_from_bin_cts_out,
)
from ._histogramdd_from_bin_tensors import (
    _histogramdd_from_bin_tensors,
    _histogramdd_from_bin_tensors_out,
)
from ._is_all_true import _is_all_true
from ._is_any_true import _is_any_true
from ._logcumsumexp import _logcumsumexp, _logcumsumexp_out
from ._native_batch_norm_legit_functional import _native_batch_norm_legit_functional
from ._native_batch_norm_legit_no_training import _native_batch_norm_legit_no_training
from ._nested_sum_backward import _nested_sum_backward
from ._philox_normal_ import _philox_normal_
from ._safe_softmax import _safe_softmax
from ._scaled_grouped_mm_v2 import _scaled_grouped_mm_v2
from ._scaled_mm_v2 import _scaled_mm_v2
from ._sparse_mm import _sparse_mm
from ._sparse_mm_reduce_impl import _sparse_mm_reduce_impl
from ._sparse_semi_structured_addmm import _sparse_semi_structured_addmm
from ._sparse_semi_structured_mm import _sparse_semi_structured_mm
from ._sparse_sparse_matmul import _sparse_sparse_matmul, _sparse_sparse_matmul_out
from ._sparse_sum_backward import _sparse_sum_backward, _sparse_sum_backward_out
from ._standard_gamma import standard_gamma
from ._standard_gamma_grad import standard_gamma_grad
from ._unsafe_masked_index_put_accumulate import _unsafe_masked_index_put_accumulate
from ._upsample_nearest_exact1d import _upsample_nearest_exact1d
from ._weight_int4pack_mm_with_scales_and_zeros import (
    _weight_int4pack_mm_with_scales_and_zeros,
)
from ._weight_norm import _weight_norm
from ._weight_norm_differentiable_backward import weight_norm_differentiable_backward
from ._wrapped_quantized_linear_prepacked import _wrapped_quantized_linear_prepacked
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .act_quant import act_quant_triton
from .add import add, add_
from .addbmm import addbmm, addbmm_
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
from .batch_norm_backward_elemt import batch_norm_backward_elemt
from .batch_norm_backward_reduce import batch_norm_backward_reduce
from .batch_norm_elemt import batch_norm_elemt
from .batch_norm_stats import batch_norm_stats
from .batch_norm_update_stats import batch_norm_update_stats
from .bernoulli_ import bernoulli_
from .bilinear import bilinear
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
from .bmm_w8a8 import bmm_w8a8
from .bmm_w8a8_fp8 import bmm_w8a8_fp8
from .cartesian_prod import run as cartesian_prod
from .cat import cat, cat_out
from .cdist import _cdist_backward, _cdist_forward, cdist
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .chain_matmul import chain_matmul, chain_matmul_out
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
from .combinations import combinations
from .concatenate import concatenate
from .conj_physical import conj_physical
from .contiguous import contiguous
from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d
from .conv_depthwise2d import _conv_depthwise2d
from .copy import copy, copy_
from .copysign import copysign, copysign_out
from .corrcoef import corrcoef
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out
from .count_nonzero import count_nonzero
from .cov import cov
from .cudnn_batch_norm_backward import cudnn_batch_norm_backward
from .cummax import cummax
from .cummin import cummin
from .cumprod import cumprod, cumprod_
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .cumulative_trapezoid import cumulative_trapezoid
from .dequantize import dequantize
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diff import diff
from .digamma_ import digamma_
from .dist import dist
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
from .fake_quantize_per_channel_affine import fake_quantize_per_channel_affine
from .fake_quantize_per_channel_affine_cachemask import (
    fake_quantize_per_channel_affine_cachemask,
    fake_quantize_per_channel_affine_cachemask_out,
)
from .fake_quantize_per_channel_affine_cachemask_backward import (
    fake_quantize_per_channel_affine_cachemask_backward,
)
from .fake_quantize_per_tensor_affine import fake_quantize_per_tensor_affine
from .fake_quantize_per_tensor_affine_cachemask import (
    fake_quantize_per_tensor_affine_cachemask,
    fake_quantize_per_tensor_affine_cachemask_out,
)
from .fake_quantize_per_tensor_affine_cachemask_backward import (
    fake_quantize_per_tensor_affine_cachemask_backward,
)
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
from .fused_moving_avg_obs_fake_quant import fused_moving_avg_obs_fake_quant
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .gemma_rms_norm import gemma_rms_norm
from .ger import ger
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
from .histogram import histogram, histogram_bin_ct, histogram_bins_tensor
from .histogramdd import histogramdd
from .hspmm import hspmm
from .hstack import hstack
from .hypot import hypot, hypot_out
from .i0 import i0, i0_out
from .i0_ import i0_
from .igamma import igamma
from .igamma_ import igamma_
from .igammac import igammac, igammac_out
from .igammac_ import igammac_
from .index import index
from .index_add import index_add, index_add_
from .index_put import (
    _index_put_impl_,
    index_put,
    index_put_,
)
from .index_select import index_select
from .inner import inner
from .int_mm import int_mm, int_mm_out
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
from .lgamma_ import lgamma, lgamma_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linalg_multi_dot import linalg_multi_dot, linalg_multi_dot_out
from .linalg_norm import linalg_norm
from .linalg_vecdot import linalg_vecdot, linalg_vecdot_out
from .linear import linear
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
from .logcumsumexp import logcumsumexp, logcumsumexp_out
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logit import logit, logit_out
from .logit_ import logit_
from .logspace import logspace
from .logsumexp import logsumexp
from .lt import lt, lt_scalar
from .make_per_channel_quantized_tensor import _make_per_channel_quantized_tensor
from .make_per_tensor_quantized_tensor import (
    _make_per_tensor_quantized_tensor,
    _make_per_tensor_quantized_tensor_out,
)
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
from .mixed_dtypes_linear import mixed_dtypes_linear
from .mm import mm, mm_out
from .mode import mode
from .mse_loss import mse_loss
from .mul import mul, mul_
from .multinomial import multinomial
from .mv import mv
from .mvlgamma import mvlgamma
from .mvlgamma_ import mvlgamma_
from .nan_to_num import nan_to_num
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nanquantile import (
    nanquantile,
    nanquantile_out,
    nanquantile_scalar,
    nanquantile_scalar_out,
)
from .native_group_norm import native_group_norm
from .native_norm import native_norm
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
from .norm_except_dim import norm_except_dim
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
from .pdist import pdist
from .per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from .pixel_shuffle import pixel_shuffle
from .pixel_unshuffle import pixel_unshuffle, pixel_unshuffle_out
from .polar import polar
from .polygamma import polygamma, polygamma_, polygamma_out
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
from .quantize_per_channel import quantize_per_channel, quantize_per_channel_out
from .quantize_per_tensor import quantize_per_tensor, quantize_per_tensor_out
from .quantized_batch_norm import quantized_batch_norm, quantized_batch_norm_out
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
from .renorm import renorm
from .renorm_ import renorm_
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
from .rms_norm_w8a16_fp8 import rms_norm_w8a16_fp8
from .roll import roll
from .round import round, round_, round_out
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub_scalar, rsub_tensor
from .scaled_grouped_mm import scaled_grouped_mm
from .scaled_mm import scaled_mm, scaled_mm_out
from .scaled_softmax import scaled_softmax_backward, scaled_softmax_forward
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .scatter_reduce_ import scatter_reduce_
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
from .smm import smm
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
from .sparse_csr_sum import sparse_csr_sum, sparse_csr_sum_out
from .sparse_sampled_addmm import sparse_sampled_addmm, sparse_sampled_addmm_out
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .sqrt import sqrt, sqrt_
from .square import square, square_, square_out
from .sspaddmm import sspaddmm, sspaddmm_out
from .stack import stack
from .std import std
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .sum_to_size import sum_to_size
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .te_rmsnorm import te_rmsnorm_bwd, te_rmsnorm_fwd
from .tensordot import tensordot, tensordot_out
from .threshold import threshold, threshold_backward
from .tile import tile
from .to import to_copy
from .topk import topk
from .trace import trace
from .trapezoid import trapezoid_x
from .trapz import trapz
from .tril import tril, tril_, tril_out
from .trilinear import _trilinear, _trilinear_out
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
from .vander import vander
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vdot import vdot
from .vector_norm import vector_norm
from .vstack import vstack
from .w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
from .weight_int4pack_mm import weight_int4pack_mm
from .weight_int8pack_mm import weight_int8pack_mm
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

from ._embedding_bag_dense_backward import _embedding_bag_dense_backward
from ._embedding_bag_per_sample_weights_backward import (
    _embedding_bag_per_sample_weights_backward,
)
from ._embedding_bag_sparse_backward import _embedding_bag_sparse_backward
from ._gather_sparse_backward import _gather_sparse_backward
from ._upsample_bilinear2d_aa import _upsample_bilinear2d_aa
from ._upsample_bilinear2d_aa_backward import _upsample_bilinear2d_aa_backward
from ._upsample_lanczos2d_aa_backward import (
    upsample_lanczos2d_aa_backward,
    upsample_lanczos2d_aa_backward_grad_input,
)
from ._upsample_nearest_exact1d_backward import (
    _upsample_nearest_exact1d_backward,
    _upsample_nearest_exact1d_backward_grad_input,
)
from ._upsample_nearest_exact2d import _upsample_nearest_exact2d
from ._upsample_nearest_exact2d_backward import _upsample_nearest_exact2d_backward
from ._upsample_nearest_exact3d import _upsample_nearest_exact3d
from ._upsample_nearest_exact3d_backward import (
    _upsample_nearest_exact3d_backward,
    _upsample_nearest_exact3d_backward_grad_input,
)
from .affine_grid_generator_backward import affine_grid_generator_backward
from .as_strided_scatter import as_strided_scatter
from .batch_norm_gather_stats import batch_norm_gather_stats
from .batch_norm_gather_stats_with_counts import batch_norm_gather_stats_with_counts
from .cosine_embedding_loss import cosine_embedding_loss
from .cudnn_affine_grid_generator_backward import (
    cudnn_affine_grid_generator_backward,
    cudnn_affine_grid_generator_backward_out,
)
from .cudnn_grid_sampler import cudnn_grid_sampler, cudnn_grid_sampler_out
from .cudnn_grid_sampler_backward import (
    cudnn_grid_sampler_backward,
    cudnn_grid_sampler_backward_out,
)
from .diagonal_scatter import diagonal_scatter
from .embedding_bag import _embedding_bag, _embedding_bag_forward_only, embedding_bag
from .embedding_bag_backward import _embedding_bag_backward
from .embedding_renorm_ import embedding_renorm_
from .embedding_sparse_backward import embedding_sparse_backward
from .embedding_spmdm import embedding_spmdm
from .embedding_spmdm_grad import embedding_spmdm_grad
from .gather_block_quantized import gather_block_quantized
from .grid_sampler import grid_sampler
from .grid_sampler_2d import grid_sampler_2d
from .grid_sampler_3d import grid_sampler_3d
from .grid_sampler_3d_backward import grid_sampler_3d_backward
from .hinge_embedding_loss import hinge_embedding_loss
from .index_fill import index_fill, index_fill_
from .index_reduce import index_reduce, index_reduce_, index_reduce_out
from .index_select_backward import index_select_backward
from .masked_scatter_backward import masked_scatter_backward
from .put import put, put_out
from .put_ import put_
from .take import take, take_out
from .take_along_dim import take_along_dim, take_along_dim_out
from .upsample_bicubic2d_backward import (
    upsample_bicubic2d_backward,
    upsample_bicubic2d_backward_grad_input,
)
from .upsample_bilinear2d import upsample_bilinear2d
from .upsample_bilinear2d_backward import (
    upsample_bilinear2d_backward,
    upsample_bilinear2d_backward_grad_input,
)
from .upsample_lanczos2d_aa import (
    _upsample_lanczos2d_aa,
    _upsample_lanczos2d_aa_out,
    _upsample_lanczos2d_aa_vec,
)
from .upsample_linear1d_backward import upsample_linear1d_backward
from .upsample_nearest1d_backward import (
    upsample_nearest1d_backward,
    upsample_nearest1d_backward_grad_input,
)
from .upsample_nearest2d_backward import (
    upsample_nearest2d_backward,
    upsample_nearest2d_backward_grad_input,
)
from .upsample_nearest3d_backward import (
    upsample_nearest3d_backward,
    upsample_nearest3d_backward_grad_input,
)
from .upsample_trilinear3d import upsample_trilinear3d
from .upsample_trilinear3d_backward import upsample_trilinear3d_backward

from .cholesky import cholesky
from .cholesky_inverse import cholesky_inverse
from .cholesky_solve import cholesky_solve
from .detach_copy import detach_copy
from .geqrf import geqrf
from .inverse import inverse
from .ldl_factor_ex import ldl_factor_ex
from .linalg__powsum import linalg__powsum
from .linalg_cholesky import linalg_cholesky
from .linalg_cholesky_ex import linalg_cholesky_ex
from .linalg_cond import linalg_cond
from .linalg_cross import linalg_cross
from .linalg_det import linalg_det
from .linalg_eig import linalg_eig
from .linalg_eigh import linalg_eigh
from .linalg_eigvalsh import linalg_eigvalsh
from .linalg_householder_product import linalg_householder_product
from .linalg_inv import linalg_inv
from .linalg_inv_ex import linalg_inv_ex
from .linalg_ldl_factor import linalg_ldl_factor
from .linalg_ldl_solve import linalg_ldl_solve
from .linalg_lstsq import linalg_lstsq
from .linalg_lu import linalg_lu
from .linalg_lu_factor import linalg_lu_factor
from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_lu_solve import linalg_lu_solve
from .linalg_matrix_exp import linalg_matrix_exp
from .linalg_matrix_norm import linalg_matrix_norm
from .linalg_matrix_power import linalg_matrix_power
from .linalg_matrix_rank import linalg_matrix_rank
from .linalg_matrix_sqrth import linalg_matrix_sqrth
from .linalg_pinv import linalg_pinv
from .linalg_polar import linalg_polar
from .linalg_qr import linalg_qr
from .linalg_slogdet import linalg_slogdet
from .linalg_solve import linalg_solve
from .linalg_solve_ex import linalg_solve_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .linalg_tensorinv import linalg_tensorinv
from .linalg_tensorsolve import linalg_tensorsolve
from .linalg_vander import linalg_vander
from .logdet import logdet
from .lu_unpack import lu_unpack
from .matrix_exp_backward import matrix_exp_backward
from .matrix_power import matrix_power
from .orgqr import orgqr
from .ormqr import ormqr
from .pinverse import pinverse
from .slogdet import slogdet
from .triangular_indices import triangular_indices
from .triangular_solve import triangular_solve
from ._cholesky_solve_helper import _cholesky_solve_helper
from ._dyn_quant_pack_4bit_weight import _dyn_quant_pack_4bit_weight
from ._linalg_det import _linalg_det
from ._linalg_slogdet import _linalg_slogdet
from ._linalg_solve_ex import _linalg_solve_ex
from ._lu_with_info import _lu_with_info
from ._saturate_weight_to_fp16 import _saturate_weight_to_fp16
from ._spsolve import _spsolve

from .argsort import argsort
from .binomial import binomial
from .bucketize import bucketize
from .cauchy import cauchy
from .fft_ifftshift import fft_ifftshift
from .fft_irfftn import fft_irfftn
from .fft_rfft import fft_rfft
from .geometric import geometric
from .gru import gru
from .gru_cell import gru_cell
from .is_nonzero import is_nonzero
from .istft import istft
from .log_normal import log_normal
from .log_normal_ import log_normal_
from .lstm import lstm
from .lstm_cell import lstm_cell
from .mkldnn_rnn_layer import mkldnn_rnn_layer
from .msort import msort
from .nonzero_static import nonzero_static
from .normal_functional import normal_functional
from .quantized_gru import quantized_gru
from .quantized_lstm import quantized_lstm
from .randint import randint
from .randint_like import randint_like
from .rnn_relu import rnn_relu
from .rnn_relu_cell import rnn_relu_cell
from .rnn_tanh import rnn_tanh
from .rnn_tanh_cell import rnn_tanh_cell
from .searchsorted import searchsorted
from .unique_dim import unique_dim
from .unique_dim_consecutive import unique_dim_consecutive
from ._dirichlet_grad import _dirichlet_grad
from ._native_multi_head_attention import _native_multi_head_attention
from ._philox_uniform_ import _philox_uniform_
from ._sample_dirichlet import _sample_dirichlet
from ._scaled_dot_product_attention_math import _scaled_dot_product_attention_math
from ._sobol_engine_ff_ import _sobol_engine_ff_
from ._sobol_engine_initialize_state_ import _sobol_engine_initialize_state_
from ._sobol_engine_scramble_ import _sobol_engine_scramble_
from ._thnn_differentiable_gru_cell_backward import (
    _thnn_differentiable_gru_cell_backward,
)
from ._thnn_fused_gru_cell import _thnn_fused_gru_cell
from ._thnn_fused_lstm_cell import _thnn_fused_lstm_cell
from ._thnn_fused_lstm_cell_backward import _thnn_fused_lstm_cell_backward
from .cross_attention import cross_attention
from .cudnn_attention_backward import cudnn_attention_backward
from .flash_attention_backward import flash_attention_backward
from .thnn_differentiable_lstm_cell_backward import (
    thnn_differentiable_lstm_cell_backward,
)
from .thnn_fused_gru_cell_backward import thnn_fused_gru_cell_backward
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .underscore_sobol_engine_draw import underscore_sobol_engine_draw

from ._adaptive_avg_pool2d_backward import _adaptive_avg_pool2d_backward
from ._adaptive_avg_pool3d import (
    _adaptive_avg_pool3d,
    _adaptive_avg_pool3d_out as adaptive_avg_pool3d_out,
    adaptive_avg_pool3d,
)
from ._convert_weight_to_int4pack import _convert_weight_to_int4pack
from ._convolution_double_backward import _convolution_double_backward
from ._slow_conv2d_backward import _slow_conv2d_backward
from ._slow_conv2d_forward import _slow_conv2d_forward
from .adaptive_avg_pool2d import (
    _adaptive_avg_pool2d_out as adaptive_avg_pool2d_out,
    adaptive_avg_pool2d,
)
from .adaptive_avg_pool3d_backward import (
    _adaptive_avg_pool3d_backward,
    adaptive_avg_pool3d_backward_grad_input,
)
from .adaptive_max_pool2d import adaptive_max_pool2d
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d import adaptive_max_pool3d
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .conv_depthwise3d import conv_depthwise3d
from .conv_tbc import conv_tbc, conv_tbc_out
from .conv_tbc_backward import conv_tbc_backward
from .conv_transpose3d import conv_transpose3d
from .convert_indices_from_coo_to_csr import (
    _convert_indices_from_coo_to_csr,
    _convert_indices_from_coo_to_csr_out,
)
from .convert_indices_from_csr_to_coo import (
    _convert_indices_from_csr_to_coo,
    _convert_indices_from_csr_to_coo_out,
)
from .convolution import convolution
from .convolution_backward import convolution_backward, convolution_backward_out
from .convolution_backward_overrideable import (
    convolution_backward_overrideable,
    convolution_backward_overrideable_out,
)
from .convolution_overrideable import (
    convolution_overrideable,
    convolution_overrideable_out,
)
from .cudnn_convolution_transpose import cudnn_convolution_transpose
from .fractional_max_pool2d import fractional_max_pool2d, fractional_max_pool2d_backward
from .fractional_max_pool3d import fractional_max_pool3d
from .fractional_max_pool3d_backward import fractional_max_pool3d_backward
from .im2col import im2col
from .max_pool1d import max_pool1d
from .max_pool1d_with_indices import max_pool1d_with_indices
from .max_pool3d_with_indices_backward import max_pool3d_with_indices_backward
from .max_unpool2d import max_unpool2d
from .max_unpool3d import max_unpool3d
from .mkldnn_reorder_conv2d_weight import (
    mkldnn_reorder_conv2d_weight,
    mkldnn_reorder_conv2d_weight_out,
)
from .quantized_max_pool1d import quantized_max_pool1d, quantized_max_pool1d_out
from .quantized_max_pool2d import quantized_max_pool2d, quantized_max_pool2d_out
from .quantized_max_pool3d import quantized_max_pool3d, quantized_max_pool3d_out
from .slow_conv_dilated2d import slow_conv_dilated2d, slow_conv_dilated2d_out
from .slow_conv_dilated3d import slow_conv_dilated3d, slow_conv_dilated3d_out
from .slow_conv_transpose2d import slow_conv_transpose2d
from .slow_conv_transpose3d import slow_conv_transpose3d
from .thnn_conv2d import thnn_conv2d

from .ccol_indices_copy import ccol_indices_copy
from .coalesce import coalesce
from .col_indices import col_indices
from .col_indices_copy import col_indices_copy
from .copy_sparse_to_sparse_ import copy_sparse_to_sparse_
from .crow_indices import crow_indices
from .crow_indices_copy import crow_indices_copy
from .indices_copy import indices_copy
from .nested_to_padded_tensor import nested_to_padded_tensor
from .row_indices_copy import row_indices_copy
from .sparse_compressed_tensor import sparse_compressed_tensor
from .sparse_coo_tensor import sparse_coo_tensor
from .sparse_csc_tensor import sparse_csc_tensor
from .sparse_csr_tensor import sparse_csr_tensor
from .sparse_mask import sparse_mask
from .sparse_resize_ import sparse_resize_
from .to_dense_backward import to_dense_backward
from .values_copy import values_copy
from ._coalesced_ import _coalesced_
from ._indices_copy import _indices_copy
from ._jagged_to_padded_dense_forward import _jagged_to_padded_dense_forward
from ._nested_from_padded import _nested_from_padded
from ._nested_from_padded_and_nested_example import (
    _nested_from_padded_and_nested_example,
)
from ._nested_from_padded_tensor import _nested_from_padded_tensor
from ._nested_select_backward import _nested_select_backward
from ._nested_tensor_from_mask import _nested_tensor_from_mask
from ._nested_tensor_from_mask_left_aligned import _nested_tensor_from_mask_left_aligned
from ._nested_tensor_softmax_with_shape import _nested_tensor_softmax_with_shape
from ._nested_view_from_jagged_copy import _nested_view_from_jagged_copy
from ._pack_padded_sequence import _pack_padded_sequence
from ._sparse_log_softmax import _sparse_log_softmax
from ._sparse_log_softmax_backward_data import _sparse_log_softmax_backward_data
from ._sparse_mask_projection import _sparse_mask_projection
from ._sparse_semi_structured_apply import _sparse_semi_structured_apply
from ._sparse_semi_structured_apply_dense import _sparse_semi_structured_apply_dense
from ._sparse_semi_structured_linear import _sparse_semi_structured_linear
from ._sparse_semi_structured_tile import _sparse_semi_structured_tile
from ._sparse_softmax import _sparse_softmax
from ._sparse_softmax_backward_data import _sparse_softmax_backward_data
from ._validate_sparse_coo_tensor_args import _validate_sparse_coo_tensor_args
from ._validate_sparse_csc_tensor_args import _validate_sparse_csc_tensor_args
from ._values_copy import _values_copy
from .padded_dense_to_jagged_forward import padded_dense_to_jagged_forward
from .sparse_broadcast_to import sparse_broadcast_to
from .sparse_bsc_tensor_unsafe import sparse_bsc_tensor_unsafe
from .sparse_bsr_tensor_unsafe import sparse_bsr_tensor_unsafe
from .sparse_csc_tensor_unsafe import sparse_csc_tensor_unsafe
from .to_sparse_semi_structured import to_sparse_semi_structured

__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "ScaleDotProductAttention",
    "_aminmax",
    "_aminmax_out",
    "_assert_async",
    "_batch_norm_impl_index",
    "_batch_norm_impl_index_backward",
    "_batch_norm_no_update",
    "_batch_norm_with_update",
    "_batch_norm_with_update_functional",
    "_cdist_backward",
    "_cdist_forward",
    "_conv_depthwise2d",
    "_cslt_sparse_mm",
    "_cummin_helper",
    "_dyn_quant_matmul_4bit",
    "_fake_quantize_learnable_per_channel_affine",
    "_fake_quantize_learnable_per_channel_affine_backward",
    "_fake_quantize_learnable_per_tensor_affine",
    "_fake_quantize_learnable_per_tensor_affine_backward",
    "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_functional_sym_constrain_range_for_size",
    "_fused_rms_norm",
    "_fused_rms_norm_backward",
    "_fused_rms_norm_forward",
    "_histogramdd_from_bin_cts",
    "_histogramdd_from_bin_cts_out",
    "_histogramdd_from_bin_tensors",
    "_histogramdd_from_bin_tensors_out",
    "_index_put_impl_",
    "_is_all_true",
    "_is_any_true",
    "_logcumsumexp",
    "_logcumsumexp_out",
    "_make_per_channel_quantized_tensor",
    "_make_per_tensor_quantized_tensor",
    "_make_per_tensor_quantized_tensor_out",
    "_native_batch_norm_legit_functional",
    "_native_batch_norm_legit_no_training",
    "_nested_sum_backward",
    "_philox_normal_",
    "_safe_softmax",
    "_scaled_grouped_mm_v2",
    "_scaled_mm_v2",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "_sparse_mm",
    "_sparse_mm_reduce_impl",
    "_sparse_semi_structured_addmm",
    "_sparse_semi_structured_mm",
    "_sparse_sparse_matmul",
    "_sparse_sparse_matmul_out",
    "_sparse_sum_backward",
    "_sparse_sum_backward_out",
    "_trilinear",
    "_trilinear_out",
    "_unique2",
    "_unsafe_masked_index_put_accumulate",
    "_upsample_bicubic2d_aa",
    "_upsample_bicubic2d_aa_backward",
    "_upsample_nearest_exact1d",
    "_weight_int4pack_mm_with_scales_and_zeros",
    "_weight_norm",
    "_wrapped_quantized_linear_prepacked",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "act_quant_triton",
    "add",
    "add_",
    "addbmm",
    "addbmm_",
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
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bernoulli_",
    "bilinear",
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
    "bmm_w8a8",
    "bmm_w8a8_fp8",
    "cartesian_prod",
    "cat",
    "cat_out",
    "cdist",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "chain_matmul",
    "chain_matmul_out",
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "col2im",
    "combinations",
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
    "corrcoef",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "cov",
    "cudnn_batch_norm_backward",
    "cummax",
    "cummin",
    "cumprod",
    "cumprod_",
    "cumsum",
    "cumsum_out",
    "cumulative_trapezoid",
    "dequantize",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "diff",
    "digamma_",
    "dist",
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
    "fake_quantize_per_channel_affine",
    "fake_quantize_per_channel_affine_cachemask",
    "fake_quantize_per_channel_affine_cachemask_backward",
    "fake_quantize_per_channel_affine_cachemask_out",
    "fake_quantize_per_tensor_affine",
    "fake_quantize_per_tensor_affine_cachemask",
    "fake_quantize_per_tensor_affine_cachemask_backward",
    "fake_quantize_per_tensor_affine_cachemask_out",
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
    "fused_moving_avg_obs_fake_quant",
    "gather",
    "gather_backward",
    "gcd",
    "gcd_out",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "gemma_rms_norm",
    "ger",
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
    "histogram",
    "histogram_bin_ct",
    "histogram_bins_tensor",
    "histogramdd",
    "hspmm",
    "hstack",
    "hypot",
    "hypot_out",
    "i0",
    "i0_",
    "i0_out",
    "igamma",
    "igamma_",
    "igammac",
    "igammac_",
    "igammac_out",
    "index",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_select",
    "inner",
    "int_mm",
    "int_mm_out",
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
    "lgamma",
    "lgamma_",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linalg_multi_dot",
    "linalg_multi_dot_out",
    "linalg_norm",
    "linalg_vecdot",
    "linalg_vecdot_out",
    "linear",
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
    "logcumsumexp",
    "logcumsumexp_out",
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
    "mixed_dtypes_linear",
    "mm",
    "mm_out",
    "mode",
    "mse_loss",
    "mul",
    "mul_",
    "multinomial",
    "mv",
    "mvlgamma",
    "mvlgamma_",
    "nan_to_num",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nanquantile",
    "nanquantile_out",
    "nanquantile_scalar",
    "nanquantile_scalar_out",
    "native_group_norm",
    "native_norm",
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
    "norm_except_dim",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "one_hot",
    "ones",
    "ones_like",
    "pad",
    "pdist",
    "per_token_group_quant_fp8",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pixel_unshuffle_out",
    "polar",
    "polygamma",
    "polygamma_",
    "polygamma_out",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "prod",
    "prod_dim",
    "quantile",
    "quantize_per_channel",
    "quantize_per_channel_out",
    "quantize_per_tensor",
    "quantize_per_tensor_out",
    "quantized_batch_norm",
    "quantized_batch_norm_out",
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
    "rms_norm_w8a16_fp8",
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
    "scaled_grouped_mm",
    "scaled_mm",
    "scaled_mm_out",
    "scaled_softmax_backward",
    "scaled_softmax_forward",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce_",
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
    "smm",
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
    "sparse_csr_sum",
    "sparse_csr_sum_out",
    "sparse_sampled_addmm",
    "sparse_sampled_addmm_out",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "sspaddmm",
    "sspaddmm_out",
    "stack",
    "standard_gamma",
    "standard_gamma_grad",
    "std",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "sum_to_size",
    "t_copy",
    "t_copy_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "te_rmsnorm_bwd",
    "te_rmsnorm_fwd",
    "tensordot",
    "tensordot_out",
    "threshold",
    "threshold_backward",
    "tile",
    "to_copy",
    "topk",
    "trace",
    "trapezoid_x",
    "trapz",
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
    "vander",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vdot",
    "vector_norm",
    "vstack",
    "w8a8_block_fp8_matmul",
    "weight_int4pack_mm",
    "weight_int8pack_mm",
    "weight_norm_differentiable_backward",
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
    "_embedding_bag",
    "_embedding_bag_backward",
    "_embedding_bag_dense_backward",
    "_embedding_bag_forward_only",
    "_embedding_bag_per_sample_weights_backward",
    "_embedding_bag_sparse_backward",
    "_gather_sparse_backward",
    "_upsample_bilinear2d_aa",
    "_upsample_bilinear2d_aa_backward",
    "_upsample_lanczos2d_aa",
    "_upsample_lanczos2d_aa_out",
    "_upsample_lanczos2d_aa_vec",
    "_upsample_nearest_exact1d_backward",
    "_upsample_nearest_exact1d_backward_grad_input",
    "_upsample_nearest_exact2d",
    "_upsample_nearest_exact2d_backward",
    "_upsample_nearest_exact3d",
    "_upsample_nearest_exact3d_backward",
    "_upsample_nearest_exact3d_backward_grad_input",
    "affine_grid_generator_backward",
    "as_strided_scatter",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "cosine_embedding_loss",
    "cudnn_affine_grid_generator_backward",
    "cudnn_affine_grid_generator_backward_out",
    "cudnn_grid_sampler",
    "cudnn_grid_sampler_backward",
    "cudnn_grid_sampler_backward_out",
    "cudnn_grid_sampler_out",
    "diagonal_scatter",
    "embedding_bag",
    "embedding_renorm_",
    "embedding_sparse_backward",
    "embedding_spmdm",
    "embedding_spmdm_grad",
    "gather_block_quantized",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "grid_sampler_3d_backward",
    "hinge_embedding_loss",
    "index_fill",
    "index_fill_",
    "index_reduce",
    "index_reduce_",
    "index_reduce_out",
    "index_select_backward",
    "masked_scatter_backward",
    "put",
    "put_",
    "put_out",
    "take",
    "take_along_dim",
    "take_along_dim_out",
    "take_out",
    "upsample_bicubic2d_backward",
    "upsample_bicubic2d_backward_grad_input",
    "upsample_bilinear2d",
    "upsample_bilinear2d_backward",
    "upsample_bilinear2d_backward_grad_input",
    "upsample_lanczos2d_aa_backward",
    "upsample_lanczos2d_aa_backward_grad_input",
    "upsample_linear1d_backward",
    "upsample_nearest1d_backward",
    "upsample_nearest1d_backward_grad_input",
    "upsample_nearest2d_backward",
    "upsample_nearest2d_backward_grad_input",
    "upsample_nearest3d_backward",
    "upsample_nearest3d_backward_grad_input",
    "upsample_trilinear3d",
    "upsample_trilinear3d_backward",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "detach_copy",
    "geqrf",
    "inverse",
    "ldl_factor_ex",
    "linalg__powsum",
    "linalg_cholesky",
    "linalg_cholesky_ex",
    "linalg_cond",
    "linalg_cross",
    "linalg_det",
    "linalg_eig",
    "linalg_eigh",
    "linalg_eigvalsh",
    "linalg_householder_product",
    "linalg_inv",
    "linalg_inv_ex",
    "linalg_ldl_factor",
    "linalg_ldl_solve",
    "linalg_lstsq",
    "linalg_lu",
    "linalg_lu_factor",
    "linalg_lu_factor_ex",
    "linalg_lu_solve",
    "linalg_matrix_exp",
    "linalg_matrix_norm",
    "linalg_matrix_power",
    "linalg_matrix_rank",
    "linalg_matrix_sqrth",
    "linalg_pinv",
    "linalg_polar",
    "linalg_qr",
    "linalg_slogdet",
    "linalg_solve",
    "linalg_solve_ex",
    "linalg_solve_triangular",
    "linalg_tensorinv",
    "linalg_tensorsolve",
    "linalg_vander",
    "logdet",
    "lu_unpack",
    "matrix_exp_backward",
    "matrix_power",
    "orgqr",
    "ormqr",
    "pinverse",
    "slogdet",
    "triangular_indices",
    "triangular_solve",
    "_cholesky_solve_helper",
    "_dyn_quant_pack_4bit_weight",
    "_linalg_det",
    "_linalg_slogdet",
    "_linalg_solve_ex",
    "_lu_with_info",
    "_saturate_weight_to_fp16",
    "_spsolve",
    "_adaptive_avg_pool2d_backward",
    "_adaptive_avg_pool3d",
    "_adaptive_avg_pool3d_backward",
    "_convert_indices_from_coo_to_csr",
    "_convert_indices_from_coo_to_csr_out",
    "_convert_indices_from_csr_to_coo",
    "_convert_indices_from_csr_to_coo_out",
    "_convert_weight_to_int4pack",
    "_convolution_double_backward",
    "_slow_conv2d_backward",
    "_slow_conv2d_forward",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_out",
    "adaptive_avg_pool3d",
    "adaptive_avg_pool3d_backward_grad_input",
    "adaptive_avg_pool3d_out",
    "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d",
    "adaptive_max_pool3d_backward",
    "conv_depthwise3d",
    "conv_tbc",
    "conv_tbc_backward",
    "conv_tbc_out",
    "conv_transpose3d",
    "convolution",
    "convolution_backward",
    "convolution_backward_out",
    "convolution_backward_overrideable",
    "convolution_backward_overrideable_out",
    "convolution_overrideable",
    "convolution_overrideable_out",
    "cudnn_convolution_transpose",
    "fractional_max_pool2d",
    "fractional_max_pool2d_backward",
    "fractional_max_pool3d",
    "fractional_max_pool3d_backward",
    "im2col",
    "max_pool1d",
    "max_pool1d_with_indices",
    "max_pool3d_with_indices_backward",
    "max_unpool2d",
    "max_unpool3d",
    "mkldnn_reorder_conv2d_weight",
    "mkldnn_reorder_conv2d_weight_out",
    "quantized_max_pool1d",
    "quantized_max_pool1d_out",
    "quantized_max_pool2d",
    "quantized_max_pool2d_out",
    "quantized_max_pool3d",
    "quantized_max_pool3d_out",
    "slow_conv_dilated2d",
    "slow_conv_dilated2d_out",
    "slow_conv_dilated3d",
    "slow_conv_dilated3d_out",
    "slow_conv_transpose2d",
    "slow_conv_transpose3d",
    "thnn_conv2d",
    "argsort",
    "binomial",
    "bucketize",
    "cauchy",
    "fft_ifftshift",
    "fft_irfftn",
    "fft_rfft",
    "geometric",
    "gru",
    "gru_cell",
    "is_nonzero",
    "istft",
    "log_normal",
    "log_normal_",
    "lstm",
    "lstm_cell",
    "mkldnn_rnn_layer",
    "msort",
    "nonzero_static",
    "normal_functional",
    "quantized_gru",
    "quantized_lstm",
    "randint",
    "randint_like",
    "rnn_relu",
    "rnn_relu_cell",
    "rnn_tanh",
    "rnn_tanh_cell",
    "searchsorted",
    "unique_dim",
    "unique_dim_consecutive",
    "_dirichlet_grad",
    "_native_multi_head_attention",
    "_philox_uniform_",
    "_sample_dirichlet",
    "_scaled_dot_product_attention_math",
    "_sobol_engine_ff_",
    "_sobol_engine_initialize_state_",
    "_sobol_engine_scramble_",
    "_thnn_differentiable_gru_cell_backward",
    "_thnn_fused_gru_cell",
    "_thnn_fused_lstm_cell",
    "_thnn_fused_lstm_cell_backward",
    "cross_attention",
    "cudnn_attention_backward",
    "flash_attention_backward",
    "thnn_differentiable_lstm_cell_backward",
    "thnn_fused_gru_cell_backward",
    "topk_w8a16_fp8",
    "underscore_sobol_engine_draw",
    "ccol_indices_copy",
    "coalesce",
    "col_indices",
    "col_indices_copy",
    "copy_sparse_to_sparse_",
    "crow_indices",
    "crow_indices_copy",
    "indices_copy",
    "nested_to_padded_tensor",
    "row_indices_copy",
    "sparse_compressed_tensor",
    "sparse_coo_tensor",
    "sparse_csc_tensor",
    "sparse_csr_tensor",
    "sparse_mask",
    "sparse_resize_",
    "to_dense_backward",
    "values_copy",
    "_coalesced_",
    "_indices_copy",
    "_jagged_to_padded_dense_forward",
    "_nested_from_padded",
    "_nested_from_padded_and_nested_example",
    "_nested_from_padded_tensor",
    "_nested_select_backward",
    "_nested_tensor_from_mask",
    "_nested_tensor_from_mask_left_aligned",
    "_nested_tensor_softmax_with_shape",
    "_nested_view_from_jagged_copy",
    "_pack_padded_sequence",
    "_sparse_log_softmax",
    "_sparse_log_softmax_backward_data",
    "_sparse_mask_projection",
    "_sparse_semi_structured_apply",
    "_sparse_semi_structured_apply_dense",
    "_sparse_semi_structured_linear",
    "_sparse_semi_structured_tile",
    "_sparse_softmax",
    "_sparse_softmax_backward_data",
    "_validate_sparse_coo_tensor_args",
    "_validate_sparse_csc_tensor_args",
    "_values_copy",
    "padded_dense_to_jagged_forward",
    "sparse_broadcast_to",
    "sparse_bsc_tensor_unsafe",
    "sparse_bsr_tensor_unsafe",
    "sparse_csc_tensor_unsafe",
    "to_sparse_semi_structured",
]

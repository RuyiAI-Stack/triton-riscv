import torch
import triton
import triton.language as tl


def make_3d_for_bn(input):
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    is_train: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)
    count = batch_dim * spatial_dim

    if is_train:
        mean = tl.full((), 0.0, dtype=tl.float32)
        m2 = tl.full((), 0.0, dtype=tl.float32)
        sample_count = tl.full((), 0.0, dtype=tl.float32)
        for offset in range(0, count):
            batch_offset = offset // spatial_dim
            spatial_offset = offset % spatial_dim
            input_offset = (
                input_feat_stride * feat_pid
                + input_batch_stride * batch_offset
                + input_spatial_stride * spatial_offset
            )
            value = tl.load(input_pointer + input_offset).to(tl.float32)
            sample_count += 1.0
            delta = value - mean
            mean += delta / sample_count
            m2 += delta * (value - mean)
        var = m2 / sample_count
        inv_std = tl.math.rsqrt(var + eps)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        tl.store(
            running_mean_pointer,
            (1 - momentum) * running_mean + momentum * mean,
        )
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * count / (count - 1),
        )
    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = tl.math.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    tl.store(feat_pid + mean_pointer, mean)
    tl.store(feat_pid + inv_std_pointer, inv_std)

    if HAS_WEIGHT:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    for offset in range(0, count):
        batch_offset = offset // spatial_dim
        spatial_offset = offset % spatial_dim
        input_offset = (
            input_feat_stride * feat_pid
            + input_batch_stride * batch_offset
            + input_spatial_stride * spatial_offset
        )
        output_offset = (
            output_feat_stride * feat_pid
            + output_batch_stride * batch_offset
            + output_spatial_stride * spatial_offset
        )
        value = tl.load(input_pointer + input_offset).to(tl.float32)
        output = weight * (value - mean) * inv_std + bias
        tl.store(output_pointer + output_offset, output)


@triton.jit
def batch_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    input_grad_mask: tl.constexpr,
    weight_grad_mask: tl.constexpr,
    bias_grad_mask: tl.constexpr,
    IS_TRAIN: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    mean = tl.load(feat_pid + mean_pointer).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)
    count = batch_dim * spatial_dim
    term1 = tl.full((), 0.0, dtype=tl.float32)
    term2 = tl.full((), 0.0, dtype=tl.float32)
    for offset in range(0, count):
        batch_offset = offset // spatial_dim
        spatial_offset = offset % spatial_dim
        output_grad_offset = (
            output_grad_feat_stride * feat_pid
            + output_grad_batch_stride * batch_offset
            + output_grad_spatial_stride * spatial_offset
        )
        input_offset = (
            input_feat_stride * feat_pid
            + input_batch_stride * batch_offset
            + input_spatial_stride * spatial_offset
        )
        curr_input = tl.load(input_pointer + input_offset).to(tl.float32)
        curr_output_grad = tl.load(output_grad_pointer + output_grad_offset).to(
            tl.float32
        )
        curr_pre_lin = (curr_input - mean) * inv_std
        term1 += curr_pre_lin * curr_output_grad
        term2 += curr_output_grad

    if weight_grad_mask:
        tl.store(feat_pid + weight_grad_pointer, term1)
    if bias_grad_mask:
        tl.store(feat_pid + bias_grad_pointer, term2)

    if not input_grad_mask:
        return

    if HAS_WEIGHT:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0

    for offset in range(0, count):
        batch_offset = offset // spatial_dim
        spatial_offset = offset % spatial_dim
        output_grad_offset = (
            output_grad_feat_stride * feat_pid
            + output_grad_batch_stride * batch_offset
            + output_grad_spatial_stride * spatial_offset
        )
        input_offset = (
            input_feat_stride * feat_pid
            + input_batch_stride * batch_offset
            + input_spatial_stride * spatial_offset
        )
        input_grad_offset = (
            input_grad_feat_stride * feat_pid
            + input_grad_batch_stride * batch_offset
            + input_grad_spatial_stride * spatial_offset
        )
        curr_input = tl.load(input_pointer + input_offset).to(tl.float32)
        curr_output_grad = tl.load(output_grad_pointer + output_grad_offset).to(
            tl.float32
        )
        curr_pre_lin = (curr_input - mean) * inv_std
        if IS_TRAIN:
            curr_input_grad = (
                inv_std
                * weight
                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
            )
        else:
            curr_input_grad = inv_std * weight * curr_output_grad
        tl.store(input_grad_pointer + input_grad_offset, curr_input_grad)


def batch_norm(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    batch_norm_forward_kernel[(feat_dim,)](
        input_3d,
        weight,
        bias,
        mean,
        inv_std,
        output,
        running_mean,
        running_var,
        batch_dim,
        spatial_dim,
        *input_3d.stride(),
        *output.stride(),
        momentum,
        eps,
        is_train=training,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )

    return output.view_as(input), mean, inv_std


def batch_norm_backward(
    grad_out,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    output_mask=None,
):
    input_3d = make_3d_for_bn(input)
    output_grad_3d = make_3d_for_bn(grad_out)

    batch_dim, feat_dim, spatial_dim = input_3d.shape

    if output_mask is None:
        output_mask = (True, True, True)

    if output_mask[0]:
        input_grad = torch.empty_like(input_3d)
    else:
        input_grad = None
    if output_mask[1]:
        weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        bias_grad = None

    input_grad_arg = input_grad if input_grad is not None else input_3d
    batch_norm_backward_kernel[(feat_dim,)](
        output_grad_3d,
        input_3d,
        save_mean,
        save_invstd,
        weight,
        input_grad_arg,
        weight_grad,
        bias_grad,
        batch_dim,
        spatial_dim,
        *output_grad_3d.stride(),
        *input_3d.stride(),
        *input_grad_arg.stride(),
        *output_mask,
        IS_TRAIN=train,
        HAS_WEIGHT=weight is not None,
    )

    return (
        input_grad.view_as(input) if input_grad is not None else None,
        weight_grad,
        bias_grad,
    )

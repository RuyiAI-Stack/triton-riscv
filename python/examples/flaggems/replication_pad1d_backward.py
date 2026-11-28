import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ATen's error messages use the C++ scalar-type spelling, e.g. "Float"/"Double".
_SCALAR_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_ACC_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}


def _scalar_name(dtype):
    return _SCALAR_NAMES.get(dtype, str(dtype).replace("torch.", ""))


@triton.jit(
    do_not_specialize=[
        "pad_left",
        "pad_right",
        "grad_out_stride_n",
        "grad_out_stride_c",
        "grad_in_stride_n",
    ]
)
def replication_pad1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    W_in: tl.constexpr,
    W_out: tl.constexpr,
    pad_left,
    pad_right,
    grad_out_stride_n,
    grad_out_stride_c,
    grad_out_stride_w,
    grad_in_stride_n,
    grad_in_stride_c,
    grad_in_stride_w,
    LEFT_STOP: tl.constexpr,
    RIGHT_START: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for replication_pad1d.

    Strategy (aten's CPU order: contributions summed in ascending output
    index, in the element dtype via ACC_DTYPE):
    - Each program handles one (N, C) slice
    - Left edge (w_in=0): sum grad_out[0:pad_left+1]
    - Middle (0 < w_in < W_in-1): direct copy grad_out[pad_left+w_in]
    - Right edge (w_in=W_in-1): sum grad_out[pad_left+W_in-1:W_out]

    The loop bounds LEFT_STOP / RIGHT_START are computed on the host so the
    scalar edge branches stay compile-time (this backend rejects runtime
    scalar if-chains). W_in/W_out are constexpr for the same reason.
    """
    pid_nc = tl.program_id(axis=0)

    n = pid_nc // C
    c = pid_nc % C

    # Base offsets
    base_out = n.to(tl.int64) * grad_out_stride_n + c.to(tl.int64) * grad_out_stride_c
    base_in = n.to(tl.int64) * grad_in_stride_n + c.to(tl.int64) * grad_in_stride_c

    # Handle left edge (w_in=0): accumulate grad_out[0:pad_left+1].
    #
    # With a negative pad_left the input column 0 is cropped away rather than
    # replicated, so it receives no gradient at all and the loop must not run:
    # clamping the lower bound to 0 (host-side, in LEFT_STOP) makes the range
    # empty for pad_left < 0.
    if W_in >= 1:
        left_acc = tl.zeros((), dtype=ACC_DTYPE)
        for i in range(0, LEFT_STOP):
            ptr = grad_out_ptr + base_out + i * grad_out_stride_w
            left_acc += tl.load(ptr).to(ACC_DTYPE)
        ptr_in = grad_in_ptr + base_in
        tl.store(ptr_in, left_acc.to(grad_out_ptr.dtype.element_ty))

    # Handle middle elements: direct 1:1 mapping (vectorized)
    for block_start in range(1, W_in - 1, BLOCK_SIZE):
        w_in = block_start + tl.arange(0, BLOCK_SIZE)
        mask = w_in < W_in - 1

        w_out = pad_left + w_in
        # Both bounds are needed. A negative pad_left makes w_out negative for
        # the leading columns; columns cropped away by negative padding receive
        # no gradient, hence the `other=0.0` store.
        out_mask = mask & (w_out >= 0) & (w_out < W_out)

        out_offset = w_out.to(tl.int64) * grad_out_stride_w
        out_ptr = grad_out_ptr + base_out + out_offset
        val = tl.load(out_ptr, mask=out_mask, other=0.0)

        in_offset = w_in.to(tl.int64) * grad_in_stride_w
        in_ptr = grad_in_ptr + base_in + in_offset
        tl.store(in_ptr, val.to(grad_out_ptr.dtype.element_ty), mask=mask)

    # Handle right edge (w_in=W_in-1): accumulate grad_out[RIGHT_START:W_out].
    #
    # RIGHT_START is clamped to 0 on the host: with a large negative pad_left
    # it can fall below the start of grad_output. The range is empty when the
    # last input column is cropped away (start >= W_out).
    if W_in >= 2:
        right_acc = tl.zeros((), dtype=ACC_DTYPE)
        for i in range(RIGHT_START, W_out):
            ptr = grad_out_ptr + base_out + i * grad_out_stride_w
            right_acc += tl.load(ptr).to(ACC_DTYPE)
        ptr_in = grad_in_ptr + base_in + (W_in - 1) * grad_in_stride_w
        tl.store(ptr_in, right_acc.to(grad_out_ptr.dtype.element_ty))
    elif W_in == 1:
        # Special case: only one element, accumulate all output
        acc = tl.zeros((), dtype=ACC_DTYPE)
        for i in range(0, W_out):
            ptr = grad_out_ptr + base_out + i * grad_out_stride_w
            acc += tl.load(ptr).to(ACC_DTYPE)
        ptr_in = grad_in_ptr + base_in
        tl.store(ptr_in, acc.to(grad_out_ptr.dtype.element_ty))


def _launch_replication_pad1d_backward_kernel(
    grad_output: torch.Tensor,
    self_tensor: torch.Tensor,
    padding,
    grad_input: torch.Tensor,
):
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    dim = self_tensor.dim()
    if dim not in (2, 3):
        raise ValueError(
            "replication_pad1d_backward expects 2D (C, W) or 3D (N, C, W) input"
        )

    if dim == 3:
        N, C, W_in = self_tensor.shape
        B = N
        grad_in_s_n, grad_in_s_c, grad_in_s_w = grad_input.stride()
        grad_out_s_n, grad_out_s_c, grad_out_s_w = grad_output.stride()
    else:
        C, W_in = self_tensor.shape
        B = 1
        grad_in_s_c, grad_in_s_w = grad_input.stride()
        grad_in_s_n = 0
        if grad_output.dim() == 2:
            grad_out_s_c, grad_out_s_w = grad_output.stride()
            grad_out_s_n = 0
        elif grad_output.dim() == 3:
            grad_out_s_n, grad_out_s_c, grad_out_s_w = grad_output.stride()
        else:
            raise ValueError("grad_output has invalid dimensions")

    W_out = W_in + left + right

    # Validate shapes
    expected_grad_output_shape = (N, C, W_out) if dim == 3 else (C, W_out)
    if tuple(grad_output.shape) != expected_grad_output_shape:
        raise ValueError(
            f"grad_output has incorrect shape. Expected {expected_grad_output_shape}, got {tuple(grad_output.shape)}"
        )

    if tuple(grad_input.shape) != tuple(self_tensor.shape):
        raise ValueError(
            f"grad_input has incorrect shape. Expected {tuple(self_tensor.shape)}, got {tuple(grad_input.shape)}"
        )

    # Grid: one block per (N, C) combination
    grid = (B * C,)
    BLOCK_SIZE = 256

    # Host-side loop bounds for the scalar edge accumulators (see kernel
    # docstring). Values follow aten's negative-padding semantics.
    left_stop = min(max(left + 1, 0), W_out)
    right_start = max(left + W_in - 1, 0)

    # Accumulate in the element dtype, exactly like aten's CPU kernel (it
    # sums in scalar_t); upstream staged fp16/bf16 through fp32.
    acc_dtype = _ACC_DTYPES.get(self_tensor.dtype, tl.float32)

    replication_pad1d_backward_kernel[grid](
        grad_output,
        grad_input,
        B,
        C,
        W_in,
        W_out,
        left,
        right,
        grad_out_s_n,
        grad_out_s_c,
        grad_out_s_w,
        grad_in_s_n,
        grad_in_s_c,
        grad_in_s_w,
        LEFT_STOP=left_stop,
        RIGHT_START=right_start,
        ACC_DTYPE=acc_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_input


def replication_pad1d_backward(
    grad_output: torch.Tensor, self_tensor: torch.Tensor, padding
) -> torch.Tensor:
    """aten::replication_pad1d_backward; padding order (left, right) like aten.

    Ported from upstream FlagGems. Deviations to match the aten oracle:
    contributions accumulate in the element dtype (aten sums in scalar_t), the
    scalar edge-loop bounds are host-computed constexpr, and W_in/W_out moved
    to constexpr so the kernel holds no runtime scalar branches.
    """
    logger.debug("GEMS REPLICATION_PAD1D_BACKWARD")
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    dim = self_tensor.dim()
    if dim not in (2, 3):
        raise ValueError(
            "replication_pad1d_backward expects 2D (C, W) or 3D (N, C, W) input"
        )

    # ATen rejects a gradient whose dtype differs from the input rather than
    # promoting it, with this wording.
    if grad_output.dtype != self_tensor.dtype:
        raise RuntimeError(
            f"expected scalar type {_scalar_name(self_tensor.dtype)} but found "
            f"{_scalar_name(grad_output.dtype)}"
        )
    if grad_output.device != self_tensor.device:
        raise RuntimeError(
            f"expected grad_output to be on device {self_tensor.device} but found "
            f"{grad_output.device}"
        )

    # The output takes the *input's* options, not the gradient's: ATen returns
    # ``at::empty_like(self)``.
    grad_input = torch.empty(
        self_tensor.shape,
        device=self_tensor.device,
        dtype=self_tensor.dtype,
    )

    if self_tensor.is_complex():
        # Triton has no complex scalar type, and this operator never mixes the
        # real and imaginary parts: it only gathers and sums gradients along W.
        # So run the real kernel on a view where the two components are
        # separate channels: view_as_real gives (..., W, 2); slicing the last
        # axis keeps the original W layout (only the element stride changes,
        # which the kernel already takes as a parameter).
        go_real = torch.view_as_real(grad_output)
        self_real = torch.view_as_real(self_tensor)
        gi_real = torch.view_as_real(grad_input)
        for comp in (0, 1):
            _launch_replication_pad1d_backward_kernel(
                go_real[..., comp],
                self_real[..., comp],
                (left, right),
                gi_real[..., comp],
            )
        return grad_input

    return _launch_replication_pad1d_backward_kernel(
        grad_output, self_tensor, (left, right), grad_input
    )

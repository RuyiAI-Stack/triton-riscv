import torch
import triton
import triton.language as tl

ADAPTIVE_AVG_POOL2D_BLOCK = 256


@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    # Input strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    MAX_H,
    MAX_W,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode output linear index into N, C, H, W (NCHW layout assumed).
    w_idx = offsets % out_w
    tmp = offsets // out_w
    h_idx = tmp % out_h
    tmp = tmp // out_h
    c_idx = tmp % in_c
    n_idx = tmp // in_c

    # Adaptive boundaries for each output position.
    h_start = (h_idx * in_h) // out_h
    h_end = (h_idx + 1) * in_h + out_h - 1
    w_start = (w_idx * in_w) // out_w
    w_end = (w_idx + 1) * in_w + out_w - 1

    h_end = h_end // out_h
    w_end = w_end // out_w

    base = n_idx * in_stride_n + c_idx * in_stride_c
    sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in range(MAX_H):
        h_in = h_start + kh
        h_in_mask = h_in < h_end
        h_in_mask = h_in_mask & (h_in < in_h)

        # keep all masks active for rows outside the window as false
        row_mask = h_in_mask

        for kw in range(MAX_W):
            w_in = w_start + kw
            col_mask = w_in < w_end
            w_in_mask = row_mask & col_mask & (w_in < in_w)
            if_mask = mask & w_in_mask

            in_offsets = base + h_in * in_stride_h + w_in * in_stride_w
            vals = tl.load(
                input_ptr + in_offsets,
                mask=if_mask,
                other=0.0,
            )
            sum_acc += vals.to(tl.float32)

    window_size = (h_end - h_start) * (w_end - w_start)
    out_vals = sum_acc / window_size.to(tl.float32)
    tl.store(output_ptr + offsets, out_vals, mask=mask)


def adaptive_avg_pool2d(input: torch.Tensor, output_size):
    unbatched = input.ndim == 3
    if unbatched:
        input = input.unsqueeze(0)
    elif input.ndim != 4:
        raise ValueError(
            f"adaptive_avg_pool2d expects a 3D or 4D tensor; got shape {tuple(input.shape)}"
        )

    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    out_h, out_w = output_size
    if out_h < 0 or out_w < 0:
        raise RuntimeError(
            "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 but "
            f"received {{{out_h}, {out_w}}}"
        )

    if out_h == 0 or out_w == 0:
        output = torch.empty(
            (input.shape[0], input.shape[1], out_h, out_w),
            device=input.device,
            dtype=input.dtype,
        )
        return output.squeeze(0) if unbatched else output

    input = input.contiguous()
    in_n, in_c, in_h, in_w = input.shape
    if input.numel() == 0:
        output = torch.empty(
            (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
        )
        return output.squeeze(0) if unbatched else output

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    n_elements = output.numel()
    if n_elements == 0:
        return output

    max_h = (in_h + out_h - 1) // out_h + 1
    max_w = (in_w + out_w - 1) // out_w + 1

    # Keep compile-time bounds manageable and clamp to a fixed maximum for safety.
    max_h = int(max_h)
    max_w = int(max_w)

    adaptive_avg_pool2d_kernel[(triton.cdiv(n_elements, ADAPTIVE_AVG_POOL2D_BLOCK),)](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        n_elements,
        BLOCK_SIZE=ADAPTIVE_AVG_POOL2D_BLOCK,
        MAX_H=max_h,
        MAX_W=max_w,
    )

    return output.squeeze(0) if unbatched else output

import torch
import triton
import triton.language as tl


@triton.jit
def vstack_kernel(
    input_ptr,
    output_ptr,
    total_row_offset,
    total_elements,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    idx = (pid_x * BLOCK_SIZE + col_idx).to(tl.int64)
    offset_mask = idx < total_elements
    in_offset = input_ptr + idx
    row_stride_offset = total_row_offset * row_stride.to(tl.int64)
    out_offset = output_ptr + row_stride_offset + idx
    out = tl.load(in_offset, mask=offset_mask)
    tl.store(out_offset, out, mask=offset_mask)


def vstack(tensors: list):
    tensors = torch.atleast_2d(tensors)
    num_tensors = len(tensors)
    assert num_tensors > 0

    device = tensors[0].device
    dtype = tensors[0].dtype
    for tensor in tensors:
        assert (
            tensor.device == device
            and tensor.dtype == dtype
            and tensors[0].shape[1:] == tensor.shape[1:]
        )

    c_tensors = [t.contiguous() for t in tensors]
    total_rows = sum(tensor.shape[0] for tensor in c_tensors)
    output_shape = list(c_tensors[0].shape)
    output_shape[0] = total_rows
    output = torch.empty(output_shape, device=device, dtype=dtype)
    row_stride = c_tensors[0].stride(0)

    total_row_offset = 0
    for tensor in c_tensors:
        total_elements = tensor.numel()
        grid = (triton.cdiv(total_elements, 1024),)
        vstack_kernel[grid](
            tensor,
            output,
            total_row_offset,
            total_elements,
            row_stride,
            BLOCK_SIZE=1024,
        )
        total_row_offset += tensor.shape[0]
    return output

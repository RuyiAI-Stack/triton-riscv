import torch
import triton
import triton.language as tl

# masked_select_backward: scatter ``grad`` back through a boolean mask
# (grad and mask broadcast together). One cumsum per block maps each
# selected lane to its source index. Three tiers by element count:
# single-block (<= 32k), redundant-prefix rows (<= 256k), and a
# count/prefix/scatter three-kernel pass above that. The upstream CUDA
# multi-processor count is replaced by a fixed program count on CPU.
# Complex grads scatter through their real/imag parts (conj-aware).


def _bracket_next_power_of_2(n, lower, upper):
    return min(max(triton.next_power_of_2(n), lower), upper)


@triton.jit
def _masked_select_backward_single_pass_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    total_count_ptr,
    grad_numel,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    grad_offsets = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    tl.store(total_count_ptr, tl.sum(selected.to(tl.int32), axis=0))
    source_valid = valid & selected & (grad_offsets < grad_numel)
    values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
    tl.store(out_ptr + offsets, tl.where(selected, values, 0), mask=valid)


@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_redundant_prefix_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    total_count_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)
    lane = tl.arange(0, BLOCK_SIZE)
    # For medium tensors, redundant mask reads are cheaper than allocating and
    # synchronizing a global prefix-sum workspace.
    grad_advance = 0
    prefix_offsets = lane
    for _ in range(0, start_block):
        grad_advance += tl.sum(tl.load(mask_ptr + prefix_offsets).to(tl.int32), axis=0)
        prefix_offsets += BLOCK_SIZE
    offsets = start_block * BLOCK_SIZE + lane
    for _ in range(start_block, last_block_id):
        selected = tl.load(mask_ptr + offsets).to(tl.int1)
        selected_int = selected.to(tl.int32)
        grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
        source_valid = selected & (grad_offsets < grad_numel)
        values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
        tl.store(out_ptr + offsets, tl.where(selected, values, 0))
        grad_advance += tl.sum(selected_int, axis=0)
        offsets += BLOCK_SIZE
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    selected_int = selected.to(tl.int32)
    grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
    source_valid = valid & selected & (grad_offsets < grad_numel)
    values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
    tl.store(out_ptr + offsets, tl.where(selected, values, 0), mask=valid)
    if last_block_id == num_blocks - 1:
        tl.store(total_count_ptr, grad_advance + tl.sum(selected_int, axis=0))


@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_count_kernel(
    mask_ptr,
    part_sums_ptr,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)
    count = 0
    for _ in range(start_block, last_block_id):
        count += tl.sum(tl.load(mask_ptr + offsets).to(tl.int32), axis=0)
        offsets += BLOCK_SIZE
    valid = offsets < n_elements
    count += tl.sum(
        tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int32), axis=0
    )
    tl.store(part_sums_ptr + row_id, count)


@triton.jit(do_not_specialize=["num_programs"])
def _masked_select_backward_prefix_kernel(
    part_sums_ptr,
    total_count_ptr,
    num_programs,
    NP_BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, NP_BLOCK)
    valid = offsets < num_programs
    counts = tl.load(part_sums_ptr + offsets, mask=valid, other=0)
    tl.store(total_count_ptr, tl.sum(counts, axis=0))
    prefixes = tl.cumsum(counts, axis=0) - counts
    tl.store(part_sums_ptr + offsets, prefixes, mask=valid)


@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_scatter_kernel(
    grad_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block_id = min(num_blocks - 1, start_block + num_blocks_per_row - 1)
    grad_advance = tl.load(part_sums_ptr + row_id)
    for _ in range(start_block, last_block_id):
        selected = tl.load(mask_ptr + offsets).to(tl.int1)
        selected_int = selected.to(tl.int32)
        grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
        source_valid = selected & (grad_offsets < grad_numel)
        values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
        tl.store(out_ptr + offsets, tl.where(selected, values, 0))
        grad_advance += tl.sum(selected_int, axis=0)
        offsets += BLOCK_SIZE
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
    selected_int = selected.to(tl.int32)
    grad_offsets = grad_advance + tl.cumsum(selected_int, axis=0) - 1
    source_valid = valid & selected & (grad_offsets < grad_numel)
    values = tl.load(grad_ptr + grad_offsets, mask=source_valid, other=0)
    tl.store(
        out_ptr + offsets,
        tl.where(selected, values, 0),
        mask=valid,
    )


# Fixed stand-in for the CUDA multi-processor count on CPU (correctness is
# independent of this value; it only splits blocks across programs).
_NUM_PROGRAMS = 8


def _masked_select_backward_real(grad, mask, out, validate=True):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grad = grad.contiguous()
    total_count = torch.empty((), dtype=torch.int64, device=out.device)
    if n_elements <= 32768:
        block_size = triton.next_power_of_2(n_elements)
        _masked_select_backward_single_pass_kernel[(1,)](
            grad,
            mask,
            out,
            total_count,
            grad.numel(),
            n_elements,
            BLOCK_SIZE=block_size,
        )
        if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
            raise RuntimeError("Number of elements of source < number of ones in mask")
        return out
    block_size = _bracket_next_power_of_2(n_elements, 128, 4096)
    num_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(num_blocks, _NUM_PROGRAMS)
    num_blocks_per_row = triton.cdiv(num_blocks, num_programs)
    num_programs = triton.cdiv(num_blocks, num_blocks_per_row)
    if n_elements <= 262144:
        _masked_select_backward_redundant_prefix_kernel[(num_programs,)](
            grad,
            mask,
            out,
            total_count,
            grad.numel(),
            n_elements,
            num_blocks,
            num_blocks_per_row,
            BLOCK_SIZE=block_size,
        )
        if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
            raise RuntimeError("Number of elements of source < number of ones in mask")
        return out
    scan_block = triton.next_power_of_2(num_programs)
    index_dtype = torch.int32 if n_elements < 2**31 else torch.int64
    part_sums = torch.empty(num_programs, dtype=index_dtype, device=out.device)
    _masked_select_backward_count_kernel[(num_programs,)](
        mask,
        part_sums,
        n_elements,
        num_blocks,
        num_blocks_per_row,
        BLOCK_SIZE=block_size,
    )
    _masked_select_backward_prefix_kernel[(1,)](
        part_sums,
        total_count,
        num_programs,
        NP_BLOCK=scan_block,
    )
    _masked_select_backward_scatter_kernel[(num_programs,)](
        grad,
        mask,
        part_sums,
        out,
        grad.numel(),
        n_elements,
        num_blocks,
        num_blocks_per_row,
        BLOCK_SIZE=block_size,
    )
    if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
        raise RuntimeError("Number of elements of source < number of ones in mask")
    return out


def _broadcast_views(input, mask):
    ndim = max(input.ndim, mask.ndim)
    input_shape = (1,) * (ndim - input.ndim) + tuple(input.shape)
    mask_shape = (1,) * (ndim - mask.ndim) + tuple(mask.shape)
    input_strides = (0,) * (ndim - input.ndim) + tuple(input.stride())
    mask_strides = (0,) * (ndim - mask.ndim) + tuple(mask.stride())
    shape = []
    for left, right in zip(input_shape, mask_shape):
        if left == right or right == 1:
            shape.append(left)
        elif left == 1:
            shape.append(right)
        else:
            raise RuntimeError("input and mask shapes cannot be broadcast together")
    input_strides = tuple(
        0 if size == 1 and output != 1 else stride
        for size, output, stride in zip(input_shape, shape, input_strides)
    )
    mask_strides = tuple(
        0 if size == 1 and output != 1 else stride
        for size, output, stride in zip(mask_shape, shape, mask_strides)
    )
    return input.as_strided(shape, input_strides), mask.as_strided(shape, mask_strides)


def masked_select_backward(grad, input, mask):
    """Scatter ``grad`` into ``input``'s shape through the broadcast ``mask`` (0 elsewhere)."""
    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_scatter_ only supports boolean masks, but got {mask.dtype}"
        )
    if grad.dtype != input.dtype:
        raise RuntimeError(
            "masked_scatter: expected self and source to have same dtypes but got "
            f"{input.dtype} and {grad.dtype}"
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("grad, input, and mask must be on the same device")
    input_expanded, mask_expanded = _broadcast_views(input, mask)
    mask_contiguous = mask_expanded.contiguous()
    result = torch.empty_like(input_expanded, memory_format=torch.preserve_format)
    if result.numel() == 0:
        return result
    if input.is_complex():
        physical_grad = grad.conj() if grad.is_conj() else grad
        grad_parts = torch.view_as_real(physical_grad)
        real = torch.empty(
            input_expanded.shape, dtype=grad_parts.dtype, device=grad.device
        )
        imag = torch.empty_like(real)
        _masked_select_backward_real(grad_parts[..., 0], mask_contiguous, real)
        _masked_select_backward_real(
            -grad_parts[..., 1] if grad.is_conj() else grad_parts[..., 1],
            mask_contiguous,
            imag,
            validate=False,
        )
        result_parts = torch.view_as_real(result)
        result_parts[..., 0].copy_(real)
        result_parts[..., 1].copy_(imag)
        return result
    else:
        contiguous_result = (
            result
            if result.is_contiguous()
            else torch.empty(
                input_expanded.shape, dtype=input.dtype, device=input.device
            )
        )
        _masked_select_backward_real(grad, mask_contiguous, contiguous_result)
    if contiguous_result is not result:
        result.copy_(contiguous_result)
    return result

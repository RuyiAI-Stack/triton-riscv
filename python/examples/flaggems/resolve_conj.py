import torch
import triton
import triton.language as tl


@triton.jit
def resolve_conj_kernel_1d(
    x_real_ptr,
    x_img_ptr,
    output_ptr,
    n_elements_total,
    is_conj: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_total

    # x.real/x.imag data_ptr points to start of complex storage with stride 2
    # Pointer arithmetic is in float32 units
    real = tl.load(x_real_ptr + 2 * offsets, mask=mask)
    imag = tl.load(x_img_ptr + 2 * offsets, mask=mask)

    output_real_offsets = 2 * offsets
    output_img_offsets = 2 * offsets + 1

    if is_conj:
        tl.store(output_ptr + output_real_offsets, real, mask=mask)
        tl.store(output_ptr + output_img_offsets, -imag, mask=mask)
    else:
        tl.store(output_ptr + output_real_offsets, real, mask=mask)
        tl.store(output_ptr + output_img_offsets, imag, mask=mask)


@triton.jit
def resolve_conj_kernel_2d_strided(
    x_real_ptr,
    x_img_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    is_conj: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col_block = tl.program_id(axis=1)
    col_start = pid_col_block * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    base_offset = pid_row * stride_row + col_offsets * stride_col
    mask = col_mask & (pid_row < n_rows)

    real = tl.load(x_real_ptr + 2 * base_offset, mask=mask)
    imag = tl.load(x_img_ptr + 2 * base_offset, mask=mask)

    output_base_offset = base_offset * 2
    if is_conj:
        tl.store(output_ptr + output_base_offset, real, mask=mask)
        tl.store(output_ptr + output_base_offset + 1, -imag, mask=mask)
    else:
        tl.store(output_ptr + output_base_offset, real, mask=mask)
        tl.store(output_ptr + output_base_offset + 1, imag, mask=mask)


@triton.jit
def resolve_conj_kernel_large_2d(
    x_real_ptr,
    x_img_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    is_conj: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)
    row_offsets = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = pid_col * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    base_offsets = row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    mask = row_mask[:, None] & col_mask[None, :]

    real = tl.load(x_real_ptr + 2 * base_offsets, mask=mask)
    imag = tl.load(x_img_ptr + 2 * base_offsets, mask=mask)

    output_base_offsets = base_offsets * 2
    if is_conj:
        tl.store(output_ptr + output_base_offsets, real, mask=mask)
        tl.store(output_ptr + output_base_offsets + 1, -imag, mask=mask)
    else:
        tl.store(output_ptr + output_base_offsets, real, mask=mask)
        tl.store(output_ptr + output_base_offsets + 1, imag, mask=mask)


def resolve_conj_triton(x: torch.Tensor, is_conj: bool) -> torch.Tensor:
    is_complex = x.is_complex()

    if not is_conj and not is_complex:
        return x.clone()
    if not is_complex:
        return x.clone()

    output = torch.empty_like(x)

    if x.dtype == torch.complex64:
        x_real = x.real
        x_img = x.imag
        output_view = output.view(torch.float32)
        shape = x.shape
        n_elements_total = x.numel()

        if len(shape) == 2:
            rows, cols = shape
            if rows * cols > 1000000:
                stride_row = x.stride(0)
                stride_col = x.stride(1)
                BLOCK_SIZE_COLS = 128
                grid_rows = rows
                grid_cols = triton.cdiv(cols, BLOCK_SIZE_COLS)
                grid = (grid_rows, grid_cols)
                resolve_conj_kernel_2d_strided[grid](
                    x_real,
                    x_img,
                    output_view,
                    rows,
                    cols,
                    stride_row,
                    stride_col,
                    is_conj,
                    BLOCK_SIZE_COLS,
                )
            else:
                BLOCK_SIZE = 256
                grid = (triton.cdiv(n_elements_total, BLOCK_SIZE),)
                resolve_conj_kernel_1d[grid](
                    x_real,
                    x_img,
                    output_view,
                    n_elements_total,
                    is_conj,
                    BLOCK_SIZE,
                )
        elif len(shape) == 3:
            BLOCK_SIZE = min(1024, n_elements_total)
            grid = (triton.cdiv(n_elements_total, BLOCK_SIZE),)
            resolve_conj_kernel_1d[grid](
                x_real,
                x_img,
                output_view,
                n_elements_total,
                is_conj,
                BLOCK_SIZE,
            )
        else:
            BLOCK_SIZE = 1024 if n_elements_total > 1000000 else 256
            grid = (triton.cdiv(n_elements_total, BLOCK_SIZE),)
            resolve_conj_kernel_1d[grid](
                x_real,
                x_img,
                output_view,
                n_elements_total,
                is_conj,
                BLOCK_SIZE,
            )
        return output
    else:
        if is_conj:
            return torch.conj(x)
        return x.clone()


def resolve_conj(A: torch.Tensor):
    if A.is_conj():
        if len(A.shape) in (2, 3):
            return resolve_conj_triton(A, is_conj=True)
        else:
            # resolve_conj: return a clone with conj bit cleared
            return A.clone()
    else:
        return A

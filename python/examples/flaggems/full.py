import math

import torch
import triton
import triton.language as tl


ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
)


def check_dtype(fill_value, dtype, device):
    if isinstance(fill_value, bool):
        if dtype != torch.bool:
            fill_value = int(fill_value)
    elif (
        dtype in ALL_INT_DTYPES
        and (fill_value < torch.iinfo(dtype).min or fill_value > torch.iinfo(dtype).max)
    ) or (
        dtype in ALL_FLOAT_DTYPES
        and not (math.isinf(fill_value) or math.isnan(fill_value))
        and (fill_value < torch.finfo(dtype).min or fill_value > torch.finfo(dtype).max)
    ):
        raise RuntimeError(
            f"value cannot be converted to type {dtype} without overflow"
        )
    if dtype == torch.float64:
        fill_value = torch.tensor(fill_value, dtype=dtype, device=device)
    return fill_value


@triton.jit
def full_kernel(
    out_ptr,
    fill_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    fill_val = tl.load(fill_ptr)
    tl.store(out_ptr + offsets, fill_val, mask=mask)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = torch.bool
        elif isinstance(fill_value, int):
            dtype = torch.int64
        else:
            dtype = torch.get_default_dtype()
    else:
        fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    if isinstance(fill_value, torch.Tensor):
        fill_tensor = fill_value.to(device=device, dtype=dtype)
    else:
        fill_tensor = torch.tensor(fill_value, device=device, dtype=dtype)

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    full_kernel[(num_blocks,)](out, fill_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

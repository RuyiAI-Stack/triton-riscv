import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "cols", "stride_row", "stride_col"])
def _stacked_wrap_load(input_ptr, output_ptr, rows, cols, stride_row, stride_col):
    output_rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    input_rows = output_rows % rows
    columns = tl.arange(0, 4)
    offsets = input_rows[:, None] * stride_row + columns[None, :] * stride_col
    values = tl.load(input_ptr + offsets, columns[None, :] < cols, other=0.0)
    tl.store(output_ptr + output_rows[:, None] * 4 + columns[None, :], values)


@pytest.mark.parametrize("layout", ["contiguous", "row", "column", "both", "sliced"])
def test_stacked_zero_stride(device, layout):
    if layout == "row":
        source = torch.arange(3, dtype=torch.float32, device=device).expand(5, 3)
    elif layout == "column":
        source = torch.arange(5, dtype=torch.float32, device=device)[:, None].expand(
            5, 3
        )
    elif layout == "both":
        source = torch.tensor([[7.0]], device=device).expand(5, 3)
    elif layout == "sliced":
        source = torch.arange(40, dtype=torch.float32, device=device).reshape(5, 8)[
            :, 1:7:2
        ]
    else:
        source = torch.arange(15, dtype=torch.float32, device=device).reshape(5, 3)
    output = torch.full((8, 4), float("nan"), device=device)
    _stacked_wrap_load[(2,)](source, output, 5, 3, *source.stride())
    expected = torch.zeros_like(output)
    expected[:, :3] = source[torch.arange(8, device=device) % 5]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

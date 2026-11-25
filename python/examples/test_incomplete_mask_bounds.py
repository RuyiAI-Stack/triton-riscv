import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["columns", "period", "enable"])
def _masked_padded_load(input_ptr, output_ptr, columns, period, enable):
    rows = tl.arange(0, 4)
    cols = tl.arange(0, 4)
    offsets = rows[:, None] * 4 + cols[None, :]
    # The remainder prevents a structured row bound. The column bound and
    # scalar condition must still be preserved when analysis falls back.
    row_mask = (rows[:, None] % period) < period
    mask = (cols[None, :] < columns) & row_mask & (enable != 0)
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values)


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("enable", [False, True])
@pytest.mark.parametrize("padding", [0.0, float("nan")], ids=["zero", "nan"])
def test_incomplete_mask_bounds(device, columns, enable, padding):
    logical = torch.arange(1, 4 * columns + 1, device=device, dtype=torch.float32)
    logical = logical.reshape(4, columns)
    padded = torch.full((4, 4), padding, device=device, dtype=torch.float32)
    padded[:, :columns] = logical
    output = torch.full_like(padded, float("nan"))

    _masked_padded_load[(1,)](padded, output, columns, 2, int(enable))

    expected = torch.zeros_like(padded)
    if enable:
        expected[:, :columns] = logical
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

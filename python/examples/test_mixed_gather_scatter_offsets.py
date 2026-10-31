"""Preserve implicit structured ranges when adding loaded gather/scatter indices."""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _mixed_offsets_kernel(
    source,
    indices,
    output,
    count,
    base,
    STRIDE: tl.constexpr,
    REVERSE_ADD: tl.constexpr,
    NEGATIVE: tl.constexpr,
    SCATTER: tl.constexpr,
    SPLIT_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(indices + lane, mask=lane < count, other=0)
    group = count - 1 - lane if NEGATIVE else lane
    structured = group * STRIDE + base
    offset = index + structured if REVERSE_ADD else structured + index
    target = output if SCATTER else source
    if SPLIT_PTR:
        pointer = (
            target + index + structured if REVERSE_ADD else target + structured + index
        )
    else:
        pointer = target + offset
    if SCATTER:
        value = tl.load(source + lane, mask=lane < count, other=0)
        tl.store(pointer, value, mask=lane < count)
    else:
        value = tl.load(pointer, mask=lane < count, other=-1)
        tl.store(output + lane, value, mask=lane < count)


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("split_ptr", [False, True])
@pytest.mark.parametrize("reverse_add", [False, True])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("stride", [0, 1, 4])
@pytest.mark.parametrize("count,base", [(17, 0), (65, 13)])
def test_mixed_range_and_loaded_offsets(
    scatter, split_ptr, reverse_add, negative, stride, count, base
):
    lane = torch.arange(count, dtype=torch.int32)
    indices = lane.remainder(stride) if stride else lane.flip(0)
    group = count - 1 - lane if negative else lane
    offsets = group.long() * stride + base + indices.long()
    # Keep a checked guard region so an address-analysis regression fails an
    # assertion instead of writing into a neighbouring allocation.
    extent = (base + max(1, stride) * count) * max(1, stride) + 64
    if scatter:
        source = torch.arange(count, dtype=torch.float32) + 1
        output = torch.full((extent,), -1.0)
        expected = output.clone()
        expected[offsets] = source
    else:
        source = torch.arange(extent, dtype=torch.float32) + 1
        output = torch.empty(count)
        expected = source[offsets]
    _mixed_offsets_kernel[(triton.cdiv(count, 16),)](
        source,
        indices,
        output,
        count,
        base,
        STRIDE=stride,
        REVERSE_ADD=reverse_add,
        NEGATIVE=negative,
        SCATTER=scatter,
        SPLIT_PTR=split_ptr,
        BLOCK=16,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@triton.jit
def _mixed_rank2_kernel(
    source,
    indices,
    output,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    SPLIT_PTR: tl.constexpr,
    INDEX_COLUMNS: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    columns = tl.arange(0, COLS)
    if INDEX_COLUMNS:
        index = tl.load(indices + columns)[None, :]
    else:
        index = tl.load(indices + rows)[:, None]
    structured = (rows[:, None] * COLS + columns[None, :]) * 4
    offset = structured + index
    pointer = source + structured + index if SPLIT_PTR else source + offset
    values = tl.load(pointer)
    tl.store(output + rows[:, None] * COLS + columns[None, :], values)


@pytest.mark.parametrize("split_ptr", [False, True])
@pytest.mark.parametrize("index_columns", [False, True])
def test_mixed_rank2_offset_fallback(split_ptr, index_columns):
    rows, columns = 4, 8
    indices = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
    if index_columns:
        indices = indices.repeat(2)
    source = torch.arange(rows * columns * 4, dtype=torch.float32)
    output = torch.empty(rows, columns)
    expected = source.reshape(rows, columns, 4)[
        torch.arange(rows)[:, None],
        torch.arange(columns)[None, :],
        (indices[None, :] if index_columns else indices[:, None]).long(),
    ]
    _mixed_rank2_kernel[(1,)](
        source,
        indices,
        output,
        ROWS=rows,
        COLS=columns,
        SPLIT_PTR=split_ptr,
        INDEX_COLUMNS=index_columns,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@triton.jit
def _mixed_singleton_base_kernel(
    source,
    indices,
    output,
    SPLIT_PTR: tl.constexpr,
    REVERSE_ADD: tl.constexpr,
    SCATTER: tl.constexpr,
):
    columns = tl.arange(0, 8)[None, :]
    index = tl.load(indices + columns)
    structured = columns * 4 + 13
    target = output if SCATTER else source
    if SPLIT_PTR:
        pointer = (
            target + index + structured if REVERSE_ADD else target + structured + index
        )
    else:
        offset = index + structured if REVERSE_ADD else structured + index
        pointer = target + offset
    if SCATTER:
        values = tl.load(source + columns)
        tl.store(pointer, values)
    else:
        values = tl.load(pointer)
        tl.store(output + columns, values)


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("reverse_add", [False, True])
@pytest.mark.parametrize("split_ptr", [False, True])
def test_mixed_singleton_base_offset(scatter, reverse_add, split_ptr):
    indices = torch.tensor([3, 0, 2, 1, 3, 0, 2, 1], dtype=torch.int32)
    offsets = torch.arange(8) * 4 + 13 + indices
    # The extra allocation contains the incorrectly doubled base offset too.
    if scatter:
        source = torch.arange(8, dtype=torch.float32) + 1
        output = torch.full((128,), -1.0)
        expected = output.clone()
        expected[offsets] = source
    else:
        source = torch.arange(128, dtype=torch.float32)
        output = torch.empty(1, 8)
        expected = source[offsets].unsqueeze(0)

    _mixed_singleton_base_kernel[(1,)](
        source,
        indices,
        output,
        SPLIT_PTR=split_ptr,
        REVERSE_ADD=reverse_add,
        SCATTER=scatter,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@triton.jit
def _chained_expanded_indices_kernel(
    source,
    indices,
    indices2,
    output,
    BASE: tl.constexpr,
    INDEX_COLUMNS: tl.constexpr,
    SCATTER: tl.constexpr,
):
    lane = tl.arange(0, 8)
    index = tl.load(indices + lane)
    index2 = tl.load(indices2 + lane)
    if INDEX_COLUMNS:
        index = index[None, :]
        index2 = index2[None, :]
        lane = lane[None, :]
    else:
        index = index[:, None]
        index2 = index2[:, None]
        lane = lane[:, None]
    target = output if SCATTER else source
    pointer = target + BASE + index + index2
    if SCATTER:
        values = tl.load(source + lane)
        tl.store(pointer, values)
    else:
        values = tl.load(pointer)
        tl.store(output + lane, values)


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("index_columns", [False, True])
@pytest.mark.parametrize("base", [0, 13])
def test_chained_expanded_indices(scatter, index_columns, base):
    indices = torch.arange(8, dtype=torch.int32) * 8 + 4
    indices2 = torch.tensor([-3, 0, -2, 1, -3, 0, -2, 1], dtype=torch.int32)
    # Addresses are positive and distinct, including with negative i32 indices.
    offsets = base + indices.long() + indices2.long()
    if scatter:
        source = torch.arange(8, dtype=torch.float32) + 1
        output = torch.full((128,), -1.0)
        expected = output.clone()
        expected[offsets] = source
    else:
        shape = (1, 8) if index_columns else (8, 1)
        source = torch.arange(128, dtype=torch.float32)
        output = torch.empty(shape)
        expected = source[offsets].reshape(shape)

    _chained_expanded_indices_kernel[(1,)](
        source,
        indices,
        indices2,
        output,
        BASE=base,
        INDEX_COLUMNS=index_columns,
        SCATTER=scatter,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)

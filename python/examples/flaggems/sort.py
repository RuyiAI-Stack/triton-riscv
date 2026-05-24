import torch
import triton
import triton.language as tl

from .topk import _get_finfo_val, _get_iinfo_val, argsort


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(
    x: tl.tensor, descending: tl.constexpr = False
):
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


@triton.jit
def compute_global_hist_kernel(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):
                n_offsets = n_start + tl.arange(0, TILE_N)
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask
                matches = tl.where(mask, (bin_indices[:, None] == key), False)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )


@triton.jit
def sweep(
    arr_ptr,
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr,
    excumsum_bins_ptr,
    status_ptr,
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1

    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask
    if associate_arr_ptr is not None:
        associate_arr = tl.load(
            associate_arr_ptr + pid_m * N + n_offsets, mask=mask
        )

    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = (
                pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            )
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += pack1 & v_mask
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

        local_ex_cumsum = tl.cumsum(matches.to(tl.uint32), axis=0) - matches
        ex_cumsum_in_bin = exclusive_prefix + local_ex_cumsum

        ex_cumsum_bins = tl.load(
            excumsum_bins_ptr
            + pid_m * (n_passes * r)
            + pass_id * r
            + bin_index
        )
        pos = ex_cumsum_bins + ex_cumsum_in_bin

        tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
        if associate_arr_ptr is not None:
            tl.store(
                associate_out_ptr + pid_m * N + pos,
                associate_arr,
                mask=matches,
            )


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (m * grid_n, 1, 1)

    global_hist = torch.zeros(
        (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
    )
    compute_global_hist_kernel[grid_for_global_hist](
        arr,
        global_hist,
        n_passes,
        m,
        n,
        tiles_n_per_cta,
        TILE_N,
        TILE_R,
        k_bits,
        descending,
    )
    ex_cumsum_bins = torch.cumsum(global_hist, -1) - global_hist
    ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

    arr_in = torch.clone(arr)
    indices_in = (
        torch.arange(0, n, dtype=torch.int64, device=arr_in.device)
        .broadcast_to(arr.shape)
        .contiguous()
    )
    arr_out = torch.empty_like(arr)
    indices_out = torch.empty_like(indices_in)

    TILE_R = 8
    grid_r = triton.cdiv(num_bins, TILE_R)
    TILE_N = 2048
    grid_n = triton.cdiv(n, TILE_N)
    grid_for_sweep = (m * grid_n, grid_r)

    status = torch.empty(
        (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
    )

    for i in range(0, n_passes):
        bit_offset = i * k_bits
        status.zero_()
        sweep[grid_for_sweep](
            arr_in,
            indices_in,
            arr_out,
            indices_out,
            ex_cumsum_bins,
            status,
            n_passes,
            i,
            bit_offset,
            m,
            n,
            grid_n,
            TILE_N,
            TILE_R,
            k_bits,
            descending,
        )
        arr_in, arr_out = arr_out, arr_in
        indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


@triton.jit
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(
            in_ptr.dtype.element_ty, return_max=not DESCENDING
        )
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
    else:
        mask_val = _get_iinfo_val(
            in_ptr.dtype.element_ty, return_max=not DESCENDING
        )
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)

    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index

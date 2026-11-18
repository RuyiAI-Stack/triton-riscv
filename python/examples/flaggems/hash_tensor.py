"""hash_tensor - XOR reduction of widened 64-bit element patterns.

Reference semantics (aten / flag_gems; registered and CPU-callable on this
build as torch.ops.aten.hash_tensor, the underscored torch._hash_tensor
spelling is not):
  Each element is widened to a 64-bit pattern -- floats are cast to
  float64 and bitcast, signed integers are sign-extended to int64, and
  uint8 / bool are zero-extended -- and the patterns are XOR-combined
  along the reduction dims into a torch.uint64 result.

Port notes (triton-riscv, vs the upstream flag_gems file):
- tl.reduce over a uint64 accumulator corrupts the heap on this backend
  (probed: native crash at process exit). XOR is bitwise, so the
  accumulator and the reduction run on int64 -- bit-identical results --
  and only the final lane is bitcast to uint64 at the store.
- One program per output row replaces the upstream 2-D tile / split-N
  launch machinery (and the libentry decorator): the CPU backend has no
  use for the occupancy tuning and the port keeps a single kernel shape.
- Upstream accepts any mode argument but the reference only defines
  mode == 0; a different mode raises RuntimeError("Unknown hash_tensor
  mode: ...") on both sides (pinned in the tests).
- Empty-tensor edges follow the upstream module, not aten: a zero-size
  reduction dim hashes to zero instead of aten's IndexError, and a full
  reduction of a numel == 0 tensor returns a zero scalar instead of
  aten's RuntimeError. Divergent cases are pinned without the aten
  oracle.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024

_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}


@triton.jit
def _xor_combine(a, b):
    return a ^ b


@triton.jit
def _to_bits(val, dtype_code: tl.constexpr):
    # Widen each element to an int64 bit pattern (see module docstring for
    # why the reduction is int64 and not uint64).
    if dtype_code == 0:  # float32 -> float64 bits
        u = val.to(tl.float64).to(tl.int64, bitcast=True)
    elif dtype_code == 1:  # float16 -> float64 bits
        u = val.to(tl.float64).to(tl.int64, bitcast=True)
    elif dtype_code == 2:  # bfloat16 -> float64 bits
        u = val.to(tl.float64).to(tl.int64, bitcast=True)
    elif dtype_code == 3:  # float64 bits
        u = val.to(tl.int64, bitcast=True)
    elif dtype_code == 4:  # int64 bits
        u = val.to(tl.int64, bitcast=True)
    elif dtype_code == 5:  # int32 -> sign-extend int64 -> bits
        u = val.to(tl.int64)
    elif dtype_code == 6:  # int16 -> sign-extend int64 -> bits
        u = val.to(tl.int64)
    elif dtype_code == 7:  # int8 -> sign-extend int64 -> bits
        u = val.to(tl.int64)
    elif dtype_code == 8:  # uint8 -> zero-extend
        u = val.to(tl.int64)
    else:  # 9: bool -> zero-extend
        u = val.to(tl.int64)
    return u


@triton.jit
def _hash_row_kernel(
    x_ptr,
    out_ptr,
    N,
    row_stride,
    col_stride,
    dtype_code: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for i in range(0, N, BLOCK):
        idx = i + offs
        m = idx < N
        v = tl.load(x_ptr + pid * row_stride + idx * col_stride, mask=m, other=0)
        u = _to_bits(v, dtype_code)
        acc = acc ^ tl.where(m, u, 0)
    r = tl.reduce(acc, 0, _xor_combine)
    tl.store(out_ptr + pid, r.to(tl.uint64, bitcast=True))


def hash_tensor(x, dim, keepdim=False, mode=0):
    """XOR-hash of 64-bit element patterns along ``dim``.

    Args:
        x: input tensor (float / signed-integer / uint8 / bool)
        dim: dims to reduce (int, list of ints, or an empty list for a
            full reduction)
        keepdim: keep reduced dims with size 1
        mode: hash mode, only 0 is defined
    Returns:
        uint64 tensor of hashes
    """
    logger.debug("GEMS HASH_TENSOR")

    if mode != 0:
        raise RuntimeError(f"Unknown hash_tensor mode: {mode}")

    assert x.dtype in _DTYPE_CODES, (
        f"hash_tensor: unsupported dtype {x.dtype}"
    )
    dtype_code = _DTYPE_CODES[x.dtype]
    device = x.device

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)

    if x.ndim == 0:
        dim = []

    dim = [(d % x.ndim) if x.ndim > 0 else 0 for d in dim]
    dim = sorted(set(dim))

    output_shape = []
    for i in range(x.ndim):
        if i not in dim:
            output_shape.append(x.shape[i])
        elif keepdim:
            output_shape.append(1)

    reduce_size = 1
    for d in dim:
        reduce_size *= x.shape[d]

    output_numel = 1
    for s in output_shape:
        output_numel *= s

    M = output_numel
    N = reduce_size

    # Logical (M, N) matrix: kept axes -> rows, reduced axes -> cols.
    # reshape returns a view when collapsible; otherwise it materializes a
    # compact copy, so the kernel only ever sees a plain 2-D layout.
    perm = [i for i in range(x.ndim) if i not in dim] + dim
    xperm = x if perm == list(range(x.ndim)) else x.permute(perm)
    xr = xperm.reshape(M, N)

    output = torch.empty(M, dtype=torch.uint64, device=device)

    if M > 0:
        _hash_row_kernel[(M,)](
            xr,
            output,
            N,
            xr.stride(0),
            xr.stride(1),
            dtype_code,
            BLOCK=_BLOCK,
        )

    if len(output_shape) == 0:
        output = output.reshape([])
    else:
        output = output.reshape(output_shape)

    return output

import importlib.util
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import torch


def write_atomic(
    path_: str,
    content: str,
    make_dirs: bool = False,
    encoding: str = "utf-8",
) -> None:
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = (
        path.parent / f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    with tmp_path.open("wt", encoding=encoding) as f:
        f.write(content)
    tmp_path.replace(path)


def _code_cache_dir():
    d = Path.home() / ".flaggems" / "code_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _generate_imports(code):
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def _generate_gather_kernel(
    inp_rank, idx_dims, index_rank, bcast_pos, kernel_name, code
):
    """Fully-strided 2D gather kernel: index tensor ``i`` applies to input
    dim ``idx_dims[i]`` at arbitrary positions/strides, and the output is
    written through its own strides with the broadcast subspace placed at
    ``bcast_pos`` -- no input permute and no output post-permute needed."""
    indices_len = len(idx_dims)
    slice_dims = [d for d in range(inp_rank) if d not in idx_dims]
    out_rank = index_rank + len(slice_dims)
    bcast_out = list(range(bcast_pos, bcast_pos + index_rank))
    rest = [p for p in range(out_rank) if p not in bcast_out]
    slice_out = dict(zip(slice_dims, rest))

    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    input_ptr,\n"
    for i in range(indices_len):
        code += f"    indices{i}_ptr,\n"
    code += "    out_ptr,\n"
    for i in range(inp_rank):
        code += f"    input_shape{i},\n"
    for j in range(index_rank):
        code += f"    indices0_shape{j},\n"
    for i in range(inp_rank):
        code += f"    input_stride{i},\n"
    for i in range(indices_len):
        for j in range(index_rank):
            code += f"    indices{i}_stride{j},\n"
    for i in range(out_rank):
        code += f"    out_stride{i},\n"
    code += "    M,\n"
    code += "    N,\n"
    code += "    BLOCK_SIZE0: tl.constexpr,\n"
    code += "    BLOCK_SIZE1: tl.constexpr,\n"
    code += "):\n"
    code += "    pid0 = tl.program_id(axis=0)\n"
    code += "    pid1 = tl.program_id(axis=1)\n"
    code += "    offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]\n"
    if slice_dims:
        code += (
            "    offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]\n"
        )
    else:
        code += "    offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]\n"
    code += "\n"
    code += "    cur_idx = offset0\n"
    for j in range(index_rank - 1, -1, -1):
        code += f"    idx_coord{j} = cur_idx % indices0_shape{j}\n"
        code += f"    cur_idx = cur_idx // indices0_shape{j}\n"
    code += "\n"
    if slice_dims:
        code += "    cur_idx = offset1\n"
        for d in reversed(slice_dims):
            code += f"    slice_coord{d} = cur_idx % input_shape{d}\n"
            code += f"    cur_idx = cur_idx // input_shape{d}\n"
        code += "\n"
    code += "    mask0 = offset0 < M\n"
    for i in range(indices_len):
        dim = idx_dims[i]
        comp = " + ".join(
            [f"idx_coord{j} * indices{i}_stride{j}" for j in range(index_rank)]
        )
        code += f"    cur_index{i} = tl.load(indices{i}_ptr + {comp}, mask=mask0, other=0)\n"
        # wrap negative indices in-kernel (Python indexing semantics)
        code += (
            f"    cur_index{i} = tl.where(cur_index{i} < 0, "
            f"cur_index{i} + input_shape{dim}, cur_index{i})\n"
        )
    code += "\n"
    # post-wrap out-of-range indices (UB under the "unsafe" contract) skip
    # the store instead of scribbling a wrong row: the masked load never
    # dereferences them
    index_mask = " & ".join(
        [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{idx_dims[i]})"
            for i in range(indices_len)
        ]
    )
    code += f"    index_mask = {index_mask}\n"
    code += "    mask1 = offset1 < N\n"
    code += "    mask = index_mask & mask0 & mask1\n"
    code += "\n"
    comp = " + ".join(
        [f"cur_index{i} * input_stride{idx_dims[i]}" for i in range(indices_len)]
    )
    comp += "".join([f" + slice_coord{d} * input_stride{d}" for d in slice_dims])
    code += f"    input_offset = {comp}\n"
    comp = " + ".join(
        [f"idx_coord{r} * out_stride{bcast_pos + r}" for r in range(index_rank)]
    )
    comp += "".join(
        [f" + slice_coord{d} * out_stride{slice_out[d]}" for d in slice_dims]
    )
    code += f"    out_offset = {comp}\n"
    code += "\n"
    code += "    cur_value = tl.load(input_ptr + input_offset, mask=mask)\n"
    code += "    tl.store(out_ptr + out_offset, cur_value, mask=mask)\n\n"
    return code


def _generate_wrapper(
    inp_rank, idx_dims, index_rank, bcast_pos, wrapper_name, kernel_name, code
):
    indices_len = len(idx_dims)
    slice_dims = [d for d in range(inp_rank) if d not in idx_dims]
    out_rank = index_rank + len(slice_dims)

    code += f"def {wrapper_name}(input, indices, out):\n"
    code += "    input_shape = input.shape\n"
    code += "    input_stride = input.stride()\n"
    for i in range(indices_len):
        code += f"    indices{i}_shape = indices[{i}].shape\n"
        code += f"    indices{i}_stride = indices[{i}].stride()\n"
    code += "    out_stride = out.stride()\n"
    code += "    M = indices[0].numel()\n"
    code += "    N = out.numel() // M\n"
    code += "\n"
    code += "    grid = lambda meta: (\n"
    code += "        triton.cdiv(M, meta['BLOCK_SIZE0']),\n"
    code += "        triton.cdiv(N, meta['BLOCK_SIZE1']),\n"
    code += "    )\n"
    code += "\n"
    code += f"    {kernel_name}[grid](\n"
    code += "        input,\n"
    for i in range(indices_len):
        code += f"        indices[{i}],\n"
    code += "        out,\n"
    for i in range(inp_rank):
        code += f"        input_shape[{i}],\n"
    for j in range(index_rank):
        code += f"        indices0_shape[{j}],\n"
    for i in range(inp_rank):
        code += f"        input_stride[{i}],\n"
    for i in range(indices_len):
        for j in range(index_rank):
            code += f"        indices{i}_stride[{j}],\n"
    for i in range(out_rank):
        code += f"        out_stride[{i}],\n"
    code += "        M,\n"
    code += "        N,\n"
    code += "        BLOCK_SIZE0=128,\n"
    code += "        BLOCK_SIZE1=512,\n"
    code += "    )\n"
    code += "    return out\n\n"
    return code


def _generate_code(
    inputs: tuple[Any],
    wrapper_name: str,
    kernel_name: str,
):
    inp, tensor_indices, idx_dims, bcast_pos = inputs
    inp_rank = inp.ndim
    indices_len = len(idx_dims)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = ""
    code = _generate_imports(code)
    code = _generate_gather_kernel(
        inp_rank, idx_dims, index_rank, bcast_pos, kernel_name, code
    )
    code = _generate_wrapper(
        inp_rank, idx_dims, index_rank, bcast_pos, wrapper_name, kernel_name, code
    )
    return code


class UnsafeIndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads = {}

    def __call__(self, inp, tensor_indices, out, idx_dims, bcast_pos):
        key = self.arg_key(inp, tensor_indices, idx_dims, bcast_pos)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = _generate_code(
                (inp, tensor_indices, idx_dims, bcast_pos),
                "_unsafe_index_wrapper",
                "_unsafe_index_jit_function",
            )
            file_name = f"unsafe_index_{key}.py"
            file_path = str(_code_cache_dir() / file_name)
            write_atomic(file_path, code)
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_unsafe_index_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_unsafe_index_wrapper")
            self.overloads[key] = overload

        return overload(inp, tensor_indices, out)

    def arg_key(self, inp, tensor_indices, idx_dims, bcast_pos):
        dims = "-".join(str(d) for d in idx_dims)
        index_rank = tensor_indices[0].ndim
        return f"r{inp.ndim}_d{dims}_ir{index_rank}_bp{bcast_pos}"


_unsafe_index_func = UnsafeIndexFunction()


# aten rejects a bad index dtype with
# ``RuntimeError: _unsafe_index found unexpected index type Bool`` where the
# name is the legacy TypeMeta name; mirror both the error type and the text.
_DTYPE_LEGACY_NAMES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
}


def _check_indices(inp, indices):
    """Validate index dtypes and move cross-device indices, in one pass.

    Mirrors ``aten._unsafe_index``: only int32/int64 index tensors are
    accepted (bool/int8/uint8/float/... are all rejected with a
    RuntimeError), and an index living on a different device than the input
    is silently moved onto it.
    """
    checked = []
    for index in indices:
        if index is None:
            checked.append(None)
            continue
        dt = index.dtype
        if dt not in (torch.int32, torch.int64):
            name = _DTYPE_LEGACY_NAMES.get(dt, str(dt))
            raise RuntimeError(f"_unsafe_index found unexpected index type {name}")
        if index.device != inp.device:
            index = index.to(inp.device)
        checked.append(index)
    return checked


def _broadcast_index_tensors(tensor_indices):
    """Broadcast index tensors, mirroring aten's IndexError on a mismatch."""
    try:
        return list(torch.broadcast_tensors(*tensor_indices))
    except RuntimeError:
        shapes = ", ".join(str(list(idx.shape)) for idx in tensor_indices)
        raise IndexError(
            "shape mismatch: indexing tensors could not be broadcast "
            f"together with shapes {shapes}"
        ) from None


def _eliminate_scalar_indices(inp, indices):
    """Resolve 0-d (scalar) tensor indices on the host.

    In aten a 0-d index tensor behaves like an integer index: the indexed
    dim is removed from the output.  Resolve them with ``select`` views so
    the generated kernels only ever see index tensors with rank >= 1 (a 0-d
    index would otherwise make the codegen emit an invalid scalar load).
    """
    if not any(idx is not None and idx.ndim == 0 for idx in indices):
        return inp, indices
    remaining = []
    removed = 0
    for i, idx in enumerate(indices):
        if idx is not None and idx.ndim == 0:
            # List position i aligns with input dim i; each selected-away
            # dim shifts the positions of the following dims down by one.
            pos = i - removed
            v = int(idx.item())
            if v < 0:
                v += inp.shape[pos]
            inp = inp.select(pos, v)
            removed += 1
        else:
            remaining.append(idx)
    return inp, remaining


def unsafe_index(inp, indices):
    """``unsafe_index`` matching ``aten._unsafe_index``.

    Host side is a straight line: validate, resolve scalar (0-d) indices,
    broadcast, compute the output layout per aten's subspace-placement rule,
    and launch.  The kernels take an explicit index->dim mapping plus full
    input/output strides, so no input permute/contiguous and no output
    post-permute is ever needed.
    """
    if not indices:
        raise ValueError("at least one index must be provided")
    indices = _check_indices(inp, list(indices))
    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    # Subspace placement is decided over the *original* advanced indices:
    # aten counts scalar (0-d) tensors as advanced indices, so a scalar
    # separated from the other index tensors by a None dim already forces
    # the broadcast subspace to the front of the output.
    advanced_dims = [i for i, idx in enumerate(indices) if idx is not None]
    subspace_split = bool(advanced_dims) and advanced_dims != list(
        range(advanced_dims[0], advanced_dims[0] + len(advanced_dims))
    )

    inp, indices = _eliminate_scalar_indices(inp, indices)
    if not indices:
        return inp  # every index was a scalar (0-d) tensor

    # Pad missing trailing dims with None (full slices).
    indices = indices + [None] * (inp.ndim - len(indices))

    kernel_indices = [idx for idx in indices if idx is not None]
    if not kernel_indices:
        # All-None indices: plain basic indexing, just a contiguous copy.
        # (aten's own kernel asserts on this case; a copy is harmless.)
        return inp.contiguous()
    if len(kernel_indices) > 1:
        kernel_indices = _broadcast_index_tensors(kernel_indices)

    idx_dims = tuple(i for i, idx in enumerate(indices) if idx is not None)
    slice_dims = [i for i, idx in enumerate(indices) if idx is None]

    # Output layout per aten's rule: when the advanced indices are separated
    # by None dims the broadcast subspace goes to the front; otherwise it
    # sits where the (contiguous) advanced block starts, counting surviving
    # dims.
    bcast_pos = 0 if subspace_split else idx_dims[0]
    index_rank = kernel_indices[0].ndim
    out_rank = index_rank + len(slice_dims)
    bcast_out = list(range(bcast_pos, bcast_pos + index_rank))
    rest = [p for p in range(out_rank) if p not in bcast_out]
    out_shape = [0] * out_rank
    for r, p in enumerate(bcast_out):
        out_shape[p] = kernel_indices[0].shape[r]
    for d, p in zip(slice_dims, rest):
        out_shape[p] = inp.shape[d]

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    # Skip the kernel launch for an empty output: an empty index would give
    # a grid of (0, ...) programs and fault the device; the empty output
    # already has the correct shape.
    if out.numel() != 0:
        _unsafe_index_func(inp, kernel_indices, out, idx_dims, bcast_pos)
    return out

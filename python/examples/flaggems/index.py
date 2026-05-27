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
        path.parent
        / f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
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


def _generate_index_kernel(
    inp_rank,
    indices_len,
    index_rank,
    kernel_name: str,
    code,
):
    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    input_ptr,\n"
    for i in range(indices_len):
        code += f"    indices{i}_ptr,\n"
    code += "    out_ptr,\n"
    for i in range(inp_rank):
        code += f"    input_shape{i},\n"
    for i in range(indices_len):
        for j in range(index_rank):
            code += f"    indices{i}_shape{j},\n"
    for i in range(inp_rank):
        code += f"    input_stride{i},\n"
    for i in range(indices_len):
        for j in range(index_rank):
            code += f"    indices{i}_stride{j},\n"
    for i in range(index_rank + inp_rank - indices_len):
        code += f"    out_stride{i},\n"
    code += "    M,\n"
    code += "    N,\n"
    code += "    BLOCK_SIZE0: tl.constexpr,\n"
    code += "    BLOCK_SIZE1: tl.constexpr,\n"
    code += "):\n"
    code += "    pid0 = tl.program_id(axis=0)\n"
    code += "    pid1 = tl.program_id(axis=1)\n"
    code += "    offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]\n"
    if inp_rank == indices_len:
        code += "    offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]\n"
    else:
        code += "    offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]\n"
    code += "\n"
    code += "    cur_idx = offset0\n"
    for i in range(index_rank - 1, -1, -1):
        code += f"    indices_idx{i} = cur_idx % indices0_shape{i}\n"
        code += f"    cur_idx = cur_idx // indices0_shape{i}\n"
    code += "\n"
    code += "    cur_idx = offset1\n"
    for i in range(inp_rank - 1, indices_len - 1, -1):
        code += f"    input_idx{i} = cur_idx % input_shape{i}\n"
        code += f"    cur_idx = cur_idx // input_shape{i}\n"
    code += "\n"
    code += "    mask0 = offset0 < M\n"
    for i in range(indices_len):
        comp = " + ".join(
            [
                f"indices_idx{j} * indices{i}_stride{j}"
                for j in range(index_rank)
            ]
        )
        code += f"    cur_index{i} = tl.load(indices{i}_ptr + {comp}, mask=mask0, other=0)\n"
    code += "\n"
    index_mask = " & ".join(
        [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
            for i in range(indices_len)
        ]
    )
    code += f"    index_mask = {index_mask}\n"
    code += "    mask1 = offset1 < N\n"
    code += "    mask = index_mask & mask0 & mask1\n"
    code += "\n"
    comp = " + ".join(
        [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
    )
    comp += "".join(
        [
            f" + input_idx{i} * input_stride{i}"
            for i in range(indices_len, inp_rank)
        ]
    )
    code += f"    input_offset = {comp}\n"
    comp = " + ".join(
        [f"indices_idx{i} * out_stride{i}" for i in range(index_rank)]
    )
    for i in range(inp_rank - indices_len):
        comp += f" + input_idx{indices_len + i} * out_stride{index_rank + i}"
    code += f"    out_offset = {comp}\n"
    code += "\n"
    code += "    cur_value = tl.load(input_ptr + input_offset, mask=mask)\n"
    code += "    tl.store(out_ptr + out_offset, cur_value, mask=mask)\n\n"
    return code


def _generate_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code,
):
    code += f"def {wrapper_name}(inp, indices, out):\n"
    code += "    input_shape = inp.shape\n"
    code += "    input_stride = inp.stride()\n"
    for i in range(indices_len):
        code += f"    indices{i}_shape = indices[{i}].shape\n"
        code += f"    indices{i}_stride = indices[{i}].stride()\n"
    code += "    out_shape = out.shape\n"
    code += "    out_stride = out.stride()\n"
    code += "    M = indices[0].numel()\n"
    code += "    N = 1\n"
    for i in range(indices_len, inp_rank):
        code += f"    N *= input_shape[{i}]\n"
    code += "\n"
    code += "    grid = lambda meta: (\n"
    code += "        triton.cdiv(M, meta['BLOCK_SIZE0']),\n"
    code += "        triton.cdiv(N, meta['BLOCK_SIZE1']),\n"
    code += "    )\n"
    code += "\n"
    code += f"    {kernel_name}[grid](\n"
    code += "        inp,\n"
    for i in range(indices_len):
        code += f"        indices[{i}],\n"
    code += "        out,\n"
    for i in range(inp_rank):
        code += f"        input_shape[{i}],\n"
    for i in range(indices_len):
        for j in range(index_rank):
            code += f"        indices{i}_shape[{j}],\n"
    for i in range(inp_rank):
        code += f"        input_stride[{i}],\n"
    for i in range(indices_len):
        for j in range(index_rank):
            code += f"        indices{i}_stride[{j}],\n"
    for i in range(index_rank + inp_rank - indices_len):
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
    inp, tensor_indices = inputs
    inp_rank = inp.ndim
    indices_len = len(tensor_indices)
    index_rank = tensor_indices[0].ndim if indices_len > 0 else 0
    code = ""
    code = _generate_imports(code)
    code = _generate_index_kernel(
        inp_rank, indices_len, index_rank, kernel_name, code
    )
    code = _generate_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, out = args
        full_args = (inp, tensor_indices)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = _generate_code(
                full_args,
                "_index_wrapper",
                "_index_jit_function",
            )
            file_name = f"index_{key}.py"
            file_path = str(_code_cache_dir() / file_name)
            write_atomic(file_path, code)
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices = args[0], args[1]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        index_rank = tensor_indices[0].ndim if indices_len > 0 else 0
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_func = IndexFunction()


def index(inp, indices):
    original_indices = list(indices)
    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    indices = [
        index.to(inp.device)
        if index is not None and index.device != inp.device
        else index
        for index in indices
    ]

    # Step 1: Process indices (convert bool/int8 to long, handle None)
    processed_indices = []
    for i, index in enumerate(indices):
        if index is not None:
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(processed_indices)
                if k + index.ndim > inp.ndim:
                    raise IndexError(
                        f"too many indices for tensor of dimension {inp.ndim}"
                    )
                for j in range(index.ndim):
                    if index.shape[j] != inp.shape[k + j]:
                        raise IndexError(
                            f"The shape of the mask {index.shape} at index {i} "
                            f"does not match the shape of the indexed tensor {inp.shape} at index {k + j}"
                        )
                for j in range(index.ndim):
                    processed_indices.append(nonzero.select(1, j))
            elif index.dtype in [
                torch.long,
                torch.int,
                torch.int32,
                torch.int64,
            ]:
                processed_indices.append(index)
            else:
                raise TypeError(
                    "tensors used as indices must be long, int, byte or bool tensors"
                )
        else:
            processed_indices.append(None)

    indices = processed_indices

    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    has_any_tensor = any(idx is not None for idx in indices)
    starts_with_none = indices[0] is None if indices else False

    # Step 2: Broadcast indices
    tensor_indices = [idx for idx in indices if idx is not None]
    if tensor_indices:
        if len(tensor_indices) > 1:
            tensor_indices = list(torch.broadcast_tensors(*tensor_indices))
        tensor_idx = 0
        for i in range(len(indices)):
            if indices[i] is not None:
                indices[i] = tensor_indices[tensor_idx]
                tensor_idx += 1

    # Step 3: Pad with None to input.ndim
    while len(indices) < inp.ndim:
        indices.append(None)

    # Step 4: Check contiguous subspace
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # Transpose if not contiguous
    need_post_process = False
    first_tensor_dim = None
    if not has_contiguous_subspace or (starts_with_none and has_any_tensor):
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        inp = inp.permute(dims)
        indices = transposed_indices

        if starts_with_none and has_any_tensor and has_contiguous_subspace:
            need_post_process = True
            for i, idx in enumerate(original_indices):
                if idx is not None:
                    first_tensor_dim = i
                    break

    # Step 5: Calculate output shape
    before_shape = []
    after_shape = []
    replacement_shape = []

    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(inp.shape[dim])
            else:
                before_shape.append(inp.shape[dim])
        else:
            if not replacement_shape:
                replacement_shape = list(index.shape)

    out_shape = before_shape + replacement_shape + after_shape
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if inp.numel() == 0:
        return out.contiguous()

    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        return inp.view(*out_shape).contiguous()

    _index_func(inp, tensor_indices, out)

    if need_post_process:
        index_rank = tensor_indices[0].ndim
        pre_dims = list(range(index_rank, index_rank + first_tensor_dim))
        broadcast_dims = list(range(index_rank))
        post_dims = list(range(index_rank + first_tensor_dim, out.ndim))
        new_order = pre_dims + broadcast_dims + post_dims
        out = out.permute(new_order)

    return out.contiguous()

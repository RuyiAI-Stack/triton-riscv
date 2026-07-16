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


def get_max_rank_shape(indices: list[torch.Tensor]) -> list[int]:
    tensor_indices = [idx for idx in indices if idx is not None]
    if len(tensor_indices) == 0:
        return []
    max_rank = max([len(index.shape) for index in tensor_indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in tensor_indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if index is not None and tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def _code_cache_dir():
    d = Path.home() / ".flaggems" / "code_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _generate_imports(code):
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def _generate_index_put_kernel(
    inp_rank,
    indices_len,
    index_rank,
    kernel_name,
    code,
):
    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    input_ptr,\n"
    for i in range(indices_len):
        code += f"    indices{i}_ptr,\n"
    code += "    values_ptr,\n"
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
        code += f"    values_stride{i},\n"
    code += "    M,\n"
    code += "    N,\n"
    code += "    IS_ACCUMULATE: tl.constexpr,\n"
    code += "    BLOCK_SIZE0: tl.constexpr = 2,\n"
    code += "    BLOCK_SIZE1: tl.constexpr = 2048,\n"
    code += "):\n"
    code += "    pid0 = tl.program_id(axis=0)\n"
    code += "    pid1 = tl.program_id(axis=1)\n"
    code += "    offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]\n"
    if inp_rank == indices_len:
        code += "    offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]\n"
    else:
        code += (
            "    offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]\n"
        )
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
            [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
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
    comp = " + ".join([f"cur_index{i} * input_stride{i}" for i in range(indices_len)])
    comp += "".join(
        [f" + input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)]
    )
    code += f"    input_offset = {comp}\n"
    comp = " + ".join([f"indices_idx{i} * values_stride{i}" for i in range(index_rank)])
    for i in range(inp_rank - indices_len):
        comp += f" + input_idx{indices_len + i} * values_stride{index_rank + i}"
    code += f"    values_offset = {comp}\n"
    code += "\n"
    code += "    cur_value = tl.load(values_ptr + values_offset, mask=mask)\n"
    code += "    if IS_ACCUMULATE:\n"
    code += "        tl.atomic_add(input_ptr + input_offset, cur_value, mask=mask)\n"
    code += "    else:\n"
    code += "        tl.store(input_ptr + input_offset, cur_value, mask=mask)\n\n"
    return code


def _generate_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code,
):
    code += f"def {wrapper_name}(inp, indices, values, accumulate):\n"
    code += "    input_shape = inp.shape\n"
    code += "    input_stride = inp.stride()\n"
    for i in range(indices_len):
        code += f"    indices{i}_shape = indices[{i}].shape\n"
        code += f"    indices{i}_stride = indices[{i}].stride()\n"
    code += "    values_stride = values.stride()\n"
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
    code += "        values,\n"
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
        code += f"        values_stride[{i}],\n"
    code += "        M,\n"
    code += "        N,\n"
    code += "        accumulate==True,\n"
    code += "    )\n"
    code += "    return inp\n\n"
    return code


def _generate_code(
    inputs: tuple[Any],
    wrapper_name: str,
    kernel_name: str,
):
    inp, tensor_indices, _ = inputs
    inp_rank = inp.ndim
    indices_len = len(tensor_indices)
    index_rank = tensor_indices[0].ndim if indices_len > 0 else 0
    code = ""
    code = _generate_imports(code)
    code = _generate_index_put_kernel(
        inp_rank, indices_len, index_rank, kernel_name, code
    )
    code = _generate_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexPutFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, values, accumulate = args
        full_args = (inp, tensor_indices, values)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = _generate_code(
                full_args,
                "_index_put_wrapper",
                "_index_put_jit_function",
            )
            file_name = f"index_put_{key}.py"
            file_path = str(_code_cache_dir() / file_name)
            write_atomic(file_path, code)
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_put_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices, _ = args[0], args[1], args[2]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        index_rank = tensor_indices[0].ndim if indices_len > 0 else 0
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_put_func = IndexPutFunction()


def index_put(inp, indices, values, accumulate=False):
    out = inp.clone()
    return index_put_(out, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")

    indices = [
        index.to(inp.device)
        if index is not None and index.device != inp.device
        else index
        for index in indices
    ]
    # Step 1: Index preprocessing
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices
    if len(indices) < inp.ndim:
        indices.extend([None] * (inp.ndim - len(indices)))

    if len(indices) > inp.ndim:
        raise IndexError(f"too many indices for tensor of dimension {inp.ndim}")

    # Step 2: Broadcast tensor indices
    tensor_pos = [i for i, x in enumerate(indices) if x is not None]
    if not tensor_pos:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [indices[i] for i in tensor_pos]
    if len(tensor_indices) > 1:
        broadcasted = torch.broadcast_tensors(*tensor_indices)
        for i, pos in enumerate(tensor_pos):
            indices[pos] = broadcasted[i]

    # Step 3: Transpose
    is_contiguous = (tensor_pos[-1] - tensor_pos[0] + 1) == len(tensor_pos)
    starts_with_none = indices[0] is None
    need_transpose = not is_contiguous or starts_with_none

    if need_transpose:
        perm_order = tensor_pos + [i for i, x in enumerate(indices) if x is None]
        inp_view = inp.permute(perm_order)
        final_indices = [indices[i] for i in tensor_pos] + [None] * (
            len(indices) - len(tensor_pos)
        )
    else:
        inp_view = inp
        final_indices = indices

    # Step 4: Handle Values shape and broadcasting
    tensors = [x for x in final_indices if x is not None]
    broadcast_shape = list(tensors[0].shape)
    slice_shape = [inp_view.shape[i] for i, x in enumerate(final_indices) if x is None]

    target_shape = broadcast_shape + slice_shape
    values = values.to(inp.device)
    if need_transpose and is_contiguous:
        num_before = tensor_pos[0]

        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        natural_shape = before_dims + broadcast_shape + after_dims
        values = values.broadcast_to(natural_shape)

        B, T = len(before_dims), len(broadcast_shape)
        val_perm = (
            list(range(B, B + T)) + list(range(0, B)) + list(range(B + T, values.ndim))
        )
        values = values.permute(val_perm)
    else:
        values = values.broadcast_to(target_shape)

    _index_put_func(inp_view, tensors, values, accumulate)

    return inp


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    indices = list(indices)
    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()
        target_shape = (K, *inp.shape[len(indices) :])

        if values.numel() == 1:
            values = torch.full(
                target_shape, values.item(), dtype=inp.dtype, device=inp.device
            )
        elif values.numel() == K:
            values = values.reshape((K,)).expand(target_shape)

    indices = [
        index.to(inp.device)
        if index is not None and index.device != inp.device
        else index
        for index in indices
    ]

    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)
    target_shape += inp.shape[len(indices) :]
    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        raise ValueError("At least one non-None index tensor is required")

    if values.device != inp.device:
        values = values.to(inp.device)
    values = torch.broadcast_to(values, target_shape)

    _index_put_func(inp, tensor_indices, values, accumulate)
    return inp

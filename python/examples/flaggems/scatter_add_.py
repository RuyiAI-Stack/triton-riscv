import importlib
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def restride_dim(src, dim, shape, step=0, storage_offset=None):
    strides = list(src.stride())
    strides[dim] *= step
    return src.as_strided(shape, strides, storage_offset)


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


@triton.jit
def scatter_add_kernel_1(
    index_dim_n,
    inp_dim_n,
    out_ptr,
    index_ptr,
    src_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * LOOP
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements
    for loop_iter in tl.static_range(LOOP):
        src_index_offsets = block_start + arange
        src_tensor = tl.load(src_ptr + src_index_offsets, mask=mask, other=0)
        index_tensor = tl.load(index_ptr + src_index_offsets, mask=mask, other=0)
        out_offsets = src_index_offsets // index_dim_n * inp_dim_n + index_tensor
        tl.atomic_add(out_ptr + out_offsets, src_tensor, mask=mask, sem="relaxed")
        block_start += BLOCK_SIZE


def generate_imports(code: str) -> str:
    code += "import torch\n"
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def generate_scatter_kernel(
    rank: int,
    kernel_name: str,
    code: str,
) -> str:
    code += "def heur_block(args):\n"
    code += "    return 128\n\n"
    code += "def loop_count(args):\n"
    code += "    return 1\n\n"

    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    src_strided,\n"
    code += "    index,\n"
    code += "    inp,\n"
    code += "    out,\n"
    for i in range(rank):
        code += f"    inp_stride_{i}: int,\n"
    for i in range(rank):
        code += f"    index_stride_{i}: int,\n"
    for i in range(rank):
        code += f"    src_stride_{i}: int,\n"
    for i in range(rank):
        code += f"    shape_{i}: int,\n"
    code += "    inp_size_dim,\n"
    code += "    stride_dim,\n"
    code += "    N,\n"
    code += "    BLOCK: tl.constexpr,\n"
    code += "    LOOP: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)\n"
    code += "    for loop_iter in tl.static_range(LOOP):\n"
    code += "        mask = offsets < N\n"
    code += "        cur_idx = offsets\n"
    code += "        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    for i in range(rank)[::-1]:
        code += f"        mod = cur_idx % shape_{i}\n"
        code += f"        inp_offsets += mod * inp_stride_{i}\n"
        code += f"        idx_offsets += mod * index_stride_{i}\n"
        code += f"        src_offsets += mod * src_stride_{i}\n"
        if i != 0:
            code += f"        cur_idx = cur_idx // shape_{i}\n"
    code += "        cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)\n"
    code += "        cur_index = tl.load(index + idx_offsets, mask=mask, other=0)\n"
    code += "        dim_offsets = cur_index * stride_dim\n"
    code += "        inp_offsets += dim_offsets\n"
    code += (
        "        tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')\n"
    )
    code += "        offsets += BLOCK\n\n"
    return code


def parameter_for_wrapper() -> str:
    parameters: list[str] = []
    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: str,
) -> str:
    parameters = parameter_for_wrapper()
    code += f"def {wrapper_name}({parameters}):\n"
    code += "    inp_strides = list(inp.stride())\n"
    code += "    index_strides = index.stride()\n"
    code += "    src_strides = src_strided.stride()\n"
    code += "    index_shapes = list(index.shape)\n"
    code += "    inp_size_dim = dim_size\n"
    code += "    stride_dim = dim_stride\n"
    code += "    def grid(meta):\n"
    code += "        return (triton.cdiv(N, meta['BLOCK'] * meta['LOOP']),)\n"
    code += f"    {kernel_name}[grid](\n"
    code += "        src_strided, index, inp, out,\n"
    for i in range(rank):
        code += f"        inp_strides[{i}],\n"
    for i in range(rank):
        code += f"        index_strides[{i}],\n"
    for i in range(rank):
        code += f"        src_strides[{i}],\n"
    for i in range(rank):
        code += f"        index_shapes[{i}],\n"
    code += "        inp_size_dim,\n"
    code += "        stride_dim,\n"
    code += "        N,\n"
    code += "    )\n"
    code += "    return out\n\n"
    return code


def generate_code(
    inputs: tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: str,
) -> str:
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = generate_code(
                args, "_scatter_add_wrapper", "_scatter_add_jit_function", ""
            )

            file_name = f"scatter_add_rank_{key}_pid_{self.pid}.py"
            cache_dir = Path.home() / ".flaggems" / "code_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = cache_dir / file_name
            write_atomic(str(file_path), code)

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                file_path,
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_add_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


def clip_tensor_to_shape(b, a):
    target_shape = a.shape
    slices = [
        slice(0, min(b.shape[i], target_shape[i])) for i in range(len(target_shape))
    ]
    clipped_b = b[tuple(slices)]
    return clipped_b


def scatter_add_0(inp, dim, index, src):
    dtype_convert = False
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        out = inp.to(torch.float32)
        dtype_convert = True
    else:
        out = inp

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
    )
    if dtype_convert:
        return inp.copy_(out.to(src.dtype))
    return out


def scatter_add_1(x, dim, index, src):
    index_dim_n = index.size(dim)
    inp_dim_n = x.size(dim)
    origin = x
    if dim != x.ndim - 1:
        x = dim_compress(x, dim)
    if dim != x.ndim - 1:
        src = dim_compress(src, dim)
    if dim != x.ndim - 1:
        index = dim_compress(index, dim)

    all_elem = max(x.numel(), index.numel())

    def grid(meta):
        return (triton.cdiv(all_elem, meta["BLOCK_SIZE"] * meta["LOOP"]),)

    dtype_convert = False
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        dtype_convert = True
        x = x.to(torch.float32)

    scatter_add_kernel_1[grid](
        index_dim_n, inp_dim_n, x, index, src, all_elem, BLOCK_SIZE=256, LOOP=1
    )

    if dim != x.ndim - 1:
        order = [i for i in range(x.ndim - 1)]
        order.insert(dim, x.ndim - 1)
        result = x.to(src.dtype) if dtype_convert else x
        return origin.copy_(result.permute(order))
    if dtype_convert:
        return origin.copy_(x.to(src.dtype))
    return x


def scatter_add_(x, dim, index, src):
    assert x.dim() == index.dim() and x.dim() == src.dim(), "Invalid dim"
    dim = dim % x.ndim
    assert dim >= 0 and dim < x.dim(), "Invalid dim"
    assert index.size(dim) <= src.size(dim), "Invalid src"
    equal_count = 0
    for d in range(x.dim()):
        if d != dim:
            assert index.size(d) <= x.size(d), "Invalid x"
            if index.size(d) == x.size(d):
                equal_count += 1
        else:
            if index.size(dim) >= x.size(dim):
                equal_count += 1

    if equal_count == x.dim() and index.shape == src.shape and dim == x.ndim - 1:
        return scatter_add_1(x, dim, index, src)
    if (index.shape == src.shape and index.shape == x.shape and dim != x.ndim - 1) or (
        x.shape[0] == 4096 and x.numel() >= 9437184 and dim != x.ndim - 1
    ):
        if index.shape != src.shape:
            src = clip_tensor_to_shape(src, index)
        return scatter_add_1(x, dim, index, src)
    else:
        return scatter_add_0(x, dim, index, src)

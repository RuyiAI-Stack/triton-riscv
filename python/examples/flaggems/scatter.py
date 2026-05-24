import enum
import importlib
import os
import threading
import uuid
from collections.abc import Callable, Mapping
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


class MemOverlap(enum.Enum):
    No = 0
    Yes = 1
    TooHard = 2


def has_internal_overlapping(x: torch.Tensor):
    if x.is_contiguous():
        return MemOverlap.No
    if torch.ops.aten.is_non_overlapping_and_dense(x):
        return MemOverlap.No
    for size, stride in zip(x.size(), x.stride()):
        if size > 1 and stride == 0:
            return MemOverlap.Yes
    return MemOverlap.TooHard


def restride_dim(src, dim, shape, step=0, storage_offset=None):
    strides = list(src.stride())
    strides[dim] *= step
    return src.as_strided(shape, strides, storage_offset)


def parameter_for_wrapper() -> str:
    parameters: list[str] = []
    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    parameters.append("reduce: tl.constexpr=None")
    parameters.append("int32_offset: tl.constexpr=None")
    return ", ".join(parameters)


def generate_imports(code: str) -> str:
    code += "import torch\n"
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def generate_scatter_kernel(rank: int, kernel_name: str, code: str) -> str:
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code += f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])\n"
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
    code += "    IS_ADD: tl.constexpr,\n"
    code += "    IS_MUL: tl.constexpr,\n"
    code += "    BLOCK: tl.constexpr,\n"
    code += "    LOOP: tl.constexpr,\n"
    code += "    INT32_OFFSET: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    if not INT32_OFFSET:\n"
    code += "        pid = pid.to(tl.int64)\n"
    code += "    offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)\n"
    code += "    for loop_iter in tl.static_range(LOOP):\n"
    code += "        mask = offsets < N\n"
    code += "        cur_idx = offsets\n"
    code += "        if INT32_OFFSET:\n"
    code += "            inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "            idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "            src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "        else:\n"
    code += "            inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    code += "            idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    code += "            src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    for i in range(rank)[::-1]:
        code += "        if INT32_OFFSET:\n"
        code += f"            shape_{i} = shape_{i}.to(tl.int32)\n"
        code += f"            inp_stride_{i} = inp_stride_{i}.to(tl.int32)\n"
        code += (
            f"            index_stride_{i} = index_stride_{i}.to(tl.int32)\n"
        )
        code += f"            src_stride_{i} = src_stride_{i}.to(tl.int32)\n"
        code += f"        mod = cur_idx % shape_{i}\n"
        code += f"        inp_offsets += mod * inp_stride_{i}\n"
        code += f"        idx_offsets += mod * index_stride_{i}\n"
        code += f"        src_offsets += mod * src_stride_{i}\n"
        if i != 0:
            code += f"        cur_idx = cur_idx // shape_{i}\n"
    code += "        cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)\n"
    code += "        cur_index = tl.load(index + idx_offsets, mask=mask, other=0)\n"
    code += "        if INT32_OFFSET:\n"
    code += "            cur_index = cur_index.to(tl.int32)\n"
    code += "            stride_dim = stride_dim.to(tl.int32)\n"
    code += "        dim_offsets = cur_index * stride_dim\n"
    code += "        inp_offsets += dim_offsets\n"
    code += "        if IS_ADD:\n"
    code += "            tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')\n"
    code += "        elif IS_MUL:\n"
    code += "            stop = tl.where(mask, 0, 1).to(tl.int1)\n"
    code += "            block_stop = False\n"
    code += "            while not block_stop:\n"
    code += "                cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)\n"
    code += (
        "                res = tl.where(stop, cur_inp, cur_inp * cur_src)\n"
    )
    code += "                cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res, sem='relaxed')\n"
    code += "                stop |= cur_inp == cas_res\n"
    code += "                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK\n"
    code += "        else:\n"
    code += "            tl.store(out + inp_offsets, cur_src, mask=mask)\n"
    code += "        offsets += BLOCK\n\n"
    return code


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
    code += "    IS_ADD = reduce == 'add'\n"
    code += "    IS_MUL = reduce == 'multiply'\n"
    code += "    int32_offset = int32_offset or True\n"
    code += "    BLOCK = 128\n"
    code += "    LOOP = 4\n"
    code += "    def grid(meta):\n"
    code += "        return (triton.cdiv(N, BLOCK * LOOP),)\n"
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
    code += "        IS_ADD,\n"
    code += "        IS_MUL,\n"
    code += "        BLOCK=BLOCK,\n"
    code += "        LOOP=LOOP,\n"
    code += "        INT32_OFFSET=int32_offset,\n"
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
    code = generate_destination_passing_wrapper(
        rank, wrapper_name, kernel_name, code
    )
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
                args, "_scatter_wrapper", "_scatter_jit_function", ""
            )

            file_name = f"scatter_rank_{key}.py"
            cache_dir = Path.home() / ".flaggems" / "code_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = cache_dir / file_name
            write_atomic(str(file_path), code)

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


def scatter(inp, dim, index, src, reduce=None):
    out = inp.clone()

    if reduce is not None:
        assert inp.dtype not in (torch.bfloat16,), (
            "Unsupported operation: reduce scatter bfloat tensors."
        )

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    def int32_size_dim(x):
        return x.stride(dim) * x.size(dim) < 2**32

    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce=reduce,
        int32_offset=use_int32_offset,
    )

    return out


def scatter_(inp, dim, index, src, reduce=None):
    out = inp

    if reduce is not None:
        assert inp.dtype not in (torch.bfloat16,), (
            "Unsupported operation: reduce scatter bfloat tensors."
        )

    assert has_internal_overlapping(out) != MemOverlap.Yes, (
        "Unsupported operation: trying to inplace write to an internally overlapping tensor."
    )

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    def int32_size_dim(x):
        return x.stride(dim) * x.size(dim) < 2**32

    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce=reduce,
        int32_offset=use_int32_offset,
    )

    return inp

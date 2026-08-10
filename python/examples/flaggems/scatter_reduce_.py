import importlib.util
import os
import threading
import uuid
from pathlib import Path

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
    code += "import torch\n"
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def _generate_scatter_reduce_kernel(rank, kernel_name, code):
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code += (
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])\n"
    )

    code += f"def {kernel_name}(\n"
    code += "    src_strided,\n"
    code += "    index,\n"
    code += "    inp,\n"
    code += "    out,\n"
    for i in range(rank):
        code += f"    inp_stride_{i},\n"
    for i in range(rank):
        code += f"    index_stride_{i},\n"
    for i in range(rank):
        code += f"    src_stride_{i},\n"
    for i in range(rank):
        code += f"    shape_{i},\n"
    code += "    inp_size_dim,\n"
    code += "    stride_dim,\n"
    code += "    N,\n"
    code += "    IS_SUM: tl.constexpr,\n"
    code += "    IS_PROD: tl.constexpr,\n"
    code += "    IS_AMAX: tl.constexpr,\n"
    code += "    IS_AMIN: tl.constexpr,\n"
    code += "    IS_MEAN: tl.constexpr,\n"
    code += "    DIM: tl.constexpr,\n"
    code += "    BLOCK: tl.constexpr,\n"
    code += "    LOOP: tl.constexpr,\n"
    code += "    INT32_OFFSET: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    if not INT32_OFFSET:\n"
    code += "        pid = pid.to(tl.int64)\n"
    code += "    offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)\n\n"
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
        code += f"            index_stride_{i} = index_stride_{i}.to(tl.int32)\n"
        code += f"            src_stride_{i} = src_stride_{i}.to(tl.int32)\n"
        code += f"        mod = cur_idx % shape_{i}\n"
        code += f"        if DIM != {i}:\n"
        code += f"            inp_offsets += mod * inp_stride_{i}\n"
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
    code += "        inp_offsets += dim_offsets\n\n"
    code += "        if IS_SUM or IS_MEAN:\n"
    code += "            tl.atomic_add(out + inp_offsets, cur_src, mask=mask)\n"
    code += "        elif IS_AMAX:\n"
    code += "            tl.atomic_max(out + inp_offsets, cur_src, mask=mask)\n"
    code += "        elif IS_AMIN:\n"
    code += "            tl.atomic_min(out + inp_offsets, cur_src, mask=mask)\n"
    code += "        offsets += BLOCK\n\n"
    return code


def _generate_scatter_prod_kernel(rank, kernel_name, code):
    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    src_strided,\n"
    code += "    index,\n"
    code += "    out,\n"
    for i in range(rank):
        code += f"    inp_stride_{i},\n"
    for i in range(rank):
        code += f"    index_stride_{i},\n"
    for i in range(rank):
        code += f"    src_stride_{i},\n"
    for i in range(rank):
        code += f"    index_shape_{i},\n"
    code += "    src_size_dim,\n"
    code += "    INCLUDE_SELF: tl.constexpr,\n"
    code += "    DIM: tl.constexpr,\n"
    code += "):\n"
    code += "    cur_idx = tl.program_id(0).to(tl.int64)\n"
    code += "    out_base = 0\n"
    code += "    index_base = 0\n"
    code += "    src_base = 0\n"
    code += "    inp_dim_stride = 0\n"
    code += "    index_dim_stride = 0\n"
    code += "    src_dim_stride = 0\n"
    for i in range(rank)[::-1]:
        code += f"    if DIM == {i}:\n"
        code += f"        inp_dim_stride = inp_stride_{i}\n"
        code += f"        index_dim_stride = index_stride_{i}\n"
        code += f"        src_dim_stride = src_stride_{i}\n"
        code += "    else:\n"
        code += f"        coord_{i} = cur_idx % index_shape_{i}\n"
        code += f"        out_base += coord_{i} * inp_stride_{i}\n"
        code += f"        index_base += coord_{i} * index_stride_{i}\n"
        code += f"        src_base += coord_{i} * src_stride_{i}\n"
        code += f"        cur_idx = cur_idx // index_shape_{i}\n"
    code += "    if not INCLUDE_SELF:\n"
    code += "        for reduce_idx in range(0, src_size_dim):\n"
    code += "            current_index = tl.load(index + index_base + reduce_idx * index_dim_stride)\n"
    code += "            out_offset = out_base + current_index * inp_dim_stride\n"
    code += "            tl.store(out + out_offset, 1)\n"
    code += "    for reduce_idx in range(0, src_size_dim):\n"
    code += "        current_index = tl.load(index + index_base + reduce_idx * index_dim_stride)\n"
    code += "        current_src = tl.load(src_strided + src_base + reduce_idx * src_dim_stride)\n"
    code += "        out_offset = out_base + current_index * inp_dim_stride\n"
    code += "        current_out = tl.load(out + out_offset)\n"
    code += "        tl.store(out + out_offset, current_out * current_src)\n\n"
    return code


def _generate_count_kernel(rank, kernel_name, code):
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))

    code += (
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{shape_vars}])\n"
    )
    code += f"def {kernel_name}(\n"
    code += "    index,\n"
    code += "    count,\n"
    for i in range(rank):
        code += f"    inp_stride_{i},\n"
    for i in range(rank):
        code += f"    index_stride_{i},\n"
    for i in range(rank):
        code += f"    shape_{i},\n"
    code += "    inp_size_dim,\n"
    code += "    stride_dim,\n"
    code += "    N,\n"
    code += "    DIM: tl.constexpr,\n"
    code += "    BLOCK: tl.constexpr,\n"
    code += "    LOOP: tl.constexpr,\n"
    code += "    INT32_OFFSET: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    if not INT32_OFFSET:\n"
    code += "        pid = pid.to(tl.int64)\n"
    code += "    offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)\n\n"
    code += "    for loop_iter in tl.static_range(LOOP):\n"
    code += "        mask = offsets < N\n"
    code += "        cur_idx = offsets\n"
    code += "        if INT32_OFFSET:\n"
    code += "            inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "            idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "        else:\n"
    code += "            inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    code += "            idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    for i in range(rank)[::-1]:
        code += "        if INT32_OFFSET:\n"
        code += f"            shape_{i} = shape_{i}.to(tl.int32)\n"
        code += f"            inp_stride_{i} = inp_stride_{i}.to(tl.int32)\n"
        code += f"            index_stride_{i} = index_stride_{i}.to(tl.int32)\n"
        code += f"        mod = cur_idx % shape_{i}\n"
        code += f"        if DIM != {i}:\n"
        code += f"            inp_offsets += mod * inp_stride_{i}\n"
        code += f"        idx_offsets += mod * index_stride_{i}\n"
        if i != 0:
            code += f"        cur_idx = cur_idx // shape_{i}\n"
    code += "        cur_index = tl.load(index + idx_offsets, mask=mask, other=0)\n"
    code += "        if INT32_OFFSET:\n"
    code += "            cur_index = cur_index.to(tl.int32)\n"
    code += "            stride_dim = stride_dim.to(tl.int32)\n"
    code += "        dim_offsets = cur_index * stride_dim\n"
    code += "        inp_offsets += dim_offsets\n\n"
    code += "        one = tl.full((BLOCK,), 1, dtype=tl.int32)\n"
    code += "        tl.atomic_add(count + inp_offsets, one, mask=mask)\n"
    code += "        offsets += BLOCK\n\n"
    return code


def _generate_init_kernel(rank, kernel_name, code):
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))

    code += (
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{shape_vars}])\n"
    )
    code += f"def {kernel_name}(\n"
    code += "    index,\n"
    code += "    out,\n"
    for i in range(rank):
        code += f"    inp_stride_{i},\n"
    for i in range(rank):
        code += f"    index_stride_{i},\n"
    for i in range(rank):
        code += f"    shape_{i},\n"
    code += "    inp_size_dim,\n"
    code += "    stride_dim,\n"
    code += "    N,\n"
    code += "    INIT_VALUE: tl.constexpr,\n"
    code += "    DIM: tl.constexpr,\n"
    code += "    BLOCK: tl.constexpr,\n"
    code += "    LOOP: tl.constexpr,\n"
    code += "    INT32_OFFSET: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    if not INT32_OFFSET:\n"
    code += "        pid = pid.to(tl.int64)\n"
    code += "    offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)\n\n"
    code += "    for loop_iter in tl.static_range(LOOP):\n"
    code += "        mask = offsets < N\n"
    code += "        cur_idx = offsets\n"
    code += "        if INT32_OFFSET:\n"
    code += "            inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "            idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)\n"
    code += "        else:\n"
    code += "            inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    code += "            idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)\n"
    for i in range(rank)[::-1]:
        code += "        if INT32_OFFSET:\n"
        code += f"            shape_{i} = shape_{i}.to(tl.int32)\n"
        code += f"            inp_stride_{i} = inp_stride_{i}.to(tl.int32)\n"
        code += f"            index_stride_{i} = index_stride_{i}.to(tl.int32)\n"
        code += f"        mod = cur_idx % shape_{i}\n"
        code += f"        if DIM != {i}:\n"
        code += f"            inp_offsets += mod * inp_stride_{i}\n"
        code += f"        idx_offsets += mod * index_stride_{i}\n"
        if i != 0:
            code += f"        cur_idx = cur_idx // shape_{i}\n"
    code += "        cur_index = tl.load(index + idx_offsets, mask=mask, other=0)\n"
    code += "        if INT32_OFFSET:\n"
    code += "            cur_index = cur_index.to(tl.int32)\n"
    code += "            stride_dim = stride_dim.to(tl.int32)\n"
    code += "        dim_offsets = cur_index * stride_dim\n"
    code += "        inp_offsets += dim_offsets\n\n"
    code += "        tl.store(out + inp_offsets, INIT_VALUE, mask=mask)\n"
    code += "        offsets += BLOCK\n\n"
    return code


def _generate_wrapper(
    rank,
    wrapper_name,
    kernel_name,
    prod_kernel_name,
    count_kernel_name,
    init_kernel_name,
    code,
):
    code += f"def {wrapper_name}(src_strided, index, inp, out, dim, dim_size, dim_stride, N, reduce=None, include_self=True, init_value=None, int32_offset=None):\n"
    code += "    inp_strides = list(inp.stride())\n"
    code += "    index_strides = list(index.stride())\n"
    code += "    src_strides = list(src_strided.stride())\n"
    code += "    index_shapes = list(index.shape)\n"
    code += "    inp_size_dim = dim_size\n"
    code += "    stride_dim = dim_stride\n\n"
    code += '    IS_SUM = reduce == "sum"\n'
    code += '    IS_PROD = reduce == "prod"\n'
    code += '    IS_AMAX = reduce == "amax"\n'
    code += '    IS_AMIN = reduce == "amin"\n'
    code += '    IS_MEAN = reduce == "mean"\n'
    code += "    int32_offset = int32_offset or True\n\n"
    code += "    BLOCK = 128\n"
    code += "    LOOP = 4\n"
    code += "    grid = lambda meta: (\n"
    code += "        triton.cdiv(N, BLOCK * LOOP),\n"
    code += "    )\n\n"
    code += "    if init_value is not None and not IS_PROD:\n"
    code += f"        {init_kernel_name}[grid](\n"
    code += "            index, out,\n"
    for i in range(rank):
        code += f"            inp_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_shapes[{i}],\n"
    code += "            inp_size_dim,\n"
    code += "            stride_dim,\n"
    code += "            N,\n"
    code += "            init_value,\n"
    code += "            dim,\n"
    code += "            BLOCK=BLOCK,\n"
    code += "            LOOP=LOOP,\n"
    code += "            INT32_OFFSET=int32_offset,\n"
    code += "        )\n\n"
    code += "    if IS_PROD:\n"
    code += "        if index_shapes[dim] == 0:\n"
    code += "            return out\n"
    code += "        prod_grid = (index.numel() // index_shapes[dim],)\n"
    code += f"        {prod_kernel_name}[prod_grid](\n"
    code += "            src_strided, index, out,\n"
    for i in range(rank):
        code += f"            inp_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_strides[{i}],\n"
    for i in range(rank):
        code += f"            src_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_shapes[{i}],\n"
    code += "            index_shapes[dim],\n"
    code += "            include_self,\n"
    code += "            dim,\n"
    code += "        )\n\n"
    code += "    else:\n"
    code += f"        {kernel_name}[grid](\n"
    code += "            src_strided, index, inp, out,\n"
    for i in range(rank):
        code += f"            inp_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_strides[{i}],\n"
    for i in range(rank):
        code += f"            src_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_shapes[{i}],\n"
    code += "            inp_size_dim,\n"
    code += "            stride_dim,\n"
    code += "            N,\n"
    code += "            IS_SUM,\n"
    code += "            IS_PROD,\n"
    code += "            IS_AMAX,\n"
    code += "            IS_AMIN,\n"
    code += "            IS_MEAN,\n"
    code += "            dim,\n"
    code += "            BLOCK=BLOCK,\n"
    code += "            LOOP=LOOP,\n"
    code += "            INT32_OFFSET=int32_offset,\n"
    code += "        )\n\n"
    code += "    if IS_MEAN:\n"
    code += "        count = torch.zeros_like(out, dtype=torch.int32)\n"
    code += "        if include_self:\n"
    code += "            count.fill_(1)\n"
    code += f"        {count_kernel_name}[grid](\n"
    code += "            index, count,\n"
    for i in range(rank):
        code += f"            inp_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_strides[{i}],\n"
    for i in range(rank):
        code += f"            index_shapes[{i}],\n"
    code += "            inp_size_dim,\n"
    code += "            stride_dim,\n"
    code += "            N,\n"
    code += "            dim,\n"
    code += "            BLOCK=BLOCK,\n"
    code += "            LOOP=LOOP,\n"
    code += "            INT32_OFFSET=int32_offset,\n"
    code += "        )\n"
    code += "        count = count.clamp(min=1)\n"
    code += "        out.div_(count)\n\n"
    code += "    return out\n\n"
    return code


def _generate_code(
    inputs,
    wrapper_name,
    kernel_name,
    prod_kernel_name,
    count_kernel_name,
    init_kernel_name,
):
    shape = inputs[1].shape
    rank = len(shape)
    code = ""
    code = _generate_imports(code)
    code = _generate_scatter_reduce_kernel(rank, kernel_name, code)
    code = _generate_scatter_prod_kernel(rank, prod_kernel_name, code)
    code = _generate_count_kernel(rank, count_kernel_name, code)
    code = _generate_init_kernel(rank, init_kernel_name, code)
    code = _generate_wrapper(
        rank,
        wrapper_name,
        kernel_name,
        prod_kernel_name,
        count_kernel_name,
        init_kernel_name,
        code,
    )
    return code


class ScatterReduceFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads = {}

    def __call__(self, *args, **kwargs):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        key = f"{max_rank}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = _generate_code(
                args,
                "_scatter_reduce_wrapper",
                "_scatter_reduce_jit_function",
                "_scatter_prod_jit_function",
                "_scatter_reduce_count_jit_function",
                "_scatter_reduce_init_jit_function",
            )
            file_name = f"scatter_reduce_rank_{key}.py"
            file_path = str(_code_cache_dir() / file_name)
            write_atomic(file_path, code)
            spec = importlib.util.spec_from_file_location(
                f"_gen_scatter_reduce_module_rank_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_reduce_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)


_scatter_reduce_func = ScatterReduceFunction()


def _get_init_value(reduce, dtype, include_self):
    if include_self:
        return None
    if reduce == "sum":
        return 0
    elif reduce == "prod":
        return 1
    elif reduce == "amax":
        return float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min
    elif reduce == "amin":
        return float("inf") if dtype.is_floating_point else torch.iinfo(dtype).max
    elif reduce == "mean":
        return 0
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    out = inp
    assert reduce in ("sum", "prod", "mean", "amax", "amin"), (
        f"Unsupported reduce operation: {reduce}"
    )
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    dim = dim % inp.ndim

    init_value = None
    if not include_self:
        init_value = _get_init_value(reduce, inp.dtype, include_self)

    src_restrided = src.as_strided(index.shape, src.stride())
    N = index.numel()

    def int32_size_dim(x):
        return x.stride(dim) * x.size(dim) < 2**32

    use_int32_offset = all(int32_size_dim(t) for t in (inp, index, src))

    _scatter_reduce_func(
        src_restrided,
        index,
        inp,
        out,
        dim,
        dim_size,
        dim_stride,
        N,
        reduce,
        include_self,
        init_value,
        int32_offset=use_int32_offset,
    )

    return inp

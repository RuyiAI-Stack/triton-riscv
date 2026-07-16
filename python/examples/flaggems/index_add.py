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
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def _generate_index_add_kernel(rank, kernel_name, code):
    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    index,\n"
    code += "    src,\n"
    code += "    out,\n"
    code += "    N,\n"
    code += "    inp_numel,\n"
    code += "    inp_stride_dim,\n"
    code += "    inp_shape_dim,\n"
    code += "    src_shape_dim,\n"
    code += "    delta,\n"
    code += "    alpha,\n"
    for i in range(rank):
        code += f"    src_stride_{i},\n"
    for i in range(rank):
        code += f"    src_shape_{i},\n"
    code += "    BLOCK_SIZE: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(axis=0)\n"
    code += "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
    code += "    mask = offsets < N\n\n"
    for i in range(rank - 1, -1, -1):
        code += f"    src_offset{i} = offsets % src_shape_{i}\n"
        code += f"    offsets = offsets // src_shape_{i}\n"
    code += "\n"
    comp = " + ".join([f"src_offset{i} * src_stride_{i}" for i in range(rank)])
    code += f"    src_offset = {comp}\n\n"
    code += "    pre_cal = (inp_stride_dim * src_shape_dim)\n\n"
    code += "    pre_idx = (src_offset // pre_cal).to(tl.int64)\n"
    code += "    dim_idx = (src_offset % pre_cal // inp_stride_dim).to(tl.int64)\n"
    code += "    src_dim_idx = (tl.load(index + dim_idx, mask=mask, other=0)).to(tl.int64)\n"
    code += '    assert (src_dim_idx >= 0) & (src_dim_idx < inp_shape_dim), "0 <= index < self.size(dim)"\n'
    code += "    input_idx = (src_offset + (delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim).to(tl.int64)\n"
    code += "    input_mask = input_idx < inp_numel\n"
    code += "    add_on = tl.load(src + src_offset, mask=mask, other=0) * alpha\n"
    code += (
        "    tl.atomic_add(out + input_idx, add_on, mask=input_mask, sem='relaxed')\n\n"
    )
    return code


def _generate_wrapper(rank, wrapper_name, kernel_name, code):
    code += f"def {wrapper_name}(out, index, src, dim, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, N, inp_numel, alpha):\n"
    code += "    src_strides = list(src.stride())\n"
    code += "    src_shapes = list(src.shape)\n"
    code += "    BLOCK_SIZE = 128\n"
    code += "    grid = (triton.cdiv(N, BLOCK_SIZE),)\n"
    code += f"    {kernel_name}[grid](\n"
    code += "        index, src, out, N, inp_numel, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, alpha,\n"
    for i in range(rank):
        code += f"        src_strides[{i}],\n"
    for i in range(rank):
        code += f"        src_shapes[{i}],\n"
    code += "        BLOCK_SIZE=BLOCK_SIZE,\n"
    code += "    )\n"
    code += "    return out\n\n"
    return code


def _generate_code(inputs, wrapper_name, kernel_name):
    shape = inputs[2].shape
    rank = len(shape)
    code = ""
    code = _generate_imports(code)
    code = _generate_index_add_kernel(rank, kernel_name, code)
    code = _generate_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexAddFunction:
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
                "_index_add_wrapper",
                "_index_add_jit_function",
            )
            file_name = f"index_add_rank_{key}.py"
            file_path = str(_code_cache_dir() / file_name)
            write_atomic(file_path, code)
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_add_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)


_index_add_func = IndexAddFunction()


def index_add(inp, dim, index, src, alpha=1):
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert inp.ndim == src.ndim, (
        "Self and source should have the same number of dimensions"
    )
    for i in range(inp.ndim):
        assert (inp.size(i) == src.size(i)) or i == dim, (
            "src.size(d) == self.size(d) for all dimensions d != dim"
        )

    out = inp.clone()

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    _index_add_func(
        out,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        alpha,
    )
    return out


def index_add_(inp, dim, index, src, alpha=1):
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert inp.ndim == src.ndim, (
        "Self and source should have the same number of dimensions"
    )
    for i in range(inp.ndim):
        assert (inp.size(i) == src.size(i)) or i == dim, (
            "src.size(d) == self.size(d) for all dimensions d != dim"
        )

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    _index_add_func(
        inp,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
        alpha,
    )
    return inp

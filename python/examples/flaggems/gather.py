import importlib.util
import os
import threading
import uuid
from pathlib import Path

import torch

from .scatter import scatter_


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


def restride_dim(inp, dim, target_shape):
    shape = list(inp.shape)
    stride = list(inp.stride())
    shape[dim] = target_shape[dim]
    stride[dim] = 0
    for i in range(len(shape)):
        if shape[i] != target_shape[i]:
            if target_shape[i] == 1:
                shape[i] = 1
                stride[i] = 0
            elif shape[i] == 1:
                shape[i] = target_shape[i]
                stride[i] = 0
    return inp.as_strided(shape, stride)


def _generate_imports(code):
    code += "import torch\n"
    code += "import triton\n"
    code += "import triton.language as tl\n\n\n"
    return code


def _generate_gather_kernel(rank, kernel_name, code):
    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    inp, index, out,\n"
    for i in range(rank):
        code += f"    inp_shape{i},\n"
    for i in range(rank):
        code += f"    index_shape{i},\n"
    for i in range(rank):
        code += f"    out_shape{i},\n"
    for i in range(rank):
        code += f"    inp_stride{i},\n"
    for i in range(rank):
        code += f"    index_stride{i},\n"
    for i in range(rank):
        code += f"    out_stride{i},\n"
    code += "    dim, dim_stride, N,\n"
    code += "    BLOCK_SIZE_N: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += (
        "    offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)\n\n"
    )
    if rank == 1:
        code += "    mask = offset < N\n"
        code += "    index_offset = offset * index_stride0\n"
        code += "    cur_index = tl.load(index + index_offset, mask=mask, other=0)\n\n"
        code += "    inp_offset = cur_index * dim_stride\n"
        code += "    cur_inp = tl.load(inp + inp_offset, mask=mask, other=0)\n\n"
        code += "    out_offset = offset * out_stride0\n"
        code += "    tl.store(out + out_offset, value=cur_inp, mask=mask)\n\n"
        return code

    code += "    cur_offset = offset\n"
    for i in range(rank - 1, -1, -1):
        code += f"    index_idx{i} = cur_offset % index_shape{i}\n"
        code += f"    cur_offset = cur_offset // index_shape{i}\n"
    code += "\n"
    comp = " + ".join([f"index_idx{i} * index_stride{i}" for i in range(rank)])
    code += f"    index_offset = {comp}\n"
    code += "    mask = offset < N\n"
    code += "    cur_index = tl.load(index + index_offset, mask=mask, other=0)\n\n"
    comp = " + ".join([f"index_idx{i} * inp_stride{i}" for i in range(rank)])
    code += f"    inp_offset = {comp}\n"
    code += "    inp_offset += cur_index * dim_stride\n"
    code += "    cur_inp = tl.load(inp + inp_offset, mask=mask, other=0)\n\n"
    comp_out = " + ".join([f"index_idx{i} * out_stride{i}" for i in range(rank)])
    code += f"    out_offset = {comp_out}\n"
    code += "    tl.store(out + out_offset, value=cur_inp, mask=mask)\n\n"
    return code


def _generate_wrapper(rank, wrapper_name, kernel_name, code):
    code += f"def {wrapper_name}(inp, dim, index, out, dim_stride, N):\n"
    code += "    inp_shape = inp.shape\n"
    code += "    inp_stride = inp.stride()\n"
    code += "    index_shape = index.shape\n"
    code += "    index_stride = index.stride()\n"
    code += "    out_shape = out.shape\n"
    code += "    out_stride = out.stride()\n"
    code += "    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )\n"
    code += f"    {kernel_name}[grid](\n"
    code += "        inp, index, out,\n"
    for i in range(rank):
        code += f"        inp_shape[{i}],\n"
    for i in range(rank):
        code += f"        index_shape[{i}],\n"
    for i in range(rank):
        code += f"        out_shape[{i}],\n"
    for i in range(rank):
        code += f"        inp_stride[{i}],\n"
    for i in range(rank):
        code += f"        index_stride[{i}],\n"
    for i in range(rank):
        code += f"        out_stride[{i}],\n"
    code += "        dim, dim_stride, N,\n"
    code += "        BLOCK_SIZE_N=512,\n"
    code += "    )\n"
    code += "    return out\n\n"
    return code


def _generate_code(inputs, wrapper_name, kernel_name):
    rank = inputs[0].ndim
    code = ""
    code = _generate_imports(code)
    code = _generate_gather_kernel(rank, kernel_name, code)
    code = _generate_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads = {}

    def __call__(self, *args, **kwargs):
        key = self._arg_key(*args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = _generate_code(
                args, "_gather_wrapper", "_gather_flaggems_jit_function"
            )
            file_name = f"gather_rank_{key}.py"
            file_path = str(_code_cache_dir() / file_name)
            write_atomic(file_path, code)
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def _arg_key(self, *args):
        return args[0].ndim


_gather_func = GatherFunction()


def gather(inp, dim, index, out=None, sparse_grad=False):
    if inp.ndim != index.ndim:
        raise IndexError(
            f"self and index must have the same number of dimensions, "
            f"got self.ndim = {inp.ndim} and index.ndim = {index.ndim}"
        )
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)
    N = index.numel()
    _gather_func(inp_strided, dim, index, out, dim_stride, N)
    return out


def gather_backward(grad, self, dim, index, sparse_grad):
    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")

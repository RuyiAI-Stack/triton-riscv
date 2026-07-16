import importlib
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


# --------------------------- repeat wrapper genration -----------------------------------
def parameter_for_wrapper() -> str:
    parameters: list[str] = []
    parameters.append("in0")
    parameters.append("sizes")
    return ", ".join(parameters)


def parameter_for_wrapper_out() -> str:
    parameters: list[str] = []
    parameters.append("in0")
    parameters.append("out0")
    return ", ".join(parameters)


def parameter_ref_for_wrapper() -> str:
    parameters: list[str] = []
    parameters.append("in0")
    parameters.append("out0")
    return ", ".join(parameters)


def output_ref_for_wrapper() -> str:
    return "out0"


def generate_imports(code: str) -> str:
    code += "import math\n"
    code += "import torch\n"
    code += "import triton\n"
    code += "from triton import language as tl\n\n\n"
    return code


def generate_functional_repeat_wrapper(
    wrapper_name: str,
    destination_passing_func_name: str,
    code: str,
) -> str:
    parameters = parameter_for_wrapper()
    code += f"def {wrapper_name}({parameters}):\n"
    code += "    in0_rank = in0.dim()\n"
    code += "    sizes_rank = len(sizes)\n"
    code += "    in0_shape = list(in0.shape)\n"
    code += "    sizes_shape = list(sizes)\n\n"
    code += "    assert sizes_rank >= in0_rank, 'Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor'\n"
    code += "    if sizes_rank > in0_rank:\n"
    code += "        diff = sizes_rank - in0_rank\n"
    code += "        ones = [1 for _ in range(diff)]\n"
    code += "        in0_shape = ones + in0_shape\n"
    code += "    is_empty = False\n"
    code += "    out_shape = []\n"
    code += "    for i in range(len(in0_shape)):\n"
    code += "        assert sizes_shape[i] >= 0, 'the number of repetitions per dimension out of range (expected to >= 0) but got {}'.format(sizes_shape[i])\n"
    code += "        if in0_shape[i] * sizes_shape[i] == 0:\n"
    code += "            is_empty = True\n"
    code += "        out_shape.append(in0_shape[i] * sizes_shape[i])\n"
    code += "    out0 = torch.empty(out_shape, device=in0.device, dtype=in0.dtype)\n"
    code += "    in0 = in0.reshape(in0_shape)\n"
    code += "    if not is_empty:\n"
    code += f"        {destination_passing_func_name}({parameter_ref_for_wrapper()})\n"
    code += "    return out0\n\n"
    return code


def generate_destination_passing_repeat_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: str,
) -> str:
    parameters = parameter_for_wrapper_out()
    code += f"def {wrapper_name}({parameters}):\n"
    code += "    shape = out0.shape\n"
    code += "    num_tasks = 1\n"
    code += "    for s in shape:\n"
    code += "        num_tasks *= s\n"

    if rank > 0:
        code += "    tile_size = min(512, triton.next_power_of_2(num_tasks))\n"
        code += "    num_warps = 4\n"
        code += "    num_ctas = min(65535, triton.cdiv(num_tasks, tile_size))\n"
        code += "    tiles_per_cta = triton.cdiv(num_tasks, tile_size * num_ctas)\n"
        code += "    grid = (num_ctas,)\n\n"
        code += "    in0_strides = in0.stride()\n"
        code += "    in0_shape = in0.shape\n"
        code += "    out0_strides = out0.stride()\n\n"
        code += f"    {kernel_name}[grid](\n"
        code += "        in0, out0,\n"
        code += (
            "        "
            + ", ".join(f"in0_strides[{j}]" for j in range(rank))
            + ",  # stride for in0\n"
        )
        code += (
            "        "
            + ", ".join(f"out0_strides[{j}]" for j in range(rank))
            + ",  # stride for out0\n"
        )
        code += (
            "        "
            + ", ".join(f"shape[{i}]" for i in range(rank))
            + ",  # task indexing space\n"
        )
        code += (
            "        "
            + ", ".join(f"in0_shape[{i}]" for i in range(rank))
            + ",  # task indexing space used when input and output tensor has different shape\n"
        )
        code += "        num_tasks,\n"
        code += "        tiles_per_cta=tiles_per_cta,\n"
        code += "        tile_size=tile_size,\n"
        code += "        one_tile_per_cta=tiles_per_cta == 1,\n"
        code += "        num_warps=num_warps,\n"
        code += "    )\n"
    else:
        code += "    num_warps = 1\n"
        code += "    grid = (1,)\n\n"
        code += f"    {kernel_name}[grid](\n"
        code += "        in0, out0,\n"
        code += "        num_tasks,\n"
        code += "        num_warps=num_warps,\n"
        code += "    )\n"

    code += "    return out0\n\n"
    return code


def generate_repeat_kernel(
    rank: int,
    kernel_name: str,
    code: str,
) -> str:
    code += "@triton.jit\n"
    code += f"def {kernel_name}(\n"
    code += "    in0_ptr,\n"
    code += "    out0_ptr,\n"
    if rank > 0:
        for j in range(rank):
            code += f"    in0_stride{j}: int,\n"
        for j in range(rank):
            code += f"    out0_stride{j}: int,\n"
        for i in range(rank):
            code += f"    s{i}: int,\n"
        for i in range(rank):
            code += f"    in_s{i}: int,\n"
        code += "    num_tasks: int,\n"
        code += "    tiles_per_cta,\n"
        code += "    tile_size: tl.constexpr,\n"
        code += "    one_tile_per_cta: tl.constexpr,\n"
    else:
        code += "    num_tasks: int,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    init_tid = pid * tile_size + tl.arange(0, tile_size)\n"
    code += "    if one_tile_per_cta:\n"
    code += "        tid = init_tid\n"
    code += "        mask = tid < num_tasks\n"
    for i in reversed(range(rank)):
        if i > 0:
            code += f"        i{i} = tid % s{i}\n"
            code += f"        tid //= s{i}\n"
        else:
            code += f"        i{i} = tid\n"
    code += (
        "        in0 = tl.load(in0_ptr + "
        + " + ".join(f"(i{j} % in_s{j}) * in0_stride{j}" for j in range(rank))
        + ", mask=mask)\n"
    )
    code += (
        "        tl.store(out0_ptr + "
        + " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
        + ", in0, mask=mask)\n"
    )
    code += "    else:\n"
    code += "        for j in range(0, tiles_per_cta):\n"
    code += "            tid = init_tid + j * tile_size * tl.num_programs(0)\n"
    code += "            mask = tid < num_tasks\n"
    for i in reversed(range(rank)):
        if i > 0:
            code += f"            i{i} = tid % s{i}\n"
            code += f"            tid //= s{i}\n"
        else:
            code += f"            i{i} = tid\n"
    code += (
        "            in0 = tl.load(in0_ptr + "
        + " + ".join(f"(i{j} % in_s{j}) * in0_stride{j}" for j in range(rank))
        + ", mask=mask)\n"
    )
    code += (
        "            tl.store(out0_ptr + "
        + " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
        + ", in0, mask=mask)\n"
    )

    return code


def generate_code(
    rank: int,
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: str,
) -> str:
    code = generate_imports(code)
    code = generate_functional_repeat_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_repeat_wrapper(
        rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_repeat_kernel(rank, kernel_name, code)
    return code


class RepeatFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: dict[str, Any] = {}

    def __call__(self, x, sizes):
        ndim = self.arg_key(x, sizes)
        key = str(ndim)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = generate_code(
                ndim,
                "_wrapper",
                "_wrapper_out",
                "_repeat_jit_function",
                "",
            )

            file_name = f"repeat_rank_{key}.py"
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
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(x, sizes)

    def arg_key(self, x, sizes):
        max_rank = max(x.ndim, len(sizes))
        return max_rank


_repeat_func = RepeatFunction()


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    return _repeat_func(inp, sizes)

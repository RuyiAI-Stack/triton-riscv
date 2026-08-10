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
        path.parent / f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    with tmp_path.open("wt", encoding=encoding) as f:
        f.write(content)
    tmp_path.replace(path)


# --------------------------- padding wrapper genration -----------------------------------
def parameter_for_wrapper() -> str:
    parameters: list[str] = []
    parameters.append("in0")
    parameters.append("pad")
    parameters.append("mode")
    parameters.append("value=0")
    return ", ".join(parameters)


def parameter_for_wrapper_out() -> str:
    parameters: list[str] = []
    parameters.append("in0")
    parameters.append("out0")
    parameters.append("dst_shape")
    parameters.append("pad_before")
    parameters.append("pad_after")
    parameters.append("mode")
    parameters.append("value=0")
    return ", ".join(parameters)


def parameter_ref_for_wrapper() -> str:
    parameters: list[str] = []
    parameters.append("in0")
    parameters.append("out0")
    parameters.append("dst_shape")
    parameters.append("pad_before")
    parameters.append("pad_after")
    parameters.append("mode")
    parameters.append("value")
    return ", ".join(parameters)


def output_ref_for_wrapper() -> str:
    return "out0"


def generate_imports(code: str) -> str:
    code += "import math\n"
    code += "import torch\n"
    code += "import triton\n"
    code += "from triton import language as tl\n\n\n"
    return code


def generate_functional_padding_wrapper(
    wrapper_name: str,
    destination_passing_func_name: str,
    code: str,
) -> str:
    parameters = parameter_for_wrapper()
    code += f"def {wrapper_name}({parameters}):\n"
    code += "    ndim = in0.ndim\n"
    code += "    pad_size = len(pad)\n"
    code += "    assert pad_size % 2 == 0\n"
    code += "    pad_before = [0 for _ in range(ndim)]\n"
    code += "    pad_after = [0 for _ in range(ndim)]\n"
    code += "    pad_pair = pad_size // 2\n"
    code += "    for i in range(pad_pair):\n"
    code += "        pad_before[ndim - i - 1] = pad[2 * i]\n"
    code += "        pad_after[ndim - i - 1] = pad[2 * i + 1]\n"
    code += "    dst_shape = list(in0.shape)\n"
    code += "    for i in range(ndim):\n"
    code += "        dst_shape[i] += pad_before[i] + pad_after[i]\n"
    code += "    out0 = torch.empty(dst_shape, device=in0.device, dtype=in0.dtype)\n"
    code += f"    {destination_passing_func_name}({parameter_ref_for_wrapper()})\n"
    code += "    return out0\n\n"
    return code


def generate_destination_passing_padding_wrapper(
    rank: int,
    destination_passing_func_name: str,
    kernel_name: str,
    code: str,
) -> str:
    parameters = parameter_for_wrapper_out()
    code += f"def {destination_passing_func_name}({parameters}):\n"
    code += "    BLOCK_SIZE = 256\n"
    code += "    grid = (triton.cdiv(out0.numel(), BLOCK_SIZE),)\n"
    code += "    x_shape = in0.shape\n"
    code += "    in_strides0 = in0.stride()\n"
    code += "    out_strides = out0.stride()\n"

    for i in range(rank):
        code += f"    valid_dim{i}_start = pad_before[{i}]\n"
        code += f"    valid_dim{i}_end = dst_shape[{i}] - pad_after[{i}]\n"

    for i in range(rank):
        code += f"    dim{i}_has_pad = pad_before[{i}] > 0 or pad_after[{i}] > 0\n"

    code += "    IS_CONSTANT = mode == 'constant'\n"
    code += "    IS_REFLECT = mode == 'reflect'\n"
    code += "    IS_REPLICATE = mode == 'replicate'\n"
    code += "    IS_CIRCULAR = mode == 'circular'\n"

    code += "    " + kernel_name + "[grid](\n"
    code += "        in0, out0,\n"
    code += (
        "        "
        + ", ".join(f"x_shape[{j}]" for j in range(rank))
        + ",  # shape for x\n"
    )
    code += (
        "        "
        + ", ".join(f"in_strides0[{j}]" for j in range(rank))
        + ",  # stride for x\n"
    )
    code += (
        "        "
        + ", ".join(f"out_strides[{j}]" for j in range(rank))
        + ",  # stride for out\n"
    )
    code += (
        "        "
        + ", ".join(f"valid_dim{j}_start" for j in range(rank))
        + ",  # valid dim start\n"
    )
    code += (
        "        "
        + ", ".join(f"valid_dim{j}_end" for j in range(rank))
        + ",  # valid dim end\n"
    )
    code += (
        "        "
        + ", ".join(f"bool(dim{i}_has_pad)" for i in range(rank))
        + ",  # dim has padding flags\n"
    )
    code += "        in0.numel(), out0.numel(), value,\n"
    code += "        IS_CONSTANT, IS_REFLECT, IS_REPLICATE, IS_CIRCULAR,\n"
    code += "        BLOCK_SIZE,\n"
    code += "    )\n"
    code += "    return out0\n\n"
    return code


def generate_pad_kernel(
    rank: int,
    kernel_name: str,
    code: str,
) -> str:
    code += '@triton.jit(do_not_specialize=["value"])\n'
    code += f"def {kernel_name}(\n"
    code += "    in0_ptr,\n"
    code += "    out0_ptr,\n"
    for j in range(rank):
        code += f"    x_shape{j}: int,\n"
    for j in range(rank):
        code += f"    in_strides{j}: int,\n"
    for j in range(rank):
        code += f"    out_strides{j}: int,\n"
    for j in range(rank):
        code += f"    valid_dim{j}_start: int,\n"
    for j in range(rank):
        code += f"    valid_dim{j}_end: int,\n"
    for i in range(rank):
        code += f"    dim{i}_has_pad: tl.constexpr,\n"
    code += "    in_elem_cnt: tl.constexpr,\n"
    code += "    out_elem_cnt: tl.constexpr,\n"
    code += "    value,\n"
    code += "    IS_CONSTANT: tl.constexpr,\n"
    code += "    IS_REFLECT: tl.constexpr,\n"
    code += "    IS_REPLICATE: tl.constexpr,\n"
    code += "    IS_CIRCULAR: tl.constexpr,\n"
    code += "    BLOCK_SIZE: tl.constexpr,\n"
    code += "):\n"
    code += "    pid = tl.program_id(0)\n"
    code += "    block_offset = pid * BLOCK_SIZE\n"
    code += "    offset = block_offset + tl.arange(0, BLOCK_SIZE)\n\n"

    code += "    remaining = offset\n"
    for i in range(rank):
        code += f"    idx = remaining // out_strides{i}\n"
        code += f"    dst_index_{i} = idx\n"
        code += f"    remaining = remaining - idx * out_strides{i}\n\n"

    code += "    if_pad_false_mask = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)\n"
    code += "    if_pad_true_mask = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)\n\n"

    code += "    cond = ((dst_index_0 >= valid_dim0_start) & (dst_index_0 < valid_dim0_end))\n"
    for i in range(1, rank):
        code += f"    cond &= ((dst_index_{i} >= valid_dim{i}_start) & (dst_index_{i} < valid_dim{i}_end))\n"

    code += "    if_pad = tl.where(cond, if_pad_false_mask, if_pad_true_mask).to(tl.int1)\n\n"

    for i in range(rank):
        code += f"    src_index_{i} = dst_index_{i} - valid_dim{i}_start\n"
    for i in range(rank):
        code += f"    src_index_{i} = tl.where(src_index_{i} < 0, 0, src_index_{i})\n"

    code += "\n    if IS_REFLECT:\n"
    for i in range(rank):
        code += f"        src_index_{i} = tl.where(dim{i}_has_pad & (dst_index_{i} < valid_dim{i}_start), valid_dim{i}_start - dst_index_{i}, src_index_{i})\n"
    for i in range(rank):
        code += f"        src_index_{i} = tl.where(dim{i}_has_pad & (dst_index_{i} >= valid_dim{i}_end), (x_shape{i} + valid_dim{i}_start - 1) * 2 - dst_index_{i} - valid_dim{i}_start, src_index_{i})\n"

    code += "\n    if IS_REPLICATE:\n"
    for i in range(rank):
        code += f"        src_index_{i} = tl.where(dim{i}_has_pad & (dst_index_{i} < valid_dim{i}_start), 0, src_index_{i})\n"
    for i in range(rank):
        code += f"        src_index_{i} = tl.where(dim{i}_has_pad & (dst_index_{i} >= valid_dim{i}_end), x_shape{i} - 1, src_index_{i})\n"

    code += "\n    if IS_CIRCULAR:\n"
    for i in range(rank):
        code += f"        src_index_{i} = tl.where(dim{i}_has_pad & (dst_index_{i} < valid_dim{i}_start), dst_index_{i} + x_shape{i} - valid_dim{i}_start, src_index_{i})\n"
    for i in range(rank):
        code += f"        src_index_{i} = tl.where(dim{i}_has_pad & (dst_index_{i} >= valid_dim{i}_end), dst_index_{i} - valid_dim{i}_end, src_index_{i})\n"

    code += "\n    src_offset = src_index_0 * in_strides0\n"
    for i in range(1, rank):
        code += f"    src_offset += src_index_{i} * in_strides{i}\n"

    code += "    load_cond = src_index_0 < x_shape0\n"
    for i in range(1, rank):
        code += f"    load_cond &= src_index_{i} < x_shape{i}\n"

    code += "\n    if IS_CONSTANT:\n"
    code += "        x_val = tl.load(in0_ptr + src_offset, mask=((if_pad == 0) & load_cond), other=value)\n"
    code += "    else:\n"
    code += "        x_val = tl.load(in0_ptr + src_offset, mask=load_cond, other=0)\n"
    code += "    tl.store(out0_ptr + offset, x_val, mask=offset < out_elem_cnt)\n"

    return code


def generate_code(
    inputs: tuple[Any],
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: str,
) -> str:
    shape = inputs[0].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_functional_padding_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_padding_wrapper(
        rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_pad_kernel(rank, kernel_name, code)
    return code


class PadFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = generate_code(
                args,
                "_pad_wrapper",
                "_pad_wrapper_out",
                "_pad_jit_function",
                "",
            )

            file_name = f"constant_pad_rank_{key}.py"
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
            overload = getattr(m, "_pad_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_pad_func = PadFunction()


def pad(self, pad, mode="constant", value=None):
    ndim = self.ndim

    if value is None:
        value = 0.0

    pad_pairs = len(pad) // 2

    if mode == "reflect":
        for i in range(pad_pairs):
            pad_l, pad_r = pad[2 * i], pad[2 * i + 1]
            input_size = self.shape[ndim - 1 - i]
            assert pad_l < input_size and pad_r < input_size, (
                f"padding size should be less than the corresponding input dimension, \
                 but got padding size: {pad_l}, {pad_r}, input size: {self.shape}"
            )

    if mode == "circular":
        for i in range(pad_pairs):
            pad_l, pad_r = pad[2 * i], pad[2 * i + 1]
            input_size = self.shape[ndim - 1 - i]
            assert pad_l <= input_size and pad_r <= input_size, (
                "Padding value causes wrapping around more than once."
            )

    out = _pad_func(self, pad, mode, float(value))
    return out


def constant_pad_nd(self, pad_list, value=0):
    return pad(self, pad_list, mode="constant", value=value)

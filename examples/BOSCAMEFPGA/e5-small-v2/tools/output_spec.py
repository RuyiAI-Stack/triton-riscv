"""Output arguments exported by the E5 fixed operator launchers.

Indices refer to ``case["arguments"]`` and the typed kernel entry. Quantizers
export both their INT8 tensor and their FP32 scale tensor; replay must load
both before invoking the unchanged launcher oracle.
"""

import importlib.util
from functools import cache
from pathlib import Path


@cache
def _describe():
    path = Path(__file__).resolve().parents[1] / "cases.py"
    spec = importlib.util.spec_from_file_location("e5_output_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.describe


_OUTPUTS = {
    "add": ((2, "Out"),),
    "attention_mask": ((2, "Out"),),
    "bias_add": ((2, "Out"),),
    "dequantization": ((3, "Out"),),
    "embedding_add": ((5, "Out"),),
    "embedding_lookup": ((2, "Out"),),
    "gelu": ((1, "Out"),),
    "head_layout": ((1, "Out"),),
    "l2_normalize": ((1, "Out"),),
    "layernorm": ((3, "Out"),),
    "matmul_i8": ((2, "C"),),
    "mean_pool": ((2, "Out"),),
    "per_token_quantization": ((1, "Q"), (2, "Scale")),
    "quantize_transpose": ((1, "Q"), (2, "Scale")),
    "softmax": ((1, "Out"),),
}


def output_indices(case):
    """Return every written argument, rejecting an unreviewed family or ABI."""
    if isinstance(case, str):
        case = _describe()(case)
    family = case["family"]
    if family not in _OUTPUTS:
        raise ValueError(f"no E5 output specification for {family!r}")
    expected = _OUTPUTS[family]
    arguments = case["arguments"]
    for index, name in expected:
        if index >= len(arguments) or arguments[index]["name"] != name:
            raise ValueError(f"{case['name']}: output argument {index} must be {name}")
    return tuple(index for index, _ in expected)

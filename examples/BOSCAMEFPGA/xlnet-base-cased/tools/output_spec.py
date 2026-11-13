"""Output argument indices for scheduler DDR exports and native replay.

The entries name the real kernel.py ABI outputs. Unknown families and changed
argument lists fail explicitly so an input cannot silently become an export.
"""

import importlib.util
from functools import cache
from pathlib import Path


@cache
def _case_reader():
    # Load this model's inventory by path: several sibling models have a
    # module named cases, and a shared download driver can inspect them all.
    path = Path(__file__).resolve().parents[1] / "cases.py"
    spec = importlib.util.spec_from_file_location("xlnet_output_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.describe


_OUTPUT_ARGUMENTS = {
    "add": ("Out",),
    "attention_score": ("Out",),
    "bias_add": ("Out",),
    "dequantization": ("Out",),
    "embedding": ("Out",),
    "gelu": ("Out",),
    "head_bias_add": ("Out",),
    "layernorm": ("Out",),
    "matmul_i8": ("C",),
    "merge_heads": ("Out",),
    "per_token_quantization": ("Q", "Scale"),
    "quantize_transpose": ("Q", "Scale"),
    "relative_shift": ("Out",),
    "segment_gather": ("Out",),
    "softmax": ("Out",),
    "split_heads": ("Out",),
}


def output_indices(case):
    """Return ordered output indices for a case name (or descriptor dict)."""
    if isinstance(case, str):
        case = _case_reader()(case)
    family, name = case["family"], case["name"]
    try:
        outputs = _OUTPUT_ARGUMENTS[family]
    except KeyError as error:
        raise ValueError(f"{name}: unsupported output family {family!r}") from error
    arguments = [argument["name"] for argument in case["arguments"]]
    if len(arguments) != len(set(arguments)):
        raise ValueError(f"{name}: duplicate ABI argument names")
    missing = [output for output in outputs if output not in arguments]
    if missing:
        raise ValueError(f"{name}: missing ABI output arguments {missing}")
    return [arguments.index(output) for output in outputs]

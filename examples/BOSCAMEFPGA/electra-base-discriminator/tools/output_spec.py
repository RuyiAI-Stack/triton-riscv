"""Output descriptor positions in the original ELECTRA typed launch ABI."""

from pathlib import Path
import runpy

_DESCRIBE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "cases.py"))[
    "describe"
]
_INDICES = {
    "add": (2,),
    "bias_add": (2,),
    "dequantization": (3,),
    "embedding": (5,),
    "gelu": (1,),
    "layernorm": (3,),
    "masked_softmax": (2,),
    "matmul_i8": (2,),
    "pack_heads": (1,),
    "unpack_heads": (1,),
    "per_token_quantization": (1, 2),
    "quantize_transpose": (1, 2),
}


def output_indices(case_name):
    case = _DESCRIBE(case_name)
    try:
        result = _INDICES[case["family"]]
    except KeyError as error:
        raise ValueError("Missing output ABI: " + case_name) from error
    expected = (
        ("Q", "Scale")
        if len(result) == 2
        else ("C",)
        if case["family"] == "matmul_i8"
        else ("Out",)
    )
    if tuple(case["arguments"][i]["name"] for i in result) != expected:
        raise ValueError("Output ABI changed: " + case_name)
    return result

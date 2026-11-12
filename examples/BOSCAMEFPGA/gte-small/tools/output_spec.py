"""Output arguments exported by each GTE standalone typed adapter.

Indices are zero based in the original ``_mlir_ciface_kernel_*`` call. Both
INT8 values and FP32 scales are outputs of quantization, including transpose.
"""

OUTPUT_INDICES = {
    "add_128x384": (2,),
    "attention_mask_12x128x128": (2,),
    "bias_add_128x384": (2,),
    "bias_add_128x1536": (2,),
    "dequantize_128x32": (3,),
    "dequantize_128x128": (3,),
    "dequantize_128x384": (3,),
    "dequantize_128x1536": (3,),
    "embedding_add_128x384": (5,),
    "embedding_lookup_128x384": (2,),
    "gelu_128x1536": (1,),
    "l2_normalize_1x384": (1,),
    "layernorm_128x384": (3,),
    "matmul_i8_128x128x32": (2,),
    "matmul_i8_128x32x128": (2,),
    "matmul_i8_128x384x384": (2,),
    "matmul_i8_128x384x1536": (2,),
    "matmul_i8_128x1536x384": (2,),
    "mean_pool_128x384": (2,),
    "merge_heads_12x128x32": (1,),
    "quantize_128x32": (1, 2),
    "quantize_128x128": (1, 2),
    "quantize_128x384": (1, 2),
    "quantize_128x1536": (1, 2),
    "quantize_384x384": (1, 2),
    "quantize_384x1536": (1, 2),
    "quantize_1536x384": (1, 2),
    "quantize_transpose_128x32": (1, 2),
    "softmax_12x128x128": (1,),
    "split_heads_128x384": (1,),
}


def output_indices(case: str) -> tuple[int, ...]:
    """Return all output descriptor positions, rejecting unknown cases."""
    try:
        return OUTPUT_INDICES[case]
    except KeyError as error:
        raise ValueError(f"Unknown GTE operator case: {case}") from error

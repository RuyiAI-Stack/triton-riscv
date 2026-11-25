def _validate_sparse_coo_tensor_args(indices, values, size):
    if indices.ndim != 2:
        raise RuntimeError("indices must be 2D")
    if indices.shape[1] != values.shape[0]:
        raise RuntimeError("nnz mismatch")
    if indices.min() < 0 or (indices[0].max() >= size[0]):
        raise RuntimeError("index out of range")
    return True

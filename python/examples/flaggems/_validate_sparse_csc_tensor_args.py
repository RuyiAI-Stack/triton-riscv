def _validate_sparse_csc_tensor_args(ccol_indices, row_indices, values, size):
    if ccol_indices[0] != 0 or ccol_indices[-1] != values.shape[0]:
        raise RuntimeError("compressed index does not match nnz")
    if row_indices.numel() and (row_indices.min() < 0 or row_indices.max() >= size[0]):
        raise RuntimeError("row index out of range")
    return True

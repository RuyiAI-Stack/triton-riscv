import torch

from .detach_copy import _copy_values


def sparse_compressed_tensor(
    compressed_indices, plain_indices, values, size, *, layout=torch.sparse_csr
):
    copied_c = _copy_values(compressed_indices.to(torch.int64))
    copied_p = _copy_values(plain_indices.to(torch.int64))
    copied_v = _copy_values(values)
    if layout == torch.sparse_csc:
        return torch.sparse_csc_tensor(
            copied_c, copied_p, copied_v, size, device=values.device
        )
    return torch.sparse_csr_tensor(
        copied_c, copied_p, copied_v, size, device=values.device
    )

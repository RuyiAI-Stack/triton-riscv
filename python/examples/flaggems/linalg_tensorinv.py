from .inverse import inverse


def linalg_tensorinv(A, ind=2):
    left = 1
    for dim in A.shape[:ind]:
        left *= dim
    mat = A.reshape(left, A.numel() // left)
    inv = inverse(mat)
    out_shape = A.shape[ind:] + A.shape[:ind]
    return inv.reshape(out_shape)


__all__ = ["linalg_tensorinv"]

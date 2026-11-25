from .unique_dim import unique_dim


def unique_dim_consecutive(input, dim=0, return_inverse=False, return_counts=False):
    return unique_dim(
        input,
        dim=dim,
        sorted=False,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

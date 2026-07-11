from .full import full


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    return full(
        x.size(),
        fill_value,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

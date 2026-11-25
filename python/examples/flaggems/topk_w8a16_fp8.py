def topk_w8a16_fp8(x, k, scale=None, dim=-1, largest=True):
    from .topk import topk

    values = x.float()
    if scale is not None:
        values = values * scale.float()
    return topk(values, k, dim=dim, largest=largest)

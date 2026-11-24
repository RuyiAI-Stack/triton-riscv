import torch


def spd():
    raw = torch.randn(3, 3)
    return raw @ raw.T + 0.5 * torch.eye(3)


def det3(a):
    return (
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )

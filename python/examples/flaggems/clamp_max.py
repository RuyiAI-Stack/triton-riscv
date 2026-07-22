from .clamp import clamp_max as _clamp_max


def clamp_max(A, maxi):
    return _clamp_max(A, maxi)


def clamp_max_(A, maxi):
    result = clamp_max(A, maxi)
    A.copy_(result)
    return A

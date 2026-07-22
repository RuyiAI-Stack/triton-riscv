from .fmod import _invoke_fmod_kernel


def fmod(A, B):
    return _invoke_fmod_kernel(A, B)


def fmod_(A, B):
    assert A.dtype.is_floating_point, "fmod_ only supports floating point dtypes"
    return _invoke_fmod_kernel(A, B, out=A)

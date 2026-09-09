from .addmm import addmm


def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
    assert self.dtype.is_floating_point, "Only floating-point dtypes are supported"
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    result = addmm(self, mat1, mat2, beta=beta, alpha=alpha)
    self.copy_(result)
    return self

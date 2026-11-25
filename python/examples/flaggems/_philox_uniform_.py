from .detach_copy import _copy_values
from .randint import _philox_uniform


def _philox_uniform_(self):
    filled = _philox_uniform(self.numel(), self.device).view(self.shape).to(self.dtype)
    self.copy_(_copy_values(filled))
    return self

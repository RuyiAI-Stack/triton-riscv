from .detach_copy import _copy_values
from .log_normal import log_normal


def log_normal_(self, mean=0.0, std=1.0):
    filled = log_normal(
        *self.shape, mean=mean, std=std, dtype=self.dtype, device=self.device
    )
    self.copy_(_copy_values(filled))
    return self

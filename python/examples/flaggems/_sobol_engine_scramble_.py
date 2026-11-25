import torch

from .detach_copy import _copy_values


def _sobol_engine_scramble_(state, ltm):
    return _copy_values((state.to(torch.int32) ^ ltm.to(torch.int32)).contiguous())

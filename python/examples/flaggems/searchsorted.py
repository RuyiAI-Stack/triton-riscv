import torch

from .bucketize import _bounded


def searchsorted(
    sorted_sequence, self, *, out_int32=False, right=False, side="left", sorter=None
):
    if side == "right":
        right = True
    if sorter is not None:
        sorted_sequence = sorted_sequence.gather(-1, sorter)
    result = _bounded(self, sorted_sequence, right=right)
    return result.to(torch.int32) if out_int32 else result

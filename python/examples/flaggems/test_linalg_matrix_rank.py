import torch

from ._refdet import spd
from .linalg_matrix_rank import linalg_matrix_rank


def test_linalg_matrix_rank():
    torch.manual_seed(0)
    a = spd()
    assert int(linalg_matrix_rank(a)) == 3

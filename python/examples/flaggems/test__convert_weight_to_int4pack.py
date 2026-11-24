import pytest
import torch

from . import _convert_weight_to_int4pack
from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__convert_weight_to_int4pack(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_convert_weight_to_int4pack", variant)


@pytest.mark.parametrize("columns", [128, 130, 514])
@pytest.mark.parametrize("strided", [False, True])
def test_int4pack_columns(monkeypatch, columns, strided):
    torch.manual_seed(1401)
    storage = torch.randint(-8, 8, (3, columns * 2), dtype=torch.int32)
    x = storage[:, ::2]
    if not strided:
        x = x.contiguous()
    expected = ((x[:, 0::2] & 15) | ((x[:, 1::2] & 15) << 4)).to(torch.uint8)

    def packing_case(name, variant):
        return _convert_weight_to_int4pack(x, innerKTiles=2), expected

    check_case(monkeypatch, packing_case, "_convert_weight_to_int4pack", 0)

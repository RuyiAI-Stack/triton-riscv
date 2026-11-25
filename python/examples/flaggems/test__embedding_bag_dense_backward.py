import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__embedding_bag_dense_backward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_embedding_bag_dense_backward", variant)

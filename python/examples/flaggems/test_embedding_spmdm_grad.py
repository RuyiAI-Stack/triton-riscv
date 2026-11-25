import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_embedding_spmdm_grad(monkeypatch, variant):
    check_case(monkeypatch, run_case, "embedding_spmdm_grad", variant)

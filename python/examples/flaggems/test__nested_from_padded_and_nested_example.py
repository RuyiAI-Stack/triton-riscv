import torch

from ._nested_from_padded_and_nested_example import (
    _nested_from_padded_and_nested_example,
)


def test__nested_from_padded_and_nested_example():
    padded = torch.arange(4.0)
    torch.testing.assert_close(
        _nested_from_padded_and_nested_example(padded, None),
        padded,
        rtol=1e-4,
        atol=1e-4,
    )

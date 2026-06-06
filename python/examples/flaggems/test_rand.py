import pytest
import torch

from .rand import rand


@pytest.mark.parametrize("shape", [(10,), (512,), (1023,), (1024,), (2, 512)])
def test_rand(shape):
    torch.manual_seed(0)

    out_triton = rand(shape, dtype=torch.float32)

    # For random functions, we can't assert exact values easily against PyTorch
    # without matching the exact PRNG state. So we just check properties.
    assert out_triton.shape == shape
    assert out_triton.dtype == torch.float32
    assert torch.all(out_triton >= 0.0)
    assert torch.all(out_triton < 1.0)


def test_rand_repeated_calls_advance_philox_offset():
    torch.manual_seed(0)

    first = rand((1024,), dtype=torch.float32)
    second = rand((1024,), dtype=torch.float32)

    assert not torch.equal(first, second)


def test_rand_empty_does_not_advance_default_generator():
    torch.manual_seed(0)
    state_before = torch.random.get_rng_state()

    out = rand((0,), dtype=torch.float32)

    assert out.numel() == 0
    torch.testing.assert_close(torch.random.get_rng_state(), state_before)

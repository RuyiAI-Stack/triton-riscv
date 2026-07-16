import pytest
import torch

from .exponential_ import exponential_


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("lambd", [1.0, 0.5, 2.0])
def test_exponential_(shape, lambd):
    torch.manual_seed(0)
    x = torch.zeros(shape, dtype=torch.float32, device="cpu")

    # Run triton implementation
    out = exponential_(x, lambd=lambd)

    # We can't do assert_close since it's a random generation
    # But we can verify statistical properties

    # Check mean: expected value is 1/lambd
    mean_val = out.mean().item()
    expected_mean = torch.distributions.Exponential(torch.tensor(lambd)).mean.item()
    # Allow 10% variance
    assert abs(mean_val - expected_mean) / expected_mean < 0.1

    # Check minimum value is >= 0
    assert out.min().item() >= 0

    # Verify no NaN or Inf
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_exponential_repeated_calls_advance_philox_offset():
    torch.manual_seed(0)
    first = torch.empty((4096,), dtype=torch.float32, device="cpu")
    second = torch.empty((4096,), dtype=torch.float32, device="cpu")

    exponential_(first)
    exponential_(second)

    assert not torch.equal(first, second)


def test_exponential_generator_state_is_consumed():
    first_generator = torch.Generator(device="cpu").manual_seed(0)
    second_generator = torch.Generator(device="cpu").manual_seed(0)
    first = torch.empty((4096,), dtype=torch.float32, device="cpu")
    repeat_from_same_seed = torch.empty((4096,), dtype=torch.float32, device="cpu")
    second = torch.empty((4096,), dtype=torch.float32, device="cpu")

    exponential_(first, generator=first_generator)
    exponential_(repeat_from_same_seed, generator=second_generator)
    exponential_(second, generator=first_generator)

    torch.testing.assert_close(first, repeat_from_same_seed)
    assert not torch.equal(first, second)


def test_exponential_empty_does_not_advance_generator():
    generator = torch.Generator(device="cpu").manual_seed(0)
    state_before = generator.get_state()
    out = torch.empty(0, dtype=torch.float32, device="cpu")

    exponential_(out, generator=generator)

    torch.testing.assert_close(generator.get_state(), state_before)

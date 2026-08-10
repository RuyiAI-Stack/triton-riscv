import pytest
import torch

from .uniform import uniform, uniform_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_uniform(size, dtype):
    out = uniform(size, from_=0.0, to=1.0, dtype=dtype)
    assert out.shape == (size,)
    assert out.dtype == dtype
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_uniform_(size, dtype):
    x = torch.empty(size, dtype=dtype, device="cpu")
    uniform_(x, from_=0.0, to=1.0)
    assert (x >= 0.0).all() and (x <= 1.0).all()


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_uniform_range(size, dtype):
    out = uniform(size, from_=-5.0, to=5.0, dtype=dtype)
    assert out.dtype == dtype
    assert (out >= -5.0).all() and (out <= 5.0).all()


def test_uniform_repeated_calls_advance_philox_offset():
    torch.manual_seed(0)
    first = uniform(1024, dtype=torch.float32)
    second = uniform(1024, dtype=torch.float32)

    assert not torch.equal(first, second)


def test_uniform_generator_state_is_consumed():
    first_generator = torch.Generator(device="cpu").manual_seed(0)
    second_generator = torch.Generator(device="cpu").manual_seed(0)
    first = torch.empty(1024, dtype=torch.float32, device="cpu")
    repeat_from_same_seed = torch.empty(1024, dtype=torch.float32, device="cpu")
    second = torch.empty(1024, dtype=torch.float32, device="cpu")

    uniform_(first, generator=first_generator)
    uniform_(repeat_from_same_seed, generator=second_generator)
    uniform_(second, generator=first_generator)

    torch.testing.assert_close(first, repeat_from_same_seed)
    assert not torch.equal(first, second)


def test_uniform_empty_does_not_advance_generator():
    generator = torch.Generator(device="cpu").manual_seed(0)
    state_before = generator.get_state()
    out = torch.empty(0, dtype=torch.float32, device="cpu")

    uniform_(out, generator=generator)

    torch.testing.assert_close(generator.get_state(), state_before)

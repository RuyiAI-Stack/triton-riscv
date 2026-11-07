import pytest
import torch

from .rrelu_with_noise_functional import rrelu_with_noise_functional


def test_rrelu_with_noise_functional_eval_matches_torch():
    x = torch.tensor([-2.0, -1.0, 0.0, 3.0], device="cpu")
    noise = torch.full_like(x, 0.25)
    lower = 0.1
    upper = 0.3

    out, returned_noise = rrelu_with_noise_functional(
        x, noise, lower, upper, training=False
    )
    ref, ref_noise = torch.ops.aten.rrelu_with_noise_functional(
        x, noise, lower, upper, False, None
    )

    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(returned_noise, ref_noise)


def test_rrelu_with_noise_functional_training_matches_torch_contract():
    x = torch.cat((torch.linspace(-2.0, -0.1, 1023), torch.linspace(0.0, 2.0, 1023)))
    noise = torch.full_like(x, 0.25)
    lower = 0.1
    upper = 0.3

    out, returned_noise = rrelu_with_noise_functional(
        x, noise, lower=lower, upper=upper, training=True
    )
    ref, ref_noise = torch.ops.aten.rrelu_with_noise_functional(
        x, noise, lower, upper, True, None
    )

    torch.testing.assert_close(out, torch.where(x >= 0, x, x * returned_noise))
    assert torch.equal(returned_noise[x > 0], torch.ones_like(returned_noise[x > 0]))
    assert torch.equal(ref_noise[x > 0], torch.ones_like(ref_noise[x > 0]))
    assert torch.all(
        (returned_noise[x <= 0] >= lower) & (returned_noise[x <= 0] <= upper)
    )
    torch.testing.assert_close(
        returned_noise[x <= 0].mean(), ref_noise[x <= 0].mean(), rtol=0, atol=1e-2
    )
    torch.testing.assert_close(
        returned_noise[x <= 0].var(), ref_noise[x <= 0].var(), rtol=0, atol=1e-3
    )
    torch.testing.assert_close(out.mean(), ref.mean(), rtol=0, atol=1e-2)
    assert returned_noise.data_ptr() != noise.data_ptr()


def test_rrelu_with_noise_functional_empty_matches_torch():
    x = torch.empty(0, dtype=torch.float32, device="cpu")
    noise = torch.empty_like(x)

    out, returned_noise = rrelu_with_noise_functional(x, noise, training=True)
    ref, ref_noise = torch.ops.aten.rrelu_with_noise_functional(
        x, noise, 0.125, 1.0 / 3.0, True, None
    )

    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(returned_noise, ref_noise)


def test_rrelu_with_noise_functional_training_generator_is_reproducible():
    x = torch.linspace(-2.0, -0.1, 1023, dtype=torch.float32, device="cpu")
    noise = torch.empty_like(x)
    first_generator = torch.Generator(device="cpu").manual_seed(0)
    second_generator = torch.Generator(device="cpu").manual_seed(0)

    out, returned_noise = rrelu_with_noise_functional(
        x, noise, training=True, generator=first_generator
    )
    repeated_out, repeated_noise = rrelu_with_noise_functional(
        x, noise, training=True, generator=second_generator
    )

    torch.testing.assert_close(out, repeated_out)
    torch.testing.assert_close(returned_noise, repeated_noise)


@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_functional_non_contiguous_matches_torch(training):
    x = torch.arange(-6.0, 6.0, dtype=torch.float32, device="cpu").reshape(3, 4).T
    noise = torch.full_like(x, 0.25)

    out, returned_noise = rrelu_with_noise_functional(
        x, noise, lower=0.1, upper=0.3, training=training
    )
    ref, ref_noise = torch.ops.aten.rrelu_with_noise_functional(
        x, noise, 0.1, 0.3, training, None
    )

    assert out.stride() == ref.stride()
    assert returned_noise.stride() == ref_noise.stride()
    if training:
        torch.testing.assert_close(out, torch.where(x >= 0, x, x * returned_noise))
        assert torch.equal(
            returned_noise[x > 0], torch.ones_like(returned_noise[x > 0])
        )
        assert torch.all(
            (returned_noise[x <= 0] >= 0.1) & (returned_noise[x <= 0] <= 0.3)
        )
    else:
        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(returned_noise, ref_noise)

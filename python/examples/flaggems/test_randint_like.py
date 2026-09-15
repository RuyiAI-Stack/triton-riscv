import pytest
import torch

from .randint_like import randint_like


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_randint_like_matches_torch_distribution(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    seed = 0
    out_generator = torch.Generator(device="cpu").manual_seed(seed)
    ref_generator = torch.Generator(device="cpu").manual_seed(seed)

    # Same as torch.randint for distribution comparison (bounds and dtype).
    ref_out = torch.randint(
        0,
        5,
        x.shape,
        dtype=x.dtype,
        device=x.device,
        generator=ref_generator,
    )
    tri_out = randint_like(x, 5, generator=out_generator)

    assert tri_out.shape == x.shape
    assert tri_out.dtype == x.dtype
    assert torch.all(tri_out >= 0)
    assert torch.all(tri_out < 5)
    torch.testing.assert_close(
        tri_out.float().mean(), ref_out.float().mean(), rtol=0.2, atol=0.2
    )
    torch.testing.assert_close(
        tri_out.float().var(unbiased=False),
        ref_out.float().var(unbiased=False),
        rtol=0.2,
        atol=0.2,
    )


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_randint_like_properties(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")

    tri_out = randint_like(x, 5)

    assert tri_out.shape == x.shape
    assert tri_out.dtype == x.dtype
    assert torch.all(tri_out >= 0)
    assert torch.all(tri_out < 5)


@pytest.mark.parametrize("shape", [(0,), (0, 10)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_randint_like_empty(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    tri_out = randint_like(x, 10)

    assert tri_out.shape == x.shape
    assert tri_out.numel() == 0


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_randint_like_reproducible_generator(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    generator_a = torch.Generator(device="cpu").manual_seed(1234)
    generator_b = torch.Generator(device="cpu").manual_seed(1234)

    tri_out = randint_like(x, 5, generator=generator_a)
    ref_out = randint_like(x, 5, generator=generator_b)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_randint_like_scalar_high_one(dtype):
    x = torch.empty((), dtype=dtype, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(999)

    out = randint_like(x, torch.tensor(1), generator=generator)

    assert out.shape == ()
    assert out.item() == 0


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(7,), (3, 5)])
def test_randint_like_high_one_returns_zeros(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device="cpu")
    out = randint_like(x, 1, generator=torch.Generator(device="cpu").manual_seed(101))

    assert out.shape == shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, torch.zeros(shape, dtype=dtype, device="cpu"))

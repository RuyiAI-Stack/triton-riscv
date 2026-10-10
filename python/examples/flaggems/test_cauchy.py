import pytest
import torch

from .cauchy import cauchy, cauchy_


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8192,), (4096,)])
def test_cauchy_matches_torch_distribution(dtype, shape):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    seed = 0
    out_generator = torch.Generator(device="cpu").manual_seed(seed)
    ref_generator = torch.Generator(device="cpu").manual_seed(seed)

    ref_out = torch.empty_like(x)
    ref_out = ref_out.cauchy_(median=0.0, sigma=1.0, generator=ref_generator)

    tri_out = cauchy(
        torch.empty_like(x),
        median=0.0,
        sigma=1.0,
        generator=out_generator,
    )

    assert tri_out.shape == x.shape
    assert tri_out.dtype == x.dtype
    assert torch.isfinite(tri_out).all()
    assert torch.isfinite(ref_out).all()
    torch.testing.assert_close(
        tri_out.abs().mean(), ref_out.abs().mean(), rtol=0.25, atol=0.5
    )
    torch.testing.assert_close(
        (tri_out.abs() < 1).float().mean(),
        (ref_out.abs() < 1).float().mean(),
        rtol=0.2,
        atol=0.1,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1024,), (512,)])
def test_cauchy_properties(dtype, shape):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")

    tri_out = cauchy(x, median=0.0, sigma=1.0)
    ret = cauchy_(x, median=0.0, sigma=1.0)

    assert ret is x
    assert tri_out.shape == x.shape
    assert tri_out.dtype == x.dtype
    assert torch.isfinite(tri_out).all()
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(256,), (128,)])
def test_cauchy_generator_reproducible(dtype, shape):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    gen_a = torch.Generator(device="cpu").manual_seed(321)
    gen_b = torch.Generator(device="cpu").manual_seed(321)

    tri_out = cauchy(torch.empty_like(x), median=0.0, sigma=1.0, generator=gen_a)
    ref_out = cauchy(torch.empty_like(x), median=0.0, sigma=1.0, generator=gen_b)

    torch.testing.assert_close(tri_out, ref_out)
    assert torch.isfinite(tri_out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cauchy_inplace_non_contiguous(dtype):
    x = torch.empty((513, 128), dtype=dtype, device="cpu")[::2]
    generator = torch.Generator(device="cpu").manual_seed(123)
    contiguous_generator = torch.Generator(device="cpu").manual_seed(123)
    torch_generator = torch.Generator(device="cpu").manual_seed(456)
    contiguous = torch.empty(x.shape, dtype=dtype, device="cpu")
    torch_ref = torch.empty((513, 128), dtype=dtype, device="cpu")[::2]

    cauchy_(x, median=0.5, sigma=1.5, generator=generator)
    cauchy_(contiguous, median=0.5, sigma=1.5, generator=contiguous_generator)
    torch_ref.cauchy_(median=0.5, sigma=1.5, generator=torch_generator)

    assert not x.is_contiguous()
    torch.testing.assert_close(x, contiguous)
    torch.testing.assert_close(x.median(), torch_ref.median(), rtol=0.1, atol=0.1)
    torch.testing.assert_close(
        (x.sub(0.5).abs() < 1.5).float().mean(),
        (torch_ref.sub(0.5).abs() < 1.5).float().mean(),
        rtol=0.1,
        atol=0.05,
    )

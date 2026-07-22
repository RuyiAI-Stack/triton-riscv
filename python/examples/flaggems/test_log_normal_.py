import pytest
import torch

from .log_normal_ import log_normal_


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_normal_matches_torch_distribution(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    mean = 1.0
    std = 2.0
    seed = 0
    out_generator = torch.Generator(device="cpu").manual_seed(seed)
    ref_generator = torch.Generator(device="cpu").manual_seed(seed)

    ref_out = torch.empty_like(x)
    ref_out = ref_out.log_normal_(mean=mean, std=std, generator=ref_generator)

    tri_out = log_normal_(
        torch.empty_like(x), mean=mean, std=std, generator=out_generator
    )

    assert tri_out.shape == x.shape
    assert tri_out.dtype == x.dtype
    assert torch.all(tri_out > 0)
    assert torch.all(ref_out > 0)
    torch.testing.assert_close(
        tri_out.log().mean(), ref_out.log().mean(), rtol=0.2, atol=0.2
    )
    torch.testing.assert_close(
        tri_out.log().std(unbiased=False),
        ref_out.log().std(unbiased=False),
        rtol=0.2,
        atol=0.5,
    )


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_normal_inplace_properties(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")

    ret = log_normal_(x, mean=0.0, std=1.0)

    assert ret is x
    assert x.shape == shape
    assert torch.isfinite(x).all()
    assert torch.all(x > 0)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_normal_generator_reproducible(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    g_a = torch.Generator(device="cpu").manual_seed(888)
    g_b = torch.Generator(device="cpu").manual_seed(888)

    tri_out = log_normal_(torch.empty_like(x), mean=0.0, std=1.0, generator=g_a)
    ref_out = log_normal_(torch.empty_like(x), mean=0.0, std=1.0, generator=g_b)

    torch.testing.assert_close(tri_out, ref_out)
    assert torch.all(tri_out > 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_normal_inplace_non_contiguous(dtype):
    x = torch.empty((513, 128), dtype=dtype, device="cpu")[::2]
    generator = torch.Generator(device="cpu").manual_seed(789)
    contiguous_generator = torch.Generator(device="cpu").manual_seed(789)
    torch_generator = torch.Generator(device="cpu").manual_seed(321)
    contiguous = torch.empty(x.shape, dtype=dtype, device="cpu")
    torch_ref = torch.empty((513, 128), dtype=dtype, device="cpu")[::2]

    log_normal_(x, mean=0.25, std=1.5, generator=generator)
    log_normal_(contiguous, mean=0.25, std=1.5, generator=contiguous_generator)
    torch_ref.log_normal_(mean=0.25, std=1.5, generator=torch_generator)

    assert not x.is_contiguous()
    torch.testing.assert_close(x, contiguous)
    torch.testing.assert_close(
        x.log().mean(), torch_ref.log().mean(), rtol=0.1, atol=0.1
    )
    torch.testing.assert_close(
        x.log().std(unbiased=False),
        torch_ref.log().std(unbiased=False),
        rtol=0.1,
        atol=0.1,
    )

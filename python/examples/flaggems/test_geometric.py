import pytest
import torch

from .geometric import geometric, geometric_


@pytest.mark.parametrize("shape", [(8192,), (16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometric_matches_torch_distribution(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    seed = 0
    out_generator = torch.Generator(device="cpu").manual_seed(seed)
    ref_generator = torch.Generator(device="cpu").manual_seed(seed)

    ref_out = torch.empty_like(x)
    ref_out = ref_out.geometric_(p=0.5, generator=ref_generator)

    tri_out = geometric(
        torch.empty_like(x),
        p=0.5,
        generator=out_generator,
    )

    assert tri_out.shape == x.shape
    assert tri_out.dtype == x.dtype
    torch.testing.assert_close(tri_out.mean(), ref_out.mean(), rtol=0.15, atol=0.2)
    torch.testing.assert_close(
        tri_out.var(unbiased=False),
        ref_out.var(unbiased=False),
        rtol=0.2,
        atol=0.2,
    )


@pytest.mark.parametrize("shape", [(1024,), (4, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometric_properties(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")

    tri_out = geometric(x, p=0.5)
    ret = geometric_(x, p=0.5)

    assert ret is x
    assert tri_out.shape == x.shape
    assert torch.all(tri_out >= 1)
    assert torch.all(x >= 1)


@pytest.mark.parametrize("shape", [(256,), (16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometric_generator_reproducible(shape, dtype):
    torch.manual_seed(0)
    x = torch.empty(shape, dtype=dtype, device="cpu")
    g_a = torch.Generator(device="cpu").manual_seed(777)
    g_b = torch.Generator(device="cpu").manual_seed(777)

    tri_out_a = geometric(torch.empty_like(x), p=0.4, generator=g_a)
    tri_out_b = geometric(torch.empty_like(x), p=0.4, generator=g_b)

    torch.testing.assert_close(tri_out_a, tri_out_b)
    assert torch.all(tri_out_a >= 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometric_inplace_non_contiguous(dtype):
    x = torch.empty((513, 128), dtype=dtype, device="cpu")[::2]
    generator = torch.Generator(device="cpu").manual_seed(456)
    contiguous_generator = torch.Generator(device="cpu").manual_seed(456)
    torch_generator = torch.Generator(device="cpu").manual_seed(789)
    contiguous = torch.empty(x.shape, dtype=dtype, device="cpu")
    torch_ref = torch.empty((513, 128), dtype=dtype, device="cpu")[::2]

    geometric_(x, p=0.4, generator=generator)
    geometric_(contiguous, p=0.4, generator=contiguous_generator)
    torch_ref.geometric_(p=0.4, generator=torch_generator)

    assert not x.is_contiguous()
    torch.testing.assert_close(x, contiguous)
    torch.testing.assert_close(x.mean(), torch_ref.mean(), rtol=0.1, atol=0.1)
    torch.testing.assert_close(
        x.var(unbiased=False), torch_ref.var(unbiased=False), rtol=0.15, atol=0.15
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometric_out_of_place_non_contiguous_input(dtype):
    x = torch.empty((5, 7), dtype=dtype, device="cpu").transpose(0, 1)
    out_generator = torch.Generator(device="cpu").manual_seed(2468)
    contiguous_generator = torch.Generator(device="cpu").manual_seed(2468)
    torch_generator = torch.Generator(device="cpu").manual_seed(1357)

    out = geometric(x, p=0.4, generator=out_generator)
    contiguous = geometric(
        torch.empty(x.shape, dtype=dtype, device="cpu"),
        p=0.4,
        generator=contiguous_generator,
    )
    torch_ref = torch.empty_like(x).geometric_(p=0.4, generator=torch_generator)

    assert not out.is_contiguous()
    torch.testing.assert_close(out, contiguous)
    torch.testing.assert_close(out.mean(), torch_ref.mean(), rtol=0.15, atol=0.15)

import pytest
import torch

from .cosine_similarity import cosine_similarity

# torch reference: torch.nn.functional.cosine_similarity computes
# <x1, x2> / (max(||x1||_2, eps) * max(||x2||_2, eps)) over `dim`.
torch_cos = torch.nn.functional.cosine_similarity


@pytest.mark.parametrize("rows", [512, 1023, 1024])
def test_cosine_similarity(rows):
    torch.manual_seed(0)
    x1 = torch.randn(rows, 64, device="cpu", dtype=torch.float32)
    x2 = torch.randn(rows, 64, device="cpu", dtype=torch.float32)

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-8)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((7,), 0),  # 1-D: a single vector pair -> scalar (0-dim) output
        ((64, 64), 0),
        ((1024, 257), 1),  # D not a multiple of the tile width
        ((16, 32, 64), -1),  # negative dim
        ((16, 32, 64), 1),  # middle dim: exercises the movedim path
        ((8, 4, 8, 16), 2),  # 4-D
    ],
)
def test_cosine_similarity_multi_dim(shape, dim):
    torch.manual_seed(0)
    x1 = torch.randn(shape, device="cpu", dtype=torch.float32)
    x2 = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch_cos(x1, x2, dim=dim, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=dim, eps=1e-8)

    assert tri_out.shape == ref_out.shape
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_cosine_similarity_default_args():
    torch.manual_seed(0)
    x1 = torch.randn(8, 16, device="cpu", dtype=torch.float32)
    x2 = torch.randn(8, 16, device="cpu", dtype=torch.float32)

    torch.testing.assert_close(
        cosine_similarity(x1, x2),
        torch_cos(x1, x2, dim=1, eps=1e-8),
        rtol=1e-4,
        atol=1e-4,
    )


def test_cosine_similarity_directions():
    # Same direction -> ~1, opposite -> ~-1, orthogonal -> ~0; the
    # [-1, 1] clamp keeps any rounding overshoot inside the range.
    torch.manual_seed(0)
    base = torch.randn(8, dtype=torch.float32)
    orth = torch.randn(8, dtype=torch.float32)
    orth = orth - (orth @ base) / (base @ base) * base  # Gram-Schmidt

    x1 = base.unsqueeze(0).expand(3, 8)
    x2 = torch.stack([base, -base, orth])

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-8)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    assert tri_out[0].item() == pytest.approx(1.0, abs=1e-5)
    assert tri_out[1].item() == pytest.approx(-1.0, abs=1e-5)
    assert tri_out[2].item() == pytest.approx(0.0, abs=1e-6)
    assert (tri_out <= 1.0).all() and (tri_out >= -1.0).all()


def test_cosine_similarity_zero_vector_eps():
    # eps clamps each L2 norm before the division: a zero row gives 0
    # (0 / (eps * ||x2||)), never NaN.  Row 0 is zero on both sides.
    torch.manual_seed(0)
    x1 = torch.zeros(4, 8, dtype=torch.float32)
    x2 = torch.randn(4, 8, dtype=torch.float32)
    x2[0] = 0.0

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-8)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    assert torch.isfinite(tri_out).all()
    assert (tri_out == 0.0).all()


def test_cosine_similarity_eps_clamps_tiny_vectors():
    # ||v|| = 1e-5 * sqrt(8) < eps = 1e-3: both denominators clamp to
    # eps, so cos = 8e-10 / 1e-6 = 8e-4 (clamping the norms shrinks the
    # similarity instead of saturating it at 1).
    v = torch.full((8,), 1e-5, dtype=torch.float32)
    x1 = v.unsqueeze(0)
    x2 = v.unsqueeze(0)

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-3)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-3)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    assert tri_out.item() == pytest.approx(8e-4, rel=1e-3)


@pytest.mark.parametrize(
    "x1_shape, x2_shape, dim",
    [
        ((3, 4), (1, 4), 1),  # row-broadcast
        ((3, 1, 5), (1, 4, 5), 2),  # 3-D two-sided broadcast -> out (3, 4)
        ((2, 3, 4, 5), (5,), 3),  # 4-D vs trailing 1-D -> out (2, 3, 4)
        ((3, 4), (4,), 1),  # 2-D vs trailing 1-D
    ],
)
def test_cosine_similarity_broadcast(x1_shape, x2_shape, dim):
    torch.manual_seed(0)
    x1 = torch.randn(x1_shape, dtype=torch.float32)
    x2 = torch.randn(x2_shape, dtype=torch.float32)

    ref_out = torch_cos(x1, x2, dim=dim, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=dim, eps=1e-8)

    assert tri_out.shape == ref_out.shape
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape, dim", [((0, 8), 1), ((0, 3, 5), 2)])
def test_cosine_similarity_empty(shape, dim):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=torch.float32)
    x2 = torch.randn(shape, dtype=torch.float32)

    ref_out = torch_cos(x1, x2, dim=dim, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=dim, eps=1e-8)

    assert tri_out.shape == ref_out.shape
    assert tri_out.numel() == 0


def test_cosine_similarity_non_contiguous():
    torch.manual_seed(0)
    x1 = torch.randn(64, 32, dtype=torch.float32).t()
    x2 = torch.randn(64, 32, dtype=torch.float32).t()
    assert not x1.is_contiguous()
    assert not x2.is_contiguous()

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-8)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        # fp16: loads narrow to fp32 for the reduction, stores back once.
        # fp64: accumulated in fp64 (a fp32 accumulator would drift by
        # ~D * eps32 ~ 3e-5 over D=257 and miss the 1e-6 bar).
        (torch.float16, 1e-3, 1e-3),
        (torch.float64, 1e-6, 1e-6),
    ],
)
def test_cosine_similarity_other_float(dtype, rtol, atol):
    torch.manual_seed(0)
    x1 = torch.randn(64, 257, dtype=torch.float32).to(dtype)
    x2 = torch.randn(64, 257, dtype=torch.float32).to(dtype)

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-8)

    torch.testing.assert_close(tri_out, ref_out, rtol=rtol, atol=atol)


def test_cosine_similarity_bf16():
    torch.manual_seed(0)
    x1 = torch.randn(64, 64).to(torch.bfloat16)
    x2 = torch.randn(64, 64).to(torch.bfloat16)

    ref_out = torch_cos(x1, x2, dim=1, eps=1e-8)
    tri_out = cosine_similarity(x1, x2, dim=1, eps=1e-8)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def test_cosine_similarity_triton_path(monkeypatch):
    # Smoke: the fp32 path must launch the Triton kernel exactly once.
    import importlib

    mod = importlib.import_module("flaggems.cosine_similarity")

    launches = []
    real_run = mod.cosine_similarity_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.cosine_similarity_kernel, "run", spy_run)
    x1 = torch.randn(64, 64, dtype=torch.float32)
    x2 = torch.randn(64, 64, dtype=torch.float32)
    out = mod.cosine_similarity(x1, x2, dim=1, eps=1e-8)
    assert len(launches) == 1, "cosine_similarity must go through the Triton kernel"
    torch.testing.assert_close(
        out, torch_cos(x1, x2, dim=1, eps=1e-8), rtol=1e-4, atol=1e-4
    )

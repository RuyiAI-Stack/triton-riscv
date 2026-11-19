import importlib

import pytest
import torch

from .lift_fresh import lift_fresh

# aten::lift_fresh(Tensor self) -> Tensor: hand a fresh tensor to autograd.
# On CPU aten returns a same-storage alias and attaches LiftFreshBackward when
# the input requires grad. The FlagGems port materializes an independent
# detached copy (torch.empty_like + on-device copy), so the autograd contract
# asserted here is the port's: fresh storage, grad_fn=None, is_leaf=True.
FLOAT_DTYPES = [torch.float32, torch.float64]
EMPTY_SHAPES = [(0,), (3, 0), (2, 0, 4)]


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,), (2, 3)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_lift_fresh(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    ref = torch.ops.aten.lift_fresh(x)
    tri = lift_fresh(x)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.dtype == dtype
    assert tri.shape == x.shape
    assert tri.data_ptr() != x.data_ptr()


def test_lift_fresh_int_bool():
    torch.manual_seed(0)
    for x in (
        torch.randint(-100, 100, (512,), dtype=torch.int32),
        torch.randint(-100, 100, (512,), dtype=torch.int64),
        torch.rand(512) < 0.5,
    ):
        tri = lift_fresh(x)
        torch.testing.assert_close(tri, torch.ops.aten.lift_fresh(x), rtol=0, atol=0)
        assert tri.dtype == x.dtype
        assert tri.data_ptr() != x.data_ptr()


def test_lift_fresh_complex_host_path():
    # Complex tensors take the documented host clone() path (Triton cannot
    # canonicalize complex pointer types on some backends).
    torch.manual_seed(0)
    for dtype in (torch.complex64, torch.complex128):
        x = torch.randn(256, dtype=dtype)
        tri = lift_fresh(x)
        torch.testing.assert_close(tri, x, rtol=0, atol=0)
        assert tri.dtype == dtype
        assert tri.data_ptr() != x.data_ptr()


def test_lift_fresh_autograd_contract():
    torch.manual_seed(0)
    # requires_grad=True input: aten attaches LiftFreshBackward (non-leaf),
    # the port returns the fresh detached copy.
    x = torch.randn(8, requires_grad=True)
    ref = torch.ops.aten.lift_fresh(x)
    assert ref.requires_grad
    assert not ref.is_leaf
    assert ref.grad_fn is not None
    tri = lift_fresh(x)
    torch.testing.assert_close(tri, x.detach(), rtol=0, atol=0)
    assert not tri.requires_grad
    assert tri.is_leaf
    assert tri.grad_fn is None

    # requires_grad=False input: both aten and the port return a plain leaf.
    y = torch.randn(8)
    ref2 = torch.ops.aten.lift_fresh(y)
    assert not ref2.requires_grad
    assert ref2.is_leaf
    assert ref2.grad_fn is None
    tri2 = lift_fresh(y)
    torch.testing.assert_close(tri2, y, rtol=0, atol=0)
    assert not tri2.requires_grad
    assert tri2.is_leaf
    assert tri2.grad_fn is None


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_lift_fresh_empty(shape):
    # numel()==0 must not launch the copy kernel (a zero grid aborts some
    # drivers); the fresh empty tensor is already the result.
    x = torch.empty(shape, dtype=torch.float32)
    tri = lift_fresh(x)
    assert tri.shape == x.shape
    assert tri.numel() == 0
    torch.testing.assert_close(tri, torch.ops.aten.lift_fresh(x), rtol=0, atol=0)


def test_lift_fresh_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(8, 6).t()
    assert not x.is_contiguous()
    tri = lift_fresh(x)
    torch.testing.assert_close(tri, torch.ops.aten.lift_fresh(x), rtol=0, atol=0)
    assert tri.shape == x.shape
    # The port materializes contiguous storage for the logical values.
    assert tri.is_contiguous()
    assert tri.data_ptr() != x.data_ptr()


def test_lift_fresh_0d():
    torch.manual_seed(0)
    x = torch.randn(())
    tri = lift_fresh(x)
    torch.testing.assert_close(tri, x, rtol=0, atol=0)
    assert tri.shape == ()


def test_lift_fresh_kwarg_self():
    torch.manual_seed(0)
    x = torch.randn(16)
    tri = lift_fresh(self=x)
    torch.testing.assert_close(tri, x, rtol=0, atol=0)
    assert tri.data_ptr() != x.data_ptr()


def test_lift_fresh_requires_tensor():
    with pytest.raises(ValueError):
        lift_fresh(3)


def test_lift_fresh_triton_path(monkeypatch):
    # Smoke: the float main path must launch the Triton copy kernel.
    mod = importlib.import_module("flaggems.lift_fresh")

    launches = []
    real_run = mod._copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._copy_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(1024)
    out = mod.lift_fresh(x)
    assert len(launches) == 1, "lift_fresh(x) must go through the Triton kernel"
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_lift_fresh_empty_no_launch(monkeypatch):
    # Smoke: numel()==0 must skip the kernel launch entirely.
    mod = importlib.import_module("flaggems.lift_fresh")

    launches = []
    real_run = mod._copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._copy_kernel, "run", spy_run)
    x = torch.empty(0)
    out = mod.lift_fresh(x)
    assert len(launches) == 0, "lift_fresh(empty) must not launch the kernel"
    assert out.numel() == 0

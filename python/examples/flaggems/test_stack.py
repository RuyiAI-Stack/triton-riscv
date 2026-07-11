import torch

from .stack import stack


def test_stack_two_tensors():
    torch.manual_seed(0)
    a = torch.randn(16, dtype=torch.float32, device="cpu")
    b = torch.randn(16, dtype=torch.float32, device="cpu")
    ref = torch.stack([a, b])
    tri = stack([a, b])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_stack_four_tensors():
    torch.manual_seed(0)
    tensors = [torch.randn(512, dtype=torch.float32, device="cpu") for _ in range(4)]
    ref = torch.stack(tensors)
    tri = stack(tensors)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_stack_three_tensors():
    torch.manual_seed(0)
    tensors = [torch.randn(128, dtype=torch.float32, device="cpu") for _ in range(3)]
    ref = torch.stack(tensors)
    tri = stack(tensors)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_stack_dim1():
    torch.manual_seed(0)
    a = torch.randn(4, 128, dtype=torch.float32, device="cpu")
    b = torch.randn(4, 128, dtype=torch.float32, device="cpu")
    ref = torch.stack([a, b], dim=1)
    tri = stack([a, b], dim=1)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_stack_neg_dim():
    torch.manual_seed(0)
    a = torch.randn(4, 128, dtype=torch.float32, device="cpu")
    b = torch.randn(4, 128, dtype=torch.float32, device="cpu")
    ref = torch.stack([a, b], dim=-1)
    tri = stack([a, b], dim=-1)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_stack_single_tensor_has_independent_storage():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    tri = stack([x], dim=0)

    torch.testing.assert_close(tri, torch.stack([x], dim=0))
    assert tri.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    tri[0, 0, 0] = -1
    assert x[0, 0].item() == 0

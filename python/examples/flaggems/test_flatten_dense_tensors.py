import importlib

import pytest
import torch

from .flatten_dense_tensors import flatten_dense_tensors


def _spy_on(kernel, launches):
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    return spy_run


def test_flatten_dense_tensors_contiguous_list_matches_aten():
    ts = [
        torch.arange(6, dtype=torch.float64).reshape(2, 3),
        torch.arange(4, dtype=torch.float64),
        torch.arange(8, dtype=torch.float64).reshape(2, 2, 2),
    ]
    out = flatten_dense_tensors(ts)
    ref = torch.ops.aten.flatten_dense_tensors(ts)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == ref.shape


def test_flatten_dense_tensors_dtype_promotion():
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(2, dtype=torch.float64)
    out = flatten_dense_tensors([a, b])
    ref = torch.ops.aten.flatten_dense_tensors([a, b])
    assert out.dtype == ref.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    c = torch.arange(2, dtype=torch.int64)
    out2 = flatten_dense_tensors([a, c])
    ref2 = torch.ops.aten.flatten_dense_tensors([a, c])
    assert out2.dtype == ref2.dtype == torch.float32
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)


def test_flatten_dense_tensors_strided_inputs():
    t = torch.arange(6, dtype=torch.float64).reshape(2, 3).t()
    assert not t.is_contiguous()
    out = flatten_dense_tensors([t])
    ref = torch.ops.aten.flatten_dense_tensors([t])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.tolist() == [0, 3, 1, 4, 2, 5]
    col = torch.arange(3, dtype=torch.float64).reshape(1, 3).t()
    out2 = flatten_dense_tensors([col])
    assert out2.tolist() == [0, 1, 2]
    t4 = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
    strided4 = t4.permute(2, 0, 3, 1)
    out3 = flatten_dense_tensors([strided4])
    ref3 = torch.ops.aten.flatten_dense_tensors([strided4])
    torch.testing.assert_close(out3, ref3, rtol=0, atol=0)


def test_flatten_dense_tensors_single_contiguous_is_view():
    x = torch.arange(6).reshape(2, 3)
    out = flatten_dense_tensors([x])
    ref = torch.ops.aten.flatten_dense_tensors([x])
    assert out.data_ptr() == x.data_ptr()  # aten also returns a view
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_flatten_dense_tensors_empty_and_0d_members():
    ts = [torch.empty(0, dtype=torch.float64), torch.arange(3, dtype=torch.float64)]
    out = flatten_dense_tensors(ts)
    ref = torch.ops.aten.flatten_dense_tensors(ts)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out0 = flatten_dense_tensors([torch.tensor(7.0)])
    ref0 = torch.ops.aten.flatten_dense_tensors([torch.tensor(7.0)])
    assert out0.shape == (1,)
    torch.testing.assert_close(out0, ref0, rtol=0, atol=0)


def test_flatten_dense_tensors_bool_and_complex():
    ts = [torch.tensor([True, False]), torch.tensor([[True]])]
    out = flatten_dense_tensors(ts)
    ref = torch.ops.aten.flatten_dense_tensors(ts)
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    cs = [torch.complex(torch.arange(3.0), torch.ones(3)), torch.tensor(2.0 + 1j)]
    outc = flatten_dense_tensors(cs)
    refc = torch.ops.aten.flatten_dense_tensors(cs)
    assert outc.dtype == torch.complex64
    torch.testing.assert_close(outc, refc, rtol=0, atol=0)


def test_flatten_dense_tensors_errors_match_aten():
    with pytest.raises(ValueError, match="expected a non-empty list of Tensors"):
        flatten_dense_tensors([])
    with pytest.raises(
        RuntimeError, match="Expected a value of type 'List\\[Tensor\\]'"
    ):
        flatten_dense_tensors(torch.arange(3))
    with pytest.raises(RuntimeError, match="but instead found type 'list'"):
        flatten_dense_tensors([torch.arange(3), 5])


def test_flatten_dense_tensors_triton_path_flat_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.flatten_dense_tensors")
    launches = []
    kernel = mod._flatten_copy_kernel
    monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
    ts = [torch.arange(4.0), torch.arange(6.0).reshape(2, 3)]
    out = flatten_dense_tensors(ts)
    assert len(launches) == 2  # one launch per tensor
    torch.testing.assert_close(
        out, torch.ops.aten.flatten_dense_tensors(ts), rtol=0, atol=0
    )


def test_flatten_dense_tensors_triton_path_strided_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.flatten_dense_tensors")
    launches = []
    kernel = mod._flatten_strided_kernel
    monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
    t = torch.arange(6, dtype=torch.float64).reshape(2, 3).t()
    out = flatten_dense_tensors([t, torch.arange(2, dtype=torch.float64)])
    assert len(launches) >= 1
    ref = torch.ops.aten.flatten_dense_tensors(
        [t, torch.arange(2, dtype=torch.float64)]
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

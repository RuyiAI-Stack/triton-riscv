import sys

import pytest
import torch

from ._list_to_tensor import _list_to_tensor

# Semantics (upstream flag_gems/ops/_list_to_tensor.py + aten, probed on this
# torch 2.15 CPU build): aten::_list_to_tensor converts List[int] into a 1-D
# int32 tensor. The upstream FlagGems file pins dtype=torch.int32 and builds
# the tensor with a Triton copy kernel (host torch.tensor -> device buffer).
# Despite the upstream docstring's "no c10 dispatcher kernel" note, this
# torch build exposes torch.ops.aten._list_to_tensor and it also returns
# int32 (probed), so it serves as a direct bit-exact reference.
# Edge behavior inherited from the host torch.tensor conversion: empty list
# -> empty int32 tensor (no kernel launch); values outside int32 raise
# RuntimeError "... overflow"; float lists truncate (the aten dispatcher
# schema-rejects them instead -- divergence pinned explicitly below).

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggems package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_L2T_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._list_to_tensor"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens._list_to_tensor"
    if name not in sys.modules:
        sys.modules[name] = _L2T_MOD
    return importlib.import_module(name)


@pytest.mark.parametrize(
    "self_list",
    [
        [1, 2, 3, 4],
        [5],
        [0, -1, 7, 100, -42],
        [2**31 - 1, -(2**31), 0],  # int32 boundary values, bit-exact
        list(range(2000)),  # spans two BLOCK_SIZE=1024 tiles
    ],
)
def test__list_to_tensor_roundtrip(self_list):
    ref = torch.ops.aten._list_to_tensor(self_list)

    out = _list_to_tensor(self_list)

    assert out.dtype == torch.int32
    assert out.shape == (len(self_list),)
    assert out.dim() == 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test__list_to_tensor_empty():
    ref = torch.ops.aten._list_to_tensor([])

    out = _list_to_tensor([])

    assert out.dtype == torch.int32
    assert out.shape == (0,)
    assert out.numel() == 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test__list_to_tensor_int32_boundary_bit_exact():
    vals = [2**31 - 1, -(2**31), 0]
    out = _list_to_tensor(vals)

    assert out.dtype == torch.int32
    assert out[0].item() == 2**31 - 1
    assert out[1].item() == -(2**31)
    assert out[2].item() == 0
    torch.testing.assert_close(
        out, torch.tensor(vals, dtype=torch.int32), rtol=0, atol=0
    )


def test__list_to_tensor_overflow_raises():
    with pytest.raises(RuntimeError, match="overflow"):
        _list_to_tensor([2**31])  # one past int32 max


def test__list_to_tensor_float_list_truncates():
    # Upstream-file behavior: the host-side torch.tensor(self, dtype=int32)
    # truncates float lists toward zero. The aten dispatcher instead
    # schema-rejects float lists ("Expected a value of type 'List[int]'");
    # this port pins the upstream file's conversion behavior.
    out = _list_to_tensor([1, -9.5, 2.9])

    torch.testing.assert_close(
        out, torch.tensor([1, -9, 2], dtype=torch.int32), rtol=0, atol=0
    )


def test__list_to_tensor_triton_path(monkeypatch):
    # Smoke: a non-empty list must launch the Triton copy kernel.
    mod = _spy_module()

    launches = []
    real_run = mod.copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.copy_kernel, "run", spy_run)
    self_list = list(range(2000))
    out = mod._list_to_tensor(self_list)
    assert len(launches) == 1, "non-empty list must go through the Triton kernel"
    assert launches[0] == (2,), "2000 elements with BLOCK_SIZE=1024 -> grid (2,)"
    torch.testing.assert_close(
        out, torch.tensor(self_list, dtype=torch.int32), rtol=0, atol=0
    )


def test__list_to_tensor_empty_no_kernel(monkeypatch):
    # The empty-list fast path must not launch the kernel.
    mod = _spy_module()

    launches = []
    real_run = mod.copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.copy_kernel, "run", spy_run)
    out = mod._list_to_tensor([])
    assert launches == [], "empty list must not launch the Triton kernel"
    assert out.numel() == 0
    assert out.dtype == torch.int32

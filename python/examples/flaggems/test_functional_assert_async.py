import sys

import pytest
import torch

from .functional_assert_async import _functional_assert_async

# Semantics (upstream flag_gems/ops/functional_assert_async.py + aten, probed
# on this torch 2.15 CPU build): a numel==1 non-zero tensor passes -- a no-op
# that returns a NEW empty dependency token (numel 0) carrying dep_token's
# dtype and device. A zero value raises RuntimeError(assert_msg) (aten's CPU
# kernel does exactly this; NaN and bool True pass). numel != 1 raises
# RuntimeError "Boolean value of Tensor with shape [...] is ambiguous"
# (FlagGems wording; aten says "of Tensor with more than one value").
# tl.device_assert never fires on this CPU backend (probed against the
# existing assert_async port), so the port enforces the value on the host
# before launching the (faithfully kept) Triton kernel.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggens package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_FAA_MOD = sys.modules[__name__.rsplit(".", 1)[0] + ".functional_assert_async"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens.functional_assert_async"
    if name not in sys.modules:
        sys.modules[name] = _FAA_MOD
    return importlib.import_module(name)


def test_pass_returns_new_empty_token():
    tensor = torch.tensor([1], dtype=torch.int32)
    dep_token = torch.empty(0, dtype=torch.int32)

    out = _functional_assert_async(tensor, "assertion failed", dep_token)

    assert out.numel() == 0
    assert out.dtype == dep_token.dtype
    assert out.device == dep_token.device
    assert out is not dep_token


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([1.5]),
        torch.tensor([float("nan")]),  # NaN != 0 -> passes (aten parity)
        torch.tensor([-1]),
        torch.tensor([[1]]),  # numel 1 in a 2-d shape (aten also passes this)
        torch.tensor(7),  # 0-d, numel 1
    ],
)
def test_pass_matrix(tensor):
    dep_token = torch.empty(0, dtype=torch.float32)

    out = _functional_assert_async(tensor, "m", dep_token)

    assert out.numel() == 0
    assert out.dtype == torch.float32
    # aten parity: the CPU reference returns the same empty token
    ref = torch.ops.aten._functional_assert_async(tensor, "m", dep_token)
    assert out.dtype == ref.dtype and out.shape == ref.shape


@pytest.mark.parametrize("token_dtype", [torch.float32, torch.int64, torch.bfloat16])
def test_token_dtype_passthrough(token_dtype):
    tensor = torch.tensor([1], dtype=torch.int32)
    dep_token = torch.empty(0, dtype=token_dtype)

    out = _functional_assert_async(tensor, "m", dep_token)

    assert out.dtype == token_dtype
    assert out.numel() == 0


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([0], dtype=torch.int32),
        torch.tensor(0),  # 0-d zero
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor([-0.0]),
    ],
)
def test_violation_raises(tensor):
    dep_token = torch.empty(0, dtype=torch.int32)

    with pytest.raises(RuntimeError) as excinfo:
        _functional_assert_async(tensor, "EXACT-MSG", dep_token)

    assert str(excinfo.value) == "EXACT-MSG", "message must be exactly assert_msg"


def test_violation_aten_parity():
    # aten's CPU kernel raises RuntimeError(assert_msg) on a zero value; the
    # port must raise the same error class with the same message.
    for tensor in [torch.tensor([0]), torch.tensor([0.0]), torch.tensor([False])]:
        dep_token = torch.empty(0, dtype=torch.int32)
        with pytest.raises(RuntimeError, match="^aten-msg$"):
            torch.ops.aten._functional_assert_async(tensor, "aten-msg", dep_token)
        with pytest.raises(RuntimeError, match="^aten-msg$"):
            _functional_assert_async(tensor, "aten-msg", dep_token)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([], dtype=torch.int64),
    ],
)
def test_numel_not_one_raises(tensor):
    dep_token = torch.empty(0, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="ambiguous"):
        _functional_assert_async(tensor, "m", dep_token)


def test_numel_not_one_message_contains_shape():
    tensor = torch.tensor([1, 1, 1], dtype=torch.int32)
    dep_token = torch.empty(0, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="ambiguous") as excinfo:
        _functional_assert_async(tensor, "m", dep_token)

    assert str(list(tensor.shape)) in str(excinfo.value)


def test_pass_triton_path(monkeypatch):
    # Smoke: the pass path must launch the Triton kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._functional_assert_async_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._functional_assert_async_kernel, "run", spy_run)
    tensor = torch.tensor([1], dtype=torch.int32)
    dep_token = torch.empty(0, dtype=torch.int32)
    out = mod._functional_assert_async(tensor, "m", dep_token)
    assert len(launches) == 1, "the pass path must launch the Triton kernel"
    assert launches[0] == (1,)
    assert out.numel() == 0


def test_violation_no_kernel_launch(monkeypatch):
    # The host-side violation check runs before the kernel launch.
    mod = _spy_module()

    launches = []
    real_run = mod._functional_assert_async_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._functional_assert_async_kernel, "run", spy_run)
    tensor = torch.tensor([0], dtype=torch.int32)
    dep_token = torch.empty(0, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="^boom$"):
        mod._functional_assert_async(tensor, "boom", dep_token)
    assert launches == [], "the violation path must not launch the kernel"

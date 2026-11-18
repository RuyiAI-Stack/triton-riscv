import importlib

import pytest
import torch

from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)

# Bit-exactness (probed against torch 2.15 CPU): aten widens fp16/bf16
# grads to fp32 opmath and fp64 grads to double, multiplies by the fp32
# inv_scale (widened exactly), and rounds back to the storage dtype --
# a single IEEE multiply, so every test below compares with rtol=0,
# atol=0.  found_inf is only ever *set* (fill 1.0), never cleared, and
# the finiteness check sees the *pre-scale* values (an fp16 overflow
# produced by the scaling itself does not set found_inf).

AMP_GROWTH = 2.0
AMP_BACKOFF = 0.5
AMP_GROWTH_INTERVAL = 2000


@pytest.fixture
def amp_scale_state():
    """Fresh (scale, growth_tracker, found_inf) triple in the layout the
    two AMP ops exchange; reused by the _amp_update_scale_ suite."""
    scale = torch.tensor(65536.0)
    growth_tracker = torch.zeros((), dtype=torch.int32)
    found_inf = torch.zeros(())
    return scale, growth_tracker, found_inf


def torch_unscale_ref(grads, found_inf, inv_scale):
    """Direct aten call on cloned inputs; returns the reference grads."""
    grad_clones = [g.clone() for g in grads]
    torch._amp_foreach_non_finite_check_and_unscale_(
        grad_clones, found_inf.clone(), inv_scale
    )
    return grad_clones


def torch_update_scale_ref(scale, growth_tracker, found_inf, growth, backoff, interval):
    """Direct aten call on cloned state; returns the reference pair."""
    s, t = scale.clone(), growth_tracker.clone()
    torch._amp_update_scale_(s, t, found_inf.clone(), growth, backoff, interval)
    return s, t


@pytest.mark.parametrize("size", [512, 1023, 1024, 3000])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
def test_unscale_basic(dtype, size):
    torch.manual_seed(0)
    g1 = torch.randn(size, dtype=dtype) * 8.0
    g2 = torch.randn(size, dtype=dtype) * 0.125
    found_inf = torch.zeros(())
    inv_scale = torch.tensor(2.0)

    ref1, ref2 = torch_unscale_ref([g1, g2], found_inf, inv_scale)

    ret = _amp_foreach_non_finite_check_and_unscale_([g1, g2], found_inf, inv_scale)

    assert ret is None
    assert found_inf.item() == 0.0
    torch.testing.assert_close(g1, ref1, rtol=0, atol=0)
    torch.testing.assert_close(g2, ref2, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_unscale_inf_nan_mixing(dtype):
    vals = [1.0, float("inf"), float("-inf"), float("nan"), -3.0, 0.0]
    g = torch.tensor(vals, dtype=dtype)
    found_inf = torch.zeros(())
    inv_scale = torch.tensor(0.5)

    ref = torch_unscale_ref([g], found_inf, inv_scale)[0]

    _amp_foreach_non_finite_check_and_unscale_([g], found_inf, inv_scale)

    assert found_inf.item() == 1.0
    # All values, non-finite included, are multiplied (aten semantics);
    # inf*0.5 stays inf and nan*0.5 stays nan, positionally.
    torch.testing.assert_close(g, ref, rtol=0, atol=0, equal_nan=True)
    assert g[1].item() == float("inf") and g[2].item() == float("-inf")
    assert torch.isnan(g[3])


def test_unscale_found_inf_accumulates():
    # found_inf is never cleared: pre-set 1.0 survives an all-finite run.
    g = torch.tensor([1.0, 2.0])
    found_inf = torch.ones(())
    _amp_foreach_non_finite_check_and_unscale_([g], found_inf, torch.tensor(2.0))
    assert found_inf.item() == 1.0
    assert g.tolist() == [2.0, 4.0]

    # fill, not add: a pre-set 2.0 with a non-finite grad becomes 1.0.
    g = torch.tensor([float("nan")])
    found_inf = torch.tensor(2.0)
    _amp_foreach_non_finite_check_and_unscale_([g], found_inf, torch.tensor(2.0))
    assert found_inf.item() == 1.0


def test_unscale_fp16_overflow_pin():
    # 60000 is finite in fp16 but 60000 * 2 overflows: aten checks
    # finiteness BEFORE scaling, so found_inf stays 0 and the grad
    # saturates to inf.  (Upstream FlagGems re-checks after scaling and
    # would set found_inf = 1 here -- a divergence pinned to aten.)
    g = torch.tensor([60000.0, 1.0], dtype=torch.float16)
    found_inf = torch.zeros(())

    ref = torch_unscale_ref([g], found_inf, torch.tensor(2.0))[0]

    _amp_foreach_non_finite_check_and_unscale_([g], found_inf, torch.tensor(2.0))

    assert found_inf.item() == 0.0
    torch.testing.assert_close(g, ref, rtol=0, atol=0)
    assert g[0].item() == float("inf")


@pytest.mark.parametrize(
    "inv", [0.0, -1.0, 1e30], ids=["zero", "negative", "huge"]
)
def test_unscale_inv_scale_boundary(inv):
    # Degenerate scales: every value (finite or not) is multiplied, so
    # inf*0 -> nan and the huge scale saturates fp16 to inf.  found_inf
    # is decided by the pre-scale values only.
    g = torch.tensor([2.0, float("inf"), -30000.0], dtype=torch.float16)
    found_inf = torch.zeros(())
    inv_scale = torch.tensor(inv)

    ref = torch_unscale_ref([g], found_inf, inv_scale)[0]

    _amp_foreach_non_finite_check_and_unscale_([g], found_inf, inv_scale)

    assert found_inf.item() == 1.0  # the pre-existing inf
    torch.testing.assert_close(g, ref, rtol=0, atol=0, equal_nan=True)


def test_unscale_fp64_precision_pin():
    # 3 * 2^-1074 is a subnormal double; fp32 opmath would flush it to
    # zero.  aten computes fp64 grads in double, so doubling must stay
    # exact -- pins the fp64 kernel path.
    tiny = torch.ldexp(
        torch.tensor(3.0, dtype=torch.float64), torch.tensor(-1074)
    ).item()
    g = torch.tensor([tiny, 1.0], dtype=torch.float64)
    found_inf = torch.zeros(())

    _amp_foreach_non_finite_check_and_unscale_([g], found_inf, torch.tensor(2.0))

    assert g[0].item() == 2 * tiny
    assert g[0].item() != 0.0
    assert found_inf.item() == 0.0


def test_unscale_empty_list():
    found_inf = torch.zeros(())
    _amp_foreach_non_finite_check_and_unscale_([], found_inf, torch.tensor(2.0))
    assert found_inf.item() == 0.0


def test_unscale_empty_tensor_in_list():
    g_empty = torch.empty(0, dtype=torch.float16)
    g_ok = torch.tensor([2.0], dtype=torch.float16)
    found_inf = torch.zeros(())

    _amp_foreach_non_finite_check_and_unscale_(
        [g_empty, g_ok], found_inf, torch.tensor(3.0)
    )

    assert found_inf.item() == 0.0
    assert g_ok.item() == 6.0


def test_unscale_non_contiguous():
    # The scaled values must land through the view into the base storage.
    torch.manual_seed(0)
    base = torch.randn(32, 64, dtype=torch.float32)
    view = base[::2, ::2].t()  # (32, 16) view, non-dense strides
    assert not view.is_contiguous()

    ref_base = base.clone()
    ref_view = ref_base[::2, ::2].t()
    torch._amp_foreach_non_finite_check_and_unscale_(
        [ref_view], torch.zeros(()), torch.tensor(0.5)
    )

    found_inf = torch.zeros(())
    _amp_foreach_non_finite_check_and_unscale_([view], found_inf, torch.tensor(0.5))

    assert found_inf.item() == 0.0
    torch.testing.assert_close(base, ref_base, rtol=0, atol=0)
    torch.testing.assert_close(view, ref_view, rtol=0, atol=0)


def test_unscale_inplace_identity():
    g = torch.randn(1024, dtype=torch.float32)
    data_ptr = g.data_ptr()
    ret = _amp_foreach_non_finite_check_and_unscale_(
        [g], torch.zeros(()), torch.tensor(2.0)
    )
    assert ret is None
    assert g.data_ptr() == data_ptr
    assert g.dtype == torch.float32


def test_unscale_multi_tensor_all_scaled_after_inf():
    # aten has no early return: every tensor is scaled even after one
    # of them set found_inf.
    g_bad = torch.tensor([float("inf")])
    g_good = torch.tensor([1.0, 2.0])
    found_inf = torch.zeros(())

    _amp_foreach_non_finite_check_and_unscale_(
        [g_bad, g_good], found_inf, torch.tensor(4.0)
    )

    assert found_inf.item() == 1.0
    assert g_bad.item() == float("inf")
    assert g_good.tolist() == [4.0, 8.0]


def test_unscale_rejects_non_float_grads():
    # aten raises NotImplementedError for every dtype outside
    # {fp16, bf16, fp32, fp64}; upstream FlagGems silently skips those
    # tensors -- a divergence pinned to aten here.
    for dtype, name in [
        (torch.int64, "Long"),
        (torch.complex64, "ComplexFloat"),
    ]:
        g = torch.zeros(4, dtype=dtype)
        with pytest.raises(NotImplementedError, match=f"not implemented for '{name}'"):
            _amp_foreach_non_finite_check_and_unscale_(
                [g], torch.zeros(()), torch.tensor(2.0)
            )


def test_unscale_dtype_gates():
    g = torch.tensor([1.0])
    # inv_scale is checked before found_inf (probed aten order).
    with pytest.raises(RuntimeError, match="inv_scale must be a float tensor."):
        _amp_foreach_non_finite_check_and_unscale_(
            [g], torch.zeros(()), torch.tensor(2.0, dtype=torch.float16)
        )
    with pytest.raises(RuntimeError, match="inv_scale must be a float tensor."):
        _amp_foreach_non_finite_check_and_unscale_(
            [g], torch.zeros(()), torch.tensor(2.0, dtype=torch.float64)
        )
    with pytest.raises(RuntimeError, match="found_inf must be a float tensor."):
        _amp_foreach_non_finite_check_and_unscale_(
            [g], torch.zeros((), dtype=torch.float64), torch.tensor(2.0)
        )
    # A non-float grad does not shadow a wrong found_inf dtype: the
    # entry gates run first (probed aten order).
    with pytest.raises(RuntimeError, match="found_inf must be a float tensor."):
        _amp_foreach_non_finite_check_and_unscale_(
            [torch.zeros(2, dtype=torch.int64)],
            torch.zeros((), dtype=torch.float64),
            torch.tensor(2.0),
        )


def test_unscale_rejects_non_list():
    with pytest.raises(TypeError, match="Expected list or tuple of tensors"):
        _amp_foreach_non_finite_check_and_unscale_(
            torch.tensor([1.0]), torch.zeros(()), torch.tensor(2.0)
        )


def test_unscale_triton_path(monkeypatch):
    # Smoke: the fp32 main path must launch the Triton kernel.
    mod = importlib.import_module(
        "flaggems._amp_foreach_non_finite_check_and_unscale_"
    )

    launches = []
    real_run = mod._amp_foreach_non_finite_check_and_unscale_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(
        mod._amp_foreach_non_finite_check_and_unscale_kernel, "run", spy_run
    )
    g = torch.randn(1024)
    out_found = torch.zeros(())
    mod._amp_foreach_non_finite_check_and_unscale_(
        [g], out_found, torch.tensor(2.0)
    )
    assert len(launches) == 1, "unscale must go through the Triton kernel"
    assert launches[0] == (1,)

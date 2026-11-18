import importlib

import pytest
import torch

from ._fused_moving_avg_obs_fq_helper import _fused_moving_avg_obs_fq_helper

# torch reference: aten::_fused_moving_avg_obs_fq_helper, CPU-callable on
# this build. Its schema takes bool TENSOR flags, and the per-tensor path
# indexes the state tensors, so per-tensor calls must pass 1-element 1-dim
# state tensors (aten's CPU kernel does running_min.select(0, 0), which
# crashes natively on a 0-dim tensor - probed).
torch_fq = torch.ops.aten._fused_moving_avg_obs_fq_helper
T = torch.tensor


def ref_call(x, observer_on, fake_quant_on, rmin, rmax, scale, zp, avg,
             qmin, qmax, ch_axis, per_row=False, sym=False):
    return torch_fq(
        x, T(observer_on), T(fake_quant_on), rmin, rmax, scale, zp,
        avg, qmin, qmax, ch_axis, per_row, sym,
    )


def fresh_state(channels, fill=float("inf")):
    return (
        torch.full((channels,), fill),
        torch.full((channels,), fill),
        torch.ones(channels),
        torch.zeros(channels),
    )


def make_x(shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32)


def assert_state_close(port, ref):
    rmin_p, rmax_p, sc_p, zp_p = port
    rmin_r, rmax_r, sc_r, zp_r = ref
    torch.testing.assert_close(rmin_p, rmin_r, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rmax_p, rmax_r, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sc_p, sc_r, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(zp_p, zp_r, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(32,), (2, 32), (3, 4, 5), (128, 3)])
@pytest.mark.parametrize("qmin, qmax", [(0, 255), (-128, 127)])
def test_fq_per_tensor_obs_and_fq(shape, qmin, qmax):
    x = make_x(shape)
    rmin, rmax, sc, zp = fresh_state(1)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, qmin, qmax, 0
    )
    out_r, mask_r = ref_call(
        x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, qmin, qmax, 0
    )

    assert out.dtype == torch.float32
    assert mask.dtype == torch.bool
    assert out.shape == x.shape and mask.shape == x.shape
    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)
    assert torch.equal(mask, mask_r)
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))


@pytest.mark.parametrize("shape", [(4, 32), (3, 4, 5), (6, 7, 8)])
@pytest.mark.parametrize("qmin, qmax", [(0, 255), (-128, 127)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_fq_per_row(shape, qmin, qmax, symmetric):
    # per_row_fake_quant with ch_axis == 0: one quantization group per
    # leading-channel, 1-dim state of C elements on both sides.
    x = make_x(shape)
    C = shape[0]
    rmin, rmax, sc, zp = fresh_state(C)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(C)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, qmin, qmax, 0,
        per_row_fake_quant=True, symmetric_quant=symmetric,
    )
    out_r, mask_r = ref_call(
        x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, qmin, qmax, 0,
        True, symmetric,
    )

    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)
    assert torch.equal(mask, mask_r)
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))


def test_fq_symmetric_one_sided_takes_asymmetric_branch():
    # symmetric_quant only switches the branch when the observed range
    # straddles zero; a strictly positive range keeps the affine path.
    torch.manual_seed(5)
    x = torch.rand(2, 32) + 1.0
    rmin, rmax, sc, zp = fresh_state(1)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1)

    _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, -128, 127, 0, False, True
    )
    ref_call(x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, -128, 127, 0, False, True)
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))
    assert zp.item() != 0.0  # affine zero point, not the symmetric 0


@pytest.mark.parametrize("observer_on, fake_quant_on", [(0, 1), (1, 0), (0, 0)])
def test_fq_flag_combinations(observer_on, fake_quant_on):
    x = make_x((2, 32), seed=6)
    rmin, rmax, sc, zp = fresh_state(1, fill=0.5)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1, fill=0.5)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, observer_on, fake_quant_on, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, mask_r = ref_call(
        x, observer_on, fake_quant_on, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0
    )

    if not fake_quant_on:
        assert torch.equal(out, x)
        assert bool(mask.all())
    else:
        torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)
        assert torch.equal(mask, mask_r)
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))
    if not observer_on:
        assert rmin.item() == 0.5  # stats untouched


def test_fq_mask_marks_clamped_values():
    # observer off + a stored range narrower than the data: outliers
    # saturate and their mask entries go False.
    torch.manual_seed(7)
    x = torch.cat([torch.randn(31), torch.tensor([100.0, -100.0])]).reshape(1, 33)
    rmin, rmax, sc, zp = fresh_state(1, fill=0.0)
    rmin.fill_(-1.0)
    rmax.fill_(1.0)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1, fill=0.0)
    rmin_r.fill_(-1.0)
    rmax_r.fill_(1.0)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 0, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, mask_r = ref_call(
        x, 0, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0
    )

    assert not bool(mask.all())
    assert torch.equal(mask, mask_r)
    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)


def test_fq_degenerate_range():
    # rmin == rmax == 0: scale degenerates to 0.1 and the zero point
    # collapses to qmin (affine, qmin >= 0).
    x = torch.zeros(2, 8)
    rmin, rmax, sc, zp = fresh_state(1, fill=0.0)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1, fill=0.0)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 0, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, _ = ref_call(x, 0, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0)

    assert sc.item() == pytest.approx(0.1, abs=1e-6)
    assert zp.item() == 0
    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)


def test_fq_scale_floor():
    # A range narrower than the floor still stores/uses 6.1e-5 while the
    # zero point keeps the unfloored derivation (128 here).
    x = torch.full((2, 8), 1e-6)
    rmin, rmax, sc, zp = fresh_state(1, fill=0.0)
    rmin.fill_(-1e-6)
    rmax.fill_(1e-6)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1, fill=0.0)
    rmin_r.fill_(-1e-6)
    rmax_r.fill_(1e-6)

    _fused_moving_avg_obs_fq_helper(
        x, 0, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    ref_call(x, 0, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0)

    assert sc.item() == pytest.approx(6.1e-5, rel=1e-3)
    assert zp.item() == 128
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))


def test_fq_zero_point_half_tie():
    # rmin/rmax = -0.5/0.5 with qmin/qmax = 0/255 puts qmin - mn/scale on a
    # knife edge; the port's plain '/' is IEEE RN, matching aten's CPU
    # division exactly (aten yields 127 on this build - probed).
    x = torch.zeros(2, 8)
    rmin, rmax, sc, zp = fresh_state(1, fill=0.0)
    rmin.fill_(-0.5)
    rmax.fill_(0.5)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1, fill=0.0)
    rmin_r.fill_(-0.5)
    rmax_r.fill_(0.5)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 0, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, _ = ref_call(x, 0, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0)

    assert zp.item() == 127
    assert sc.item() == pytest.approx(1.0 / 255.0, rel=1e-6)
    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("averaging_const", [0.0, 0.5, 1.0])
def test_fq_averaging_const_extremes(averaging_const):
    # 0.0 freezes the running stats, 1.0 replaces them with the batch
    # min/max, 0.5 blends - the moving-average update is exact in fp32.
    x = make_x((4, 16), seed=8)
    rmin, rmax, sc, zp = fresh_state(4)
    rmin.fill_(0.0)
    rmax.fill_(0.0)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(4)
    rmin_r.fill_(0.0)
    rmax_r.fill_(0.0)

    _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, averaging_const, 0, 255, 0, True
    )
    ref_call(
        x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, averaging_const, 0, 255, 0, True
    )

    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))
    cmin = x.reshape(4, -1).min(dim=1).values
    if averaging_const == 0.0:
        assert torch.equal(rmin, torch.zeros(4))
    elif averaging_const == 1.0:
        torch.testing.assert_close(rmin, cmin, rtol=0, atol=0)


def test_fq_multi_step_state_evolution():
    # Two consecutive observations: the stats move by the moving average
    # each step and both ports stay locked together.
    x1 = make_x((2, 32), seed=9)
    x2 = make_x((2, 32), seed=10)
    rmin, rmax, sc, zp = fresh_state(1)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1)

    for x in (x1, x2):
        out, mask = _fused_moving_avg_obs_fq_helper(
            x, 1, 1, rmin, rmax, sc, zp, 0.1, 0, 255, 0
        )
        out_r, mask_r = ref_call(
            x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, 0.1, 0, 255, 0
        )
        torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)
        assert torch.equal(mask, mask_r)
        assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))

    # Both steps complete without NaN leaking out of the initial inf state;
    # the per-step checks above already pin equality with aten.
    assert torch.isfinite(rmin).all() and torch.isfinite(rmax).all()
    assert torch.isfinite(sc).all() and torch.isfinite(zp).all()


def test_fq_state_updated_in_place():
    # The mutating contract: same storage, new values.
    x = make_x((2, 32), seed=11)
    rmin, rmax, sc, zp = fresh_state(1)

    rmin_ptr = rmin.data_ptr()
    rmax_ptr = rmax.data_ptr()
    sc_ptr = sc.data_ptr()
    zp_ptr = zp.data_ptr()
    _fused_moving_avg_obs_fq_helper(x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0)

    assert rmin.data_ptr() == rmin_ptr
    assert rmax.data_ptr() == rmax_ptr
    assert sc.data_ptr() == sc_ptr
    assert zp.data_ptr() == zp_ptr
    assert rmin.item() != float("inf")


def test_fq_per_tensor_with_0dim_state():
    # 0-dim state tensors are legal for the port (the kernel indexes the
    # data pointer); aten's CPU kernel crashes on them (probed), so this
    # case is port-only.
    x = make_x((2, 32), seed=12)
    rmin = torch.tensor(float("inf"))
    rmax = torch.tensor(float("inf"))
    sc = torch.tensor(1.0)
    zp = torch.tensor(0)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    assert out.shape == x.shape
    assert rmin.item() != float("inf")
    assert sc.item() > 0.0


def test_fq_large_per_tensor_multi_block():
    # numel beyond the single-program bound: partial reduction + finalize +
    # grid-strided elementwise passes.
    x = make_x((4097,), seed=13)
    rmin, rmax, sc, zp = fresh_state(1)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, mask_r = ref_call(
        x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0
    )

    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)
    assert torch.equal(mask, mask_r)
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))


def test_fq_large_per_tensor_copy_pass():
    # Same multi-block path with fake_quant off: the copy pass runs, mask
    # is all ones, stats still update.
    x = make_x((5000,), seed=14)
    rmin, rmax, sc, zp = fresh_state(1)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 0, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, mask_r = ref_call(
        x, 1, 0, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0
    )

    assert torch.equal(out, x)
    assert bool(mask.all())
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))


def test_fq_non_contiguous_input():
    x = make_x((8, 6), seed=15).t()
    assert not x.is_contiguous()
    rmin, rmax, sc, zp = fresh_state(1)
    rmin_r, rmax_r, sc_r, zp_r = fresh_state(1)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    out_r, mask_r = ref_call(
        x, 1, 1, rmin_r, rmax_r, sc_r, zp_r, 0.01, 0, 255, 0
    )

    torch.testing.assert_close(out, out_r, rtol=1e-6, atol=1e-6)
    assert torch.equal(mask, mask_r)
    assert_state_close((rmin, rmax, sc, zp), (rmin_r, rmax_r, sc_r, zp_r))


def test_fq_empty_input():
    # aten errors natively ("aminmax over an empty dimension"); the port
    # returns empty outputs and leaves the state untouched.
    x = torch.zeros(0, 8)
    rmin, rmax, sc, zp = fresh_state(1)

    out, mask = _fused_moving_avg_obs_fq_helper(
        x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )

    assert out.shape == (0, 8)
    assert mask.shape == (0, 8)
    assert rmin.item() == float("inf")
    assert sc.item() == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_fq_rejects_non_float32(dtype):
    # aten: "expected scalar type Float but found Half/Double"; identical
    # guard text in the port.
    x = torch.randn(2, 32, dtype=dtype)
    rmin, rmax, sc, zp = fresh_state(1)

    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        _fused_moving_avg_obs_fq_helper(
            x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
        )
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        ref_call(x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0)


def test_fq_rejects_per_row_off_axis():
    # per-row is only supported on ch_axis == 0 (aten rejects other axes
    # with its own message).
    x = make_x((4, 32), seed=16)
    rmin, rmax, sc, zp = fresh_state(32)

    with pytest.raises(RuntimeError, match="axis == 0"):
        _fused_moving_avg_obs_fq_helper(
            x, 1, 1, rmin, rmax, sc, zp, 0.01, 1, 255, 1, True
        )
    with pytest.raises(RuntimeError):
        ref_call(x, 1, 1, rmin, rmax, sc, zp, 0.01, 1, 255, 1, True)


def test_fq_triton_paths(monkeypatch):
    mod = importlib.import_module("flaggems._fused_moving_avg_obs_fq_helper")

    counts = {"fused": 0, "partials": 0, "finalize": 0, "elem": 0, "copy": 0}
    kernels = {
        "fused": mod._fused_kernel,
        "partials": mod._reduce_partials,
        "finalize": mod._finalize_pt,
        "elem": mod._fq_elem,
        "copy": mod._copy_elem,
    }

    def spy(name, real):
        def spy_run(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)

        return spy_run

    for name, kernel in kernels.items():
        monkeypatch.setattr(kernel, "run", spy(name, kernel.run))

    x = make_x((2, 64), seed=17)
    rmin, rmax, sc, zp = fresh_state(1)
    _fused_moving_avg_obs_fq_helper(x, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0)
    assert counts["fused"] == 1
    assert counts["partials"] == 0
    assert counts["finalize"] == 0
    assert counts["elem"] == 0
    assert counts["copy"] == 0

    x_big = make_x((4097,), seed=18)
    rmin, rmax, sc, zp = fresh_state(1)
    _fused_moving_avg_obs_fq_helper(
        x_big, 1, 1, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    assert counts["fused"] == 1  # unchanged: the large path bypasses it
    assert counts["partials"] == 1
    assert counts["finalize"] == 1
    assert counts["elem"] == 1
    assert counts["copy"] == 0

    x_big2 = make_x((4097,), seed=19)
    rmin, rmax, sc, zp = fresh_state(1)
    _fused_moving_avg_obs_fq_helper(
        x_big2, 1, 0, rmin, rmax, sc, zp, 0.01, 0, 255, 0
    )
    assert counts["copy"] == 1
    assert counts["elem"] == 1

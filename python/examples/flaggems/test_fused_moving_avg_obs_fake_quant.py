import importlib

import torch

from .fused_moving_avg_obs_fake_quant import fused_moving_avg_obs_fake_quant


def _ref_fused(
    input,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    symmetric_quant=False,
):
    """Reference from the upstream FlagGems test suite (pure PyTorch)."""
    x = input.float()
    if observer_on:
        running_min.lerp_(x.min(), averaging_const)
        running_max.lerp_(x.max(), averaging_const)
    rmin = min(running_min.item(), 0.0)
    rmax = max(running_max.item(), 0.0)
    qmin_f, qmax_f = float(quant_min), float(quant_max)
    sc = (rmax - rmin) / (qmax_f - qmin_f)
    if sc == 0.0:
        sc = 0.1
    zp = round(qmin_f - rmin / sc)
    zp = min(max(zp, qmin_f), qmax_f)
    if symmetric_quant and rmin < 0.0 and rmax > 0.0:
        sc = max(-rmin / (-qmin_f), rmax / qmax_f)
        if sc == 0.0:
            sc = 0.1
        zp = round((qmin_f + qmax_f) / 2.0)
    scale.fill_(sc)
    zero_point.fill_(int(zp))
    if fake_quant_on:
        sc_t = torch.tensor(sc, dtype=torch.float32, device=input.device)
        zp_t = torch.tensor(zp, dtype=torch.float32, device=input.device)
        q = torch.round(x / sc_t) + zp_t
        q = q.clamp(qmin_f, qmax_f)
        return ((q - zp_t) * sc_t).to(input.dtype)
    return input.clone()


def _state():
    return (
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.zeros(1, dtype=torch.int32),
    )


def _run(
    x,
    observer_on,
    fake_quant_on,
    averaging_const=0.01,
    symmetric=False,
    quant_min=0,
    quant_max=255,
):
    rmin, rmax, sct, zpt = _state()
    out = fused_moving_avg_obs_fake_quant(
        x,
        torch.tensor([int(observer_on)], dtype=torch.int32),
        torch.tensor([int(fake_quant_on)], dtype=torch.int32),
        rmin,
        rmax,
        sct,
        zpt,
        averaging_const,
        quant_min,
        quant_max,
        -1,
        symmetric_quant=symmetric,
    )
    return out, rmin, rmax, sct, zpt


def test_fused_output_and_observer_state_all_flag_combinations():
    torch.manual_seed(42)
    x = torch.randn(2048)
    for observer_on in (True, False):
        for fake_quant_on in (True, False):
            out, rmin, rmax, sct, zpt = _run(x, observer_on, fake_quant_on)
            rmin_r, rmax_r, sct_r, zpt_r = _state()
            ref = _ref_fused(
                x,
                observer_on,
                fake_quant_on,
                rmin_r,
                rmax_r,
                sct_r,
                zpt_r,
                0.01,
                0,
                255,
            )
            # value check within one quantum of the derived scale
            torch.testing.assert_close(
                out,
                ref,
                rtol=0,
                atol=float(sct_r) * 1.01,
                msg=f"obs={observer_on},fq={fake_quant_on}",
            )
            # observer state: EMA updates only when the observer is on
            torch.testing.assert_close(rmin, rmin_r, rtol=0, atol=1e-5)
            torch.testing.assert_close(rmax, rmax_r, rtol=0, atol=1e-5)
            # scale / zero_point are always refreshed (even with fake quant off)
            torch.testing.assert_close(sct, sct_r, rtol=0, atol=1e-5)
            assert zpt.dtype == torch.int32
            assert zpt.item() == int(zpt_r.item())


def test_fused_observer_off_leaves_running_state_untouched():
    torch.manual_seed(0)
    x = torch.randn(100)
    out, rmin, rmax, sct, zpt = _run(x, False, True)
    assert rmin.item() == -1.0 and rmax.item() == 1.0


def test_fused_fake_quant_off_returns_input_bitwise():
    torch.manual_seed(1)
    x = torch.randn(256)
    out, _, _, _, _ = _run(x, True, False)
    assert torch.equal(out, x)
    out, _, _, _, _ = _run(x, False, False)
    assert torch.equal(out, x)


def test_fused_matches_aten_helper_with_observer_on():
    # aten._fused_moving_avg_obs_fq_helper agrees bit-exactly when the
    # observer runs; it returns (output, mask) and mutates the same state.
    torch.manual_seed(2)
    x = torch.randn(1024)
    out, rmin, rmax, sct, zpt = _run(x, True, True)
    rmin_a, rmax_a, sct_a, zpt_a = _state()
    aten_out, _mask = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        x,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        rmin_a,
        rmax_a,
        sct_a,
        zpt_a,
        0.01,
        0,
        255,
        -1,
    )
    torch.testing.assert_close(out, aten_out, rtol=0, atol=0)
    assert rmin.item() == rmin_a.item() and rmax.item() == rmax_a.item()
    assert sct.item() == sct_a.item() and zpt.item() == zpt_a.item()


def test_fused_symmetric_quant_signed_range():
    torch.manual_seed(3)
    x = torch.randn(500)
    out, rmin, rmax, sct, zpt = _run(
        x, True, True, symmetric=True, quant_min=-128, quant_max=127
    )
    rmin_r, rmax_r, sct_r, zpt_r = _state()
    ref = _ref_fused(
        x,
        True,
        True,
        rmin_r,
        rmax_r,
        sct_r,
        zpt_r,
        0.01,
        -128,
        127,
        symmetric_quant=True,
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=float(sct_r) * 1.01)
    torch.testing.assert_close(sct, sct_r, rtol=0, atol=1e-5)
    # symmetric zero point is the midpoint of the signed range
    assert zpt.item() == 0


def test_fused_shapes_and_dtypes():
    torch.manual_seed(4)
    for shape in ((100,), (2, 3, 17), (5, 33)):
        x = torch.randn(shape)
        out, *_ = _run(x, True, True)
        assert isinstance(out, torch.Tensor)
        assert out.shape == shape and out.dtype == torch.float32
    xh = torch.randn(64, dtype=torch.float16)
    out, _, _, sct, _ = _run(xh, True, True)
    assert out.dtype == torch.float16
    rmin_r, rmax_r, sct_r, zpt_r = _state()
    ref = _ref_fused(xh, True, True, rmin_r, rmax_r, sct_r, zpt_r, 0.01, 0, 255)
    torch.testing.assert_close(out, ref, rtol=0, atol=float(sct_r) * 1.01)


def test_fused_averaging_const_extremes():
    torch.manual_seed(5)
    x = torch.randn(64)
    out, rmin, rmax, sct, zpt = _run(x, True, True, averaging_const=1.0)
    # full replacement: running min/max become the batch min/max
    assert rmin.item() == x.min().item()
    assert rmax.item() == x.max().item()
    out, rmin, rmax, sct, zpt = _run(x, True, True, averaging_const=0.0)
    assert rmin.item() == -1.0 and rmax.item() == 1.0


def test_fused_scales_write_is_part_of_contract():
    torch.manual_seed(6)
    x = torch.rand(128) + 0.5  # all-positive data: rmin clamps to 0.0
    out, rmin, rmax, sct, zpt = _run(x, True, False)
    rmin_r, rmax_r, sct_r, zpt_r = _state()
    _ref_fused(x, True, False, rmin_r, rmax_r, sct_r, zpt_r, 0.01, 0, 255)
    assert sct.item() == sct_r.item()
    assert zpt.item() == zpt_r.item()


def test_fused_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fused_moving_avg_obs_fake_quant")
    for kernel_name in ("_minmax_reduce_kernel", "_fake_quant_kernel"):
        launches = []
        kernel = getattr(mod, kernel_name)
        real_run = kernel.run

        def spy_run(*args, _real=real_run, **kwargs):
            launches.append(kwargs.get("grid"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        torch.manual_seed(7)
        x = torch.randn(2048)
        out, *_ = _run(x, True, True)
        assert len(launches) >= 1, f"{kernel_name} must be launched"
        rmin_r, rmax_r, sct_r, zpt_r = _state()
        ref = _ref_fused(x, True, True, rmin_r, rmax_r, sct_r, zpt_r, 0.01, 0, 255)
        torch.testing.assert_close(out, ref, rtol=0, atol=float(sct_r) * 1.01)

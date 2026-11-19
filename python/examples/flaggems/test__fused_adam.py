import importlib

import pytest
import torch

from ._fused_adam import _fused_adam, _fused_adam_

T = torch.tensor


def make_state(n, seed=0, amsgrad=False):
    torch.manual_seed(seed)
    param = torch.randn(n)
    grad = torch.randn(n)
    exp_avg = torch.randn(n) * 0.1
    exp_avg_sq = torch.rand(n) * 0.1
    max_exp_avg_sq = torch.rand(n) * 0.1 if amsgrad else None
    return param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq


def clone_state(state):
    return tuple(t.clone() if t is not None else None for t in state)


def ref_step(states, steps, **kw):
    params = [s[0] for s in states]
    grads = [s[1] for s in states]
    eas = [s[2] for s in states]
    easq = [s[3] for s in states]
    maxes = [s[4] if s[4] is not None else torch.empty(0) for s in states]
    amsgrad = kw.get("amsgrad", False)
    torch.ops.aten._fused_adam_(
        params,
        grads,
        eas,
        easq,
        maxes if amsgrad else [],
        [T(s) for s in steps],
        lr=kw.get("lr", 0.001),
        beta1=kw.get("beta1", 0.9),
        beta2=kw.get("beta2", 0.999),
        weight_decay=kw.get("weight_decay", 0.0),
        eps=kw.get("eps", 1e-8),
        amsgrad=amsgrad,
        maximize=kw.get("maximize", False),
    )
    return params, grads, eas, easq


@pytest.mark.parametrize("n", [128, 1023, 4097])
def test_fused_adam_matches_aten(n):
    st = make_state(n, seed=0)
    ref = clone_state(st)
    port = clone_state(st)

    _fused_adam(
        [port[0]],
        [port[1]],
        [port[2]],
        [port[3]],
        [port[4]] if port[4] is not None else [],
        [T(1)],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
    )
    rp, rg, rma, rmsq = ref_step([ref], [1], lr=0.01, beta1=0.9, beta2=0.999)

    torch.testing.assert_close(port[0], rp[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], rma[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[3], rmsq[0], rtol=1e-6, atol=1e-7)


def test_fused_adam_bias_correction_multi_step():
    st = make_state(256, seed=1)
    ref = clone_state(st)
    port = clone_state(st)
    for step in (1, 2, 5):
        _fused_adam([port[0]], [port[1]], [port[2]], [port[3]], [], [T(step)], lr=0.05)
        ref_step([ref], [step], lr=0.05)
        torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(port[3], ref[3], rtol=1e-6, atol=1e-7)


def test_fused_adam_amsgrad():
    st = make_state(512, seed=2, amsgrad=True)
    ref = clone_state(st)
    port = clone_state(st)
    _fused_adam(
        [port[0]], [port[1]], [port[2]], [port[3]], [port[4]], [T(1)], amsgrad=True
    )
    rp, _, rma, rmsq = ref_step([ref], [1], amsgrad=True)
    torch.testing.assert_close(port[0], rp[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[4], ref[4], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[3], rmsq[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("weight_decay", [0.01, 0.1])
def test_fused_adam_weight_decay(weight_decay):
    st = make_state(256, seed=3)
    ref = clone_state(st)
    port = clone_state(st)
    _fused_adam(
        [port[0]],
        [port[1]],
        [port[2]],
        [port[3]],
        [],
        [T(1)],
        weight_decay=weight_decay,
    )
    rp, _, _, _ = ref_step([ref], [1], weight_decay=weight_decay)
    torch.testing.assert_close(port[0], rp[0], rtol=1e-6, atol=1e-7)


def test_fused_adam_maximize():
    st = make_state(256, seed=4)
    ref = clone_state(st)
    port = clone_state(st)
    _fused_adam([port[0]], [port[1]], [port[2]], [port[3]], [], [T(1)], maximize=True)
    rp, _, _, _ = ref_step([ref], [1], maximize=True)
    torch.testing.assert_close(port[0], rp[0], rtol=1e-6, atol=1e-7)


def test_fused_adam_grad_scale():
    st = make_state(128, seed=5)
    ref = clone_state(st)
    port = clone_state(st)
    scale = 0.5
    _fused_adam(
        [port[0]], [port[1]], [port[2]], [port[3]], [], [T(1)], grad_scale=scale
    )
    torch.ops.aten._fused_adam_(
        [ref[0]],
        [ref[1] * scale],
        [ref[2]],
        [ref[3]],
        [],
        [T(1)],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


def test_fused_adam_found_inf_skips():
    st = make_state(128, seed=6)
    port = clone_state(st)
    before = [t.clone() if t is not None else None for t in port]
    _fused_adam(
        [port[0]],
        [port[1]],
        [port[2]],
        [port[3]],
        [],
        [T(1)],
        found_inf=T(1.0),
    )
    for a, b in zip(port, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fused_adam_multiple_groups():
    states = [make_state(128, seed=7), make_state(64, seed=8), make_state(300, seed=9)]
    refs = [clone_state(s) for s in states]
    ports = [clone_state(s) for s in states]
    _fused_adam(
        [p[0] for p in ports],
        [p[1] for p in ports],
        [p[2] for p in ports],
        [p[3] for p in ports],
        [],
        [T(1)] * 3,
        lr=0.02,
    )
    rp, _, rma, rmsq = ref_step(refs, [1, 1, 1], lr=0.02)
    for i in range(3):
        torch.testing.assert_close(ports[i][0], rp[i], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ports[i][2], rma[i], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ports[i][3], rmsq[i], rtol=1e-6, atol=1e-7)


def test_fused_adam_inplace_returns_none():
    st = make_state(128, seed=10)
    port = clone_state(st)
    ref = clone_state(st)
    ret = _fused_adam_(
        [port[0]],
        [port[1]],
        [port[2]],
        [port[3]],
        [],
        [T(1)],
    )
    assert ret is None
    ref_step([ref], [1])
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


def test_fused_adam_rejects_non_float32():
    param = torch.randn(8, dtype=torch.float64)
    with pytest.raises(AssertionError):
        _fused_adam(
            [param], [torch.randn(8)], [torch.zeros(8)], [torch.zeros(8)], [], [T(1)]
        )


def test_fused_adam_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._fused_adam")
    calls = []
    real_run = mod._fused_adam_kernel.run

    def spy_run(*args, **kwargs):
        calls.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._fused_adam_kernel, "run", spy_run)
    st = make_state(128, seed=11)
    _fused_adam([st[0]], [st[1]], [st[2]], [st[3]], [], [T(1)])
    assert len(calls) == 1

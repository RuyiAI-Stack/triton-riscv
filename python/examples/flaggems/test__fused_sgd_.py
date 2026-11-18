import importlib

import pytest
import torch

from ._fused_sgd_ import _fused_sgd_

T = torch.tensor


def make(n, seed=0, with_momentum=True):
    torch.manual_seed(seed)
    param = torch.randn(n)
    grad = torch.randn(n)
    buf = torch.randn(n) * 0.1 if with_momentum else None
    return param, grad, buf


def ref_step(params, grads, bufs, **kw):
    torch.ops.aten._fused_sgd_(
        params,
        grads,
        bufs,
        weight_decay=kw.get("weight_decay", 0.0),
        momentum=kw.get("momentum", 0.0),
        lr=kw.get("lr", 0.01),
        dampening=kw.get("dampening", 0.0),
        nesterov=kw.get("nesterov", False),
        maximize=kw.get("maximize", False),
        is_first_step=kw.get("is_first_step", False),
    )


@pytest.mark.parametrize("n", [128, 1023, 4097])
def test_fused_sgd_plain(n):
    p, g, _ = make(n, seed=0, with_momentum=False)
    ref = (p.clone(), g.clone())
    port = (p.clone(), g.clone())
    ret = _fused_sgd_([port[0]], [port[1]], [], lr=0.01)
    assert ret is None
    ref_step([ref[0]], [ref[1]], [], lr=0.01)
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("momentum", [0.5, 0.9])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("is_first_step", [False, True])
def test_fused_sgd_momentum_matrix(momentum, nesterov, is_first_step):
    p, g, b = make(256, seed=1)
    ref = (p.clone(), g.clone(), b.clone())
    port = (p.clone(), g.clone(), b.clone())
    _fused_sgd_(
        [port[0]],
        [port[1]],
        [port[2]],
        momentum=momentum,
        lr=0.01,
        nesterov=nesterov,
        is_first_step=is_first_step,
    )
    ref_step(
        [ref[0]],
        [ref[1]],
        [ref[2]],
        momentum=momentum,
        lr=0.01,
        nesterov=nesterov,
        is_first_step=is_first_step,
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("weight_decay", [0.01, 0.1])
def test_fused_sgd_weight_decay(weight_decay):
    p, g, b = make(256, seed=2)
    ref = (p.clone(), g.clone(), b.clone())
    port = (p.clone(), g.clone(), b.clone())
    _fused_sgd_(
        [port[0]],
        [port[1]],
        [port[2]],
        momentum=0.9,
        lr=0.01,
        weight_decay=weight_decay,
    )
    ref_step(
        [ref[0]], [ref[1]], [ref[2]], momentum=0.9, lr=0.01, weight_decay=weight_decay
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


def test_fused_sgd_dampening():
    p, g, b = make(256, seed=3)
    ref = (p.clone(), g.clone(), b.clone())
    port = (p.clone(), g.clone(), b.clone())
    _fused_sgd_([port[0]], [port[1]], [port[2]], momentum=0.9, lr=0.01, dampening=0.5)
    ref_step([ref[0]], [ref[1]], [ref[2]], momentum=0.9, lr=0.01, dampening=0.5)
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)


def test_fused_sgd_maximize():
    p, g, b = make(256, seed=4)
    ref = (p.clone(), g.clone(), b.clone())
    port = (p.clone(), g.clone(), b.clone())
    _fused_sgd_([port[0]], [port[1]], [port[2]], momentum=0.9, lr=0.01, maximize=True)
    ref_step([ref[0]], [ref[1]], [ref[2]], momentum=0.9, lr=0.01, maximize=True)
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("found_inf_val", [1.0, 2.0])
def test_fused_sgd_found_inf_exact(found_inf_val):
    p, g, b = make(128, seed=5)
    ref = (p.clone(), g.clone(), b.clone())
    port = (p.clone(), g.clone(), b.clone())
    fi = T(found_inf_val)
    _fused_sgd_([port[0]], [port[1]], [port[2]], momentum=0.9, lr=0.01, found_inf=fi)
    torch.ops.aten._fused_sgd_(
        [ref[0]],
        [ref[1]],
        [ref[2]],
        weight_decay=0.0,
        momentum=0.9,
        lr=0.01,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        found_inf=fi.clone(),
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


def test_fused_sgd_grad_scale_writes_back_grad():
    p, g, b = make(128, seed=6)
    ref = (p.clone(), g.clone(), b.clone())
    port = (p.clone(), g.clone(), b.clone())
    _fused_sgd_(
        [port[0]], [port[1]], [port[2]], momentum=0.9, lr=0.01, grad_scale=T(0.5)
    )
    torch.ops.aten._fused_sgd_(
        [ref[0]],
        [ref[1]],
        [ref[2]],
        weight_decay=0.0,
        momentum=0.9,
        lr=0.01,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        grad_scale=T(0.5),
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[1], ref[1], rtol=1e-6, atol=1e-7)


def test_fused_sgd_multiple_groups():
    groups = [make(128, seed=7), make(300, seed=8)]
    refs = [tuple(t.clone() for t in grp) for grp in groups]
    ports = [tuple(t.clone() for t in grp) for grp in groups]
    _fused_sgd_(
        [grp[0] for grp in ports],
        [grp[1] for grp in ports],
        [grp[2] for grp in ports],
        momentum=0.9,
        lr=0.01,
    )
    ref_step(
        [r[0] for r in refs],
        [r[1] for r in refs],
        [r[2] for r in refs],
        momentum=0.9,
        lr=0.01,
    )
    for i in range(2):
        torch.testing.assert_close(ports[i][0], refs[i][0], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ports[i][2], refs[i][2], rtol=1e-6, atol=1e-7)


def test_fused_sgd_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._fused_sgd_")
    calls = []
    real_run = mod._fused_sgd_kernel.run

    def spy_run(*args, **kwargs):
        calls.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._fused_sgd_kernel, "run", spy_run)
    p, g, b = make(128, seed=9)
    _fused_sgd_([p], [g], [b], momentum=0.9, lr=0.01)
    assert len(calls) == 1

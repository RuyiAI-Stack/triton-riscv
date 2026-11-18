import importlib

import pytest
import torch

from ._fused_adagrad_ import _fused_adagrad_

T = torch.tensor


def make(n, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return (
        torch.randn(n, dtype=dtype),
        torch.randn(n, dtype=dtype),
        torch.rand(n, dtype=dtype) * 0.1,
    )


def ref_step(params, grads, sums, steps, **kw):
    torch.ops.aten._fused_adagrad_(
        params,
        grads,
        sums,
        [T(s) for s in steps],
        lr=kw.get("lr", 1e-2),
        lr_decay=kw.get("lr_decay", 0.0),
        weight_decay=kw.get("weight_decay", 0.0),
        eps=kw.get("eps", 1e-10),
        maximize=kw.get("maximize", False),
    )


@pytest.mark.parametrize("n", [128, 1023, 4097])
def test_fused_adagrad_matches_aten(n):
    p, g, s = make(n, seed=0)
    ref = (p.clone(), g.clone(), s.clone())
    port = (p.clone(), g.clone(), s.clone())

    ret = _fused_adagrad_([port[0]], [port[1]], [port[2]], [T(2)])
    assert ret is None
    ref_step([ref[0]], [ref[1]], [ref[2]], [2])

    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[1], ref[1], rtol=0, atol=0)  # grads read-only


@pytest.mark.parametrize("lr_decay", [0.0, 0.1])
def test_fused_adagrad_lr_decay(lr_decay):
    p, g, s = make(256, seed=1)
    ref = (p.clone(), g.clone(), s.clone())
    port = (p.clone(), g.clone(), s.clone())
    _fused_adagrad_([port[0]], [port[1]], [port[2]], [T(5)], lr_decay=lr_decay)
    ref_step([ref[0]], [ref[1]], [ref[2]], [5], lr_decay=lr_decay)
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("weight_decay", [0.01, 0.1])
def test_fused_adagrad_weight_decay(weight_decay):
    p, g, s = make(256, seed=2)
    ref = (p.clone(), g.clone(), s.clone())
    port = (p.clone(), g.clone(), s.clone())
    _fused_adagrad_([port[0]], [port[1]], [port[2]], [T(1)], weight_decay=weight_decay)
    ref_step([ref[0]], [ref[1]], [ref[2]], [1], weight_decay=weight_decay)
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


def test_fused_adagrad_maximize():
    p, g, s = make(256, seed=3)
    ref = (p.clone(), g.clone(), s.clone())
    port = (p.clone(), g.clone(), s.clone())
    _fused_adagrad_([port[0]], [port[1]], [port[2]], [T(1)], maximize=True)
    ref_step([ref[0]], [ref[1]], [ref[2]], [1], maximize=True)
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)


def test_fused_adagrad_grad_scale():
    p, g, s = make(128, seed=4)
    ref = (p.clone(), g.clone(), s.clone())
    port = (p.clone(), g.clone(), s.clone())
    _fused_adagrad_([port[0]], [port[1]], [port[2]], [T(1)], grad_scale=T(0.5))
    torch.ops.aten._fused_adagrad_(
        [ref[0]],
        [ref[1] / 0.5],
        [ref[2]],
        [T(1)],
        lr=1e-2,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("found_inf_val", [1.0, 2.0])
def test_fused_adagrad_found_inf_exact_compare(found_inf_val):
    # aten skips only when found_inf == 1 exactly; 2.0 still applies the step.
    p, g, s = make(128, seed=5)
    ref = (p.clone(), g.clone(), s.clone())
    port = (p.clone(), g.clone(), s.clone())
    fi = T(found_inf_val)
    _fused_adagrad_([port[0]], [port[1]], [port[2]], [T(1)], found_inf=fi)
    torch.ops.aten._fused_adagrad_(
        [ref[0]],
        [ref[1]],
        [ref[2]],
        [T(1)],
        lr=1e-2,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
        found_inf=fi.clone(),
    )
    torch.testing.assert_close(port[0], ref[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(port[2], ref[2], rtol=1e-6, atol=1e-7)


def test_fused_adagrad_multiple_groups():
    groups = [make(128, seed=6), make(64, seed=7)]
    refs = [tuple(t.clone() for t in grp) for grp in groups]
    ports = [tuple(t.clone() for t in grp) for grp in groups]
    _fused_adagrad_(
        [grp[0] for grp in ports],
        [grp[1] for grp in ports],
        [grp[2] for grp in ports],
        [T(3), T(3)],
        lr=0.05,
    )
    ref_step(
        [r[0] for r in refs],
        [r[1] for r in refs],
        [r[2] for r in refs],
        [3, 3],
        lr=0.05,
    )
    for i in range(2):
        torch.testing.assert_close(ports[i][0], refs[i][0], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ports[i][2], refs[i][2], rtol=1e-6, atol=1e-7)


def test_fused_adagrad_length_mismatch():
    p, g, s = make(64, seed=8)
    with pytest.raises(RuntimeError):
        _fused_adagrad_([p], [g], [], [T(1)])


def test_fused_adagrad_rejects_unsupported_dtype():
    p = torch.randint(0, 8, (8,), dtype=torch.int32)
    g = torch.randint(0, 8, (8,), dtype=torch.int32)
    s = torch.randint(0, 8, (8,), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _fused_adagrad_([p], [g], [s], [T(1)])


def test_fused_adagrad_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._fused_adagrad_")
    calls = []
    real_run = mod._fused_adagrad_kernel.run

    def spy_run(*args, **kwargs):
        calls.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._fused_adagrad_kernel, "run", spy_run)
    p, g, s = make(128, seed=9)
    _fused_adagrad_([p], [g], [s], [T(1)])
    assert len(calls) == 1

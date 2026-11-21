import importlib

import pytest
import torch

from ._amp_update_scale_ import _amp_update_scale_
from .test__amp_foreach_non_finite_check_and_unscale_ import (
    AMP_BACKOFF,
    AMP_GROWTH,
    AMP_GROWTH_INTERVAL,
    torch_update_scale_ref,
)

# Bit-exactness (probed against torch 2.15 CPU): the state machine is a
# single fp32 multiply on each transition, exact for the power-of-two /
# 1.5-chain factors used below, so every comparison is rtol=0, atol=0.
#
# State machine (aten contract, probed):
#   found_inf != 0 (any non-zero, including NaN)  -> scale *= backoff,
#                                                    growth_tracker = 0
#   otherwise                                     -> tracker += 1; when
#     tracker == growth_interval: scale *= growth and tracker = 0.
# The op returns `self` (the scale tensor) and updates both scalar
# tensors in place.  Gates, in probed aten order: growth_tracker must be
# int32 ("growth_tracker must be an int tensor."), self float32
# ("current_scale must be a float tensor."), found_inf float32
# ("found_inf must be a float tensor.")


def test_update_scale_growth_mid(amp_scale_state):
    scale, tracker, found_inf = amp_scale_state
    tracker.fill_(3)

    ref_s, ref_t = torch_update_scale_ref(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    ret = _amp_update_scale_(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    assert ret is scale
    assert scale.item() == ref_s.item() == 65536.0
    assert tracker.item() == ref_t.item() == 4


def test_update_scale_growth_hits_interval(amp_scale_state):
    scale, tracker, found_inf = amp_scale_state
    tracker.fill_(AMP_GROWTH_INTERVAL - 1)

    ref_s, ref_t = torch_update_scale_ref(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    ret = _amp_update_scale_(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    assert ret is scale
    assert scale.item() == ref_s.item() == 131072.0
    assert tracker.item() == ref_t.item() == 0


def test_update_scale_backoff_resets_tracker(amp_scale_state):
    # aten resets the growth streak on backoff (probed); the scale is
    # multiplied by the backoff factor and the tracker goes to 0.
    scale, tracker, found_inf = amp_scale_state
    tracker.fill_(7)
    found_inf.fill_(1.0)

    ref_s, ref_t = torch_update_scale_ref(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    ret = _amp_update_scale_(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    assert ret is scale
    assert scale.item() == ref_s.item() == 32768.0
    assert tracker.item() == ref_t.item() == 0


@pytest.mark.parametrize("found_value", [-1.0, 2.0, float("nan")])
def test_update_scale_found_inf_nonzero_variants(amp_scale_state, found_value):
    # Any non-zero found_inf (negative, > 1, NaN) selects the backoff
    # branch -- the test is `!= 0`, not `> 0`.
    scale, tracker, found_inf = amp_scale_state
    tracker.fill_(5)
    found_inf.fill_(found_value)

    ref_s, ref_t = torch_update_scale_ref(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    _amp_update_scale_(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    assert scale.item() == ref_s.item()
    assert tracker.item() == ref_t.item() == 0


def test_update_scale_sequence(amp_scale_state):
    # Multi-step schedule with growth_interval=2: grow on every second
    # finite step, back off (and reset the streak) on inf.  Expected
    # chain probed against aten: (s,1) (2s,0) (s,0) (s,1) (2s,0).
    scale, tracker, found_inf = amp_scale_state
    interval = 2

    ref_s, ref_t = scale.clone(), tracker.clone()
    port_s, port_t = scale.clone(), tracker.clone()
    ref_found = port_found = torch.zeros(())

    for f in (0.0, 0.0, 1.0, 0.0, 0.0):
        torch._amp_update_scale_(
            ref_s, ref_t, ref_found.fill_(f), AMP_GROWTH, AMP_BACKOFF, interval
        )
        _amp_update_scale_(
            port_s, port_t, port_found.fill_(f), AMP_GROWTH, AMP_BACKOFF, interval
        )
        assert port_s.item() == ref_s.item()
        assert port_t.item() == ref_t.item()

    assert port_s.item() == 131072.0  # 65536 grew twice net of one backoff
    assert port_t.item() == 0


def test_update_scale_growth_interval_one(amp_scale_state):
    # interval == 1: tracker+1 == 1 immediately, so every finite step
    # grows the scale (probed).
    scale, tracker, found_inf = amp_scale_state
    tracker.fill_(0)

    ref_s, _ = torch_update_scale_ref(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, 1
    )
    _amp_update_scale_(scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, 1)

    assert scale.item() == ref_s.item() == 131072.0
    assert tracker.item() == 0


def test_update_scale_exact_chain_15(amp_scale_state):
    # Non-power-of-two growth keeps the fp32 chain exact: 4 * 1.5**3.
    scale, tracker, found_inf = amp_scale_state
    scale.fill_(4.0)

    ref_s, _ = torch_update_scale_ref(scale, tracker, found_inf, 1.5, 0.5, 1)
    for _ in range(2):
        ref_s, _ = torch_update_scale_ref(ref_s, tracker, found_inf, 1.5, 0.5, 1)
    for _ in range(3):
        _amp_update_scale_(scale, tracker, found_inf, 1.5, 0.5, 1)

    assert scale.item() == ref_s.item() == 13.5


def test_update_scale_inplace_identity(amp_scale_state):
    scale, tracker, found_inf = amp_scale_state
    scale_ptr, tracker_ptr = scale.data_ptr(), tracker.data_ptr()

    ret = _amp_update_scale_(
        scale, tracker, found_inf, AMP_GROWTH, AMP_BACKOFF, AMP_GROWTH_INTERVAL
    )

    assert ret is scale
    assert scale.data_ptr() == scale_ptr
    assert tracker.data_ptr() == tracker_ptr
    assert tracker.item() == 1  # first step of the growth streak


def test_update_scale_accepts_python_int_factors(amp_scale_state):
    scale, tracker, found_inf = amp_scale_state
    scale.fill_(4.0)
    tracker.fill_(1)

    ref_s, ref_t = torch_update_scale_ref(scale, tracker, found_inf, 2, 1, 2)
    ret = _amp_update_scale_(scale, tracker, found_inf, 2, 1, 2)

    assert ret is scale
    assert scale.item() == ref_s.item() == 8.0
    assert tracker.item() == ref_t.item() == 0


def test_update_scale_dtype_gates():
    scale = torch.tensor(4.0)
    good_tracker = torch.tensor(3, dtype=torch.int32)
    good_found = torch.tensor(0.0)

    # Probed aten gate order: growth_tracker, then self, then found_inf.
    with pytest.raises(RuntimeError, match="growth_tracker must be an int tensor."):
        _amp_update_scale_(
            torch.tensor(4), torch.tensor(3.0), torch.tensor(0), 2.0, 0.5, 2
        )
    with pytest.raises(RuntimeError, match="growth_tracker must be an int tensor."):
        _amp_update_scale_(
            scale, torch.tensor(3, dtype=torch.int64), good_found, 2.0, 0.5, 2
        )
    with pytest.raises(RuntimeError, match="growth_tracker must be an int tensor."):
        _amp_update_scale_(
            scale, torch.tensor(3, dtype=torch.int16), good_found, 2.0, 0.5, 2
        )
    with pytest.raises(RuntimeError, match="current_scale must be a float tensor."):
        _amp_update_scale_(
            torch.tensor(4, dtype=torch.int64),
            good_tracker,
            torch.tensor(0),
            2.0,
            0.5,
            2,
        )
    with pytest.raises(RuntimeError, match="current_scale must be a float tensor."):
        _amp_update_scale_(
            torch.tensor(4.0, dtype=torch.float64),
            good_tracker,
            good_found,
            2.0,
            0.5,
            2,
        )
    with pytest.raises(RuntimeError, match="found_inf must be a float tensor."):
        _amp_update_scale_(
            scale, good_tracker, torch.tensor(0.0, dtype=torch.float16), 2.0, 0.5, 2
        )
    with pytest.raises(RuntimeError, match="found_inf must be a float tensor."):
        _amp_update_scale_(
            scale, good_tracker, torch.tensor(0, dtype=torch.int32), 2.0, 0.5, 2
        )


def test_update_scale_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    mod = importlib.import_module("flaggems._amp_update_scale_")

    launches = []
    real_run = mod._amp_update_scale_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._amp_update_scale_kernel, "run", spy_run)
    scale, tracker, found_inf = (
        torch.tensor(4.0),
        torch.tensor(3, dtype=torch.int32),
        torch.tensor(0.0),
    )
    mod._amp_update_scale_(scale, tracker, found_inf, 2.0, 0.5, 2000)
    assert len(launches) == 1, "update_scale must go through the Triton kernel"
    assert launches[0] == (1,)

import torch
import triton
import triton.language as tl


@triton.jit
def _amp_update_scale_kernel(
    scale_ptr,
    growth_tracker_ptr,
    found_inf_ptr,
    scale_growth_factor,
    scale_backoff_factor,
    growth_interval,
):
    # All operands are 0-dim; a single program runs the whole state
    # machine (the same shape as aten's single-thread kernel).
    scale = tl.load(scale_ptr)
    found_inf = tl.load(found_inf_ptr)
    tracker = tl.load(growth_tracker_ptr)

    # Any non-zero found_inf (negative, > 1, NaN) selects backoff:
    # shrink the scale and reset the growth streak.
    has_inf = found_inf != 0.0

    # Growth branch: increment the streak; once it reaches the
    # interval, scale up and reset.
    new_tracker_grow = tracker + 1
    reached_interval = new_tracker_grow == growth_interval
    tracker_grow = tl.where(reached_interval, 0, new_tracker_grow)
    scale_grow = tl.where(reached_interval, scale * scale_growth_factor, scale)

    # Backoff branch: shrink and reset the streak (probed: aten CPU and
    # CUDA both reset the tracker here).
    scale_backoff = scale * scale_backoff_factor

    new_scale = tl.where(has_inf, scale_backoff, scale_grow)
    new_tracker = tl.where(has_inf, 0, tracker_grow)

    tl.store(scale_ptr, new_scale)
    tl.store(growth_tracker_ptr, new_tracker)


def _amp_update_scale_(
    self,
    growth_tracker,
    found_inf,
    scale_growth_factor,
    scale_backoff_factor,
    growth_interval,
):
    """Update the AMP loss scale in place.

    Mirrors ``aten::_amp_update_scale_`` (probed on torch 2.15 CPU):

    - ``found_inf != 0`` -> ``self *= scale_backoff_factor`` and the
      growth tracker resets to 0;
    - otherwise the tracker increments, and when it reaches
      ``growth_interval`` the scale grows by ``scale_growth_factor``
      and the tracker resets to 0;
    - returns ``self``; both scalar tensors are updated in place.

    Gate order and messages are aten's (probed): the tracker must be
    int32, then self float32, then found_inf float32.  Upstream
    FlagGems accepts wider int trackers but converts to int32 first,
    which silently drops the in-place update -- pinned to aten here.
    """
    if growth_tracker.dtype != torch.int32:
        raise RuntimeError("growth_tracker must be an int tensor.")
    if self.dtype != torch.float32:
        raise RuntimeError("current_scale must be a float tensor.")
    if found_inf.dtype != torch.float32:
        raise RuntimeError("found_inf must be a float tensor.")

    _amp_update_scale_kernel[(1,)](
        self,
        growth_tracker,
        found_inf,
        float(scale_growth_factor),
        float(scale_backoff_factor),
        int(growth_interval),
    )

    return self

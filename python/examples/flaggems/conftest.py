import pytest
import torch


@pytest.fixture
def amp_scale_state():
    """Fresh (scale, growth_tracker, found_inf) triple in the layout the
    two AMP ops exchange; shared by the AMP foreach/update-scale suites."""
    scale = torch.tensor(65536.0)
    growth_tracker = torch.zeros((), dtype=torch.int32)
    found_inf = torch.zeros(())
    return scale, growth_tracker, found_inf

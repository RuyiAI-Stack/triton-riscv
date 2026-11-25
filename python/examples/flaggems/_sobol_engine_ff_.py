from .detach_copy import _copy_values


def _sobol_engine_ff_(state):
    return _copy_values(state)

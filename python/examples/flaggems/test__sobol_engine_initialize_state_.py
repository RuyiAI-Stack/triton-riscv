from ._sobol_engine_initialize_state_ import _sobol_engine_initialize_state_


def test__sobol_engine_initialize_state_():
    state = _sobol_engine_initialize_state_(2)
    assert state.shape[0] == 2

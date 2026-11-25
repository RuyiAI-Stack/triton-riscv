import torch


_SOBOL_DIRECTION = [
    [1],
    [1, 3],
    [1, 3, 1],
    [1, 1, 1],
    [1, 1, 3, 3],
    [1, 3, 5, 13],
    [1, 1, 5, 5],
    [1, 1, 5, 5, 17],
]


def _sobol_engine_initialize_state_(dimension):
    state = torch.zeros(dimension, len(_SOBOL_DIRECTION), dtype=torch.int32)
    for dim, bits in enumerate(_SOBOL_DIRECTION[:dimension]):
        for bit, value in enumerate(bits):
            state[dim, bit] = value
    return state

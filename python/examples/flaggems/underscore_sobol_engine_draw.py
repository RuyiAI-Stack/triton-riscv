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


def _sobol_point(index, dimension):
    point = torch.zeros(dimension, dtype=torch.float64)
    for dim in range(dimension):
        bits = _SOBOL_DIRECTION[dim]
        value = 0
        i = index
        bit = 0
        while i and bit < len(bits):
            if i & 1:
                value ^= bits[bit] << (32 - (bit + 1))
            i >>= 1
            bit += 1
        point[dim] = value / 2**32
    return point


def underscore_sobol_engine_draw(quasi, n, sobolstate, dimension, num_generated):
    _ = (quasi, sobolstate)
    out = torch.stack([_sobol_point(num_generated + i, dimension) for i in range(n)])
    return out.to(torch.float32), out.to(torch.float32)

from .binomial import binomial


def test_binomial():
    draws = binomial(4, 0.3, (16,))
    assert draws.shape == (16,) and int(draws.min()) >= 0 and int(draws.max()) <= 4

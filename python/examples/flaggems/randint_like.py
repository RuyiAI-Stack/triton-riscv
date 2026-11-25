from .randint import randint


def randint_like(self, low, high=None, *, dtype=None, device=None):
    return randint(
        low,
        high,
        tuple(self.shape),
        dtype=self.dtype if dtype is None else dtype,
        device=self.device if device is None else device,
    )

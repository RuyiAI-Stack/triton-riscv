import torch

from .detach_copy import _copy_values


def _batched_mm(a, b):
    from .mm import mm

    if a.ndim == 2 and b.ndim == 2:
        return mm(a, b)
    if b.ndim == 2:
        b = b.expand(a.shape[0], *b.shape)
    if a.ndim == 2:
        a = a.expand(b.shape[0], *a.shape)
    batch = a.reshape(-1, a.shape[-2], a.shape[-1])
    rhs = b.reshape(-1, b.shape[-2], b.shape[-1])
    if rhs.shape[0] == 1 and batch.shape[0] != 1:
        rhs = rhs.expand(batch.shape[0], -1, -1)
    if batch.shape[0] == 1 and rhs.shape[0] != 1:
        batch = batch.expand(rhs.shape[0], -1, -1)
    out = torch.empty(
        batch.shape[0],
        batch.shape[1],
        rhs.shape[-1],
        dtype=a.dtype,
        device=a.device,
    )
    for i in range(batch.shape[0]):
        out[i] = mm(batch[i], rhs[i])
    return out.reshape(a.shape[:-1] + (rhs.shape[-1],))


def _sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def _tanh(x):
    return torch.tanh(x)


def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    h, c = hx
    gates = _batched_mm(input, w_ih.transpose(0, 1)) + _batched_mm(
        h, w_hh.transpose(0, 1)
    )
    if b_ih is not None:
        gates = gates + b_ih
    if b_hh is not None:
        gates = gates + b_hh
    ing, forget, cell, out = gates.chunk(4, dim=-1)
    c_new = _sigmoid(forget) * c + _sigmoid(ing) * _tanh(cell)
    h_new = _sigmoid(out) * _tanh(c_new)
    return _copy_values(h_new), _copy_values(c_new)

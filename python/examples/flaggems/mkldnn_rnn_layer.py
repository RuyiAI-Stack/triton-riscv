import torch

from .rnn_relu_cell import rnn_relu_cell
from .rnn_tanh_cell import rnn_tanh_cell


def mkldnn_rnn_layer(
    input, hx, params, has_biases, reverse=False, batch_sizes=None, mode="tanh"
):
    _ = batch_sizes
    cell = rnn_relu_cell if mode == "relu" else rnn_tanh_cell
    seq = torch.flip(input, (0,)) if reverse else input
    state = hx[0] if hx.ndim == 3 else hx
    steps = []
    for t in range(seq.shape[0]):
        state = cell(
            seq[t],
            state,
            params[0],
            params[1],
            params[2] if has_biases else None,
            params[3] if has_biases else None,
        )
        steps.append(state)
    output = torch.stack(steps)
    if reverse:
        output = torch.flip(output, (0,))
    return output, state.unsqueeze(0)

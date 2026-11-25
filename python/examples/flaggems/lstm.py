import torch

from .lstm_cell import lstm_cell


def _split_params(params, has_biases, num_layers, bidirectional):
    directions = 2 if bidirectional else 1
    grouped = []
    cursor = 0
    per_dir = 4 if has_biases else 2
    for _layer in range(num_layers):
        for _direction in range(directions):
            grouped.append(params[cursor : cursor + per_dir])
            cursor += per_dir
    return grouped


def lstm(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    h0, c0 = hx
    seq = input.transpose(0, 1) if batch_first else input
    groups = _split_params(params, has_biases, num_layers, bidirectional)
    layer_input = seq
    hy = []
    cy = []
    index = 0
    for layer in range(num_layers):
        weights = groups[index]
        index += 1
        bias_ih = weights[2] if has_biases else None
        bias_hh = weights[3] if has_biases else None
        h = h0[layer]
        c = c0[layer]
        steps = []
        for t in range(layer_input.shape[0]):
            h, c = lstm_cell(
                layer_input[t], (h, c), weights[0], weights[1], bias_ih, bias_hh
            )
            steps.append(h)
        layer_input = torch.stack(steps)
        hy.append(h)
        cy.append(c)
    output = layer_input.transpose(0, 1) if batch_first else layer_input
    return output, (torch.stack(hy), torch.stack(cy))

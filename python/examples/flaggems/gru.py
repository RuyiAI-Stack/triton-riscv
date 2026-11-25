import torch

from .gru_cell import gru_cell


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


def _run_rnn(
    cell,
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
    _ = (dropout, train)
    seq = input.transpose(0, 1) if batch_first else input
    directions = 2 if bidirectional else 1
    groups = _split_params(params, has_biases, num_layers, bidirectional)
    h = hx
    layer_input = seq
    outputs_h = []
    group_index = 0
    for _layer in range(num_layers):
        dir_outputs = []
        dir_h = []
        for direction in range(directions):
            weights = groups[group_index]
            group_index += 1
            bias_ih = weights[2] if has_biases else None
            bias_hh = weights[3] if has_biases else None
            state = h[direction + _layer * directions]
            steps = []
            times = range(layer_input.shape[0])
            if direction == 1:
                times = reversed(list(times))
            ordered = []
            running = state
            for t in times:
                running = cell(
                    layer_input[t], running, weights[0], weights[1], bias_ih, bias_hh
                )
                ordered.append((t, running))
            ordered.sort(key=lambda item: item[0])
            steps = torch.stack([item[1] for item in ordered])
            dir_outputs.append(steps)
            dir_h.append(running)
        if directions == 2:
            layer_input = torch.cat(dir_outputs, dim=-1)
        else:
            layer_input = dir_outputs[0]
        outputs_h.extend(dir_h)
    output = layer_input.transpose(0, 1) if batch_first else layer_input
    return output, torch.stack(outputs_h)


def gru(
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
    return _run_rnn(
        gru_cell,
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )

import torch

from .rnn_tanh import rnn_tanh


def test_rnn_tanh():
    torch.manual_seed(0)
    layer = torch.nn.RNN(3, 4, nonlinearity="tanh")
    seq = torch.randn(2, 2, 3)
    hidden = torch.zeros(1, 2, 4)
    params = [
        layer.weight_ih_l0,
        layer.weight_hh_l0,
        layer.bias_ih_l0,
        layer.bias_hh_l0,
    ]
    out, hy = rnn_tanh(seq, hidden, params, True, 1, 0.0, False, False, False)
    ref_out, ref_hy = layer(seq, hidden)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hy, ref_hy, rtol=1e-4, atol=1e-4)

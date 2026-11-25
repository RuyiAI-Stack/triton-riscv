import torch

from .lstm_cell import lstm_cell


def test_lstm_cell():
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(3, 4)
    inputs = torch.randn(2, 3)
    hidden = (torch.randn(2, 4), torch.randn(2, 4))
    got_h, got_c = lstm_cell(
        inputs, hidden, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh
    )
    ref_h, ref_c = cell(inputs, hidden)
    torch.testing.assert_close(got_h, ref_h, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_c, ref_c, rtol=1e-4, atol=1e-4)

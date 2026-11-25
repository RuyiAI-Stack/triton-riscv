import torch

from ._thnn_fused_gru_cell import _thnn_fused_gru_cell


def test__thnn_fused_gru_cell():
    torch.manual_seed(0)
    cell = torch.nn.GRUCell(3, 4)
    inputs = torch.randn(2, 3)
    hidden = torch.randn(2, 4)
    torch.testing.assert_close(
        _thnn_fused_gru_cell(
            inputs, hidden, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh
        ),
        cell(inputs, hidden),
        rtol=1e-4,
        atol=1e-4,
    )

from .lstm_cell import lstm_cell


def _thnn_fused_lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    return lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh)

from .gru_cell import gru_cell


def _thnn_fused_gru_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    return gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh)

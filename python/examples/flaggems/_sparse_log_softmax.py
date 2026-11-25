import torch


def _sparse_log_softmax(self, dim=0):
    dense = self.to_dense()
    return torch.log_softmax(dense, dim=dim).to_sparse()

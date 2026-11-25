import torch


def _sparse_softmax(self, dim=0):
    dense = self.to_dense()
    result = torch.softmax(dense, dim=dim)
    return result.to_sparse()

def sparse_broadcast_to(self, size):
    return self.to_dense().broadcast_to(size).to_sparse()

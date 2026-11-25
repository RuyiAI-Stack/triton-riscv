def sparse_resize_(self, size, sparse_dim, dense_dim):
    _ = (sparse_dim, dense_dim)
    self.sparse_resize_(size, self.sparse_dim(), self.dense_dim())
    return self

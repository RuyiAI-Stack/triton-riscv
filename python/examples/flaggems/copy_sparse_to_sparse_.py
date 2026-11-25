def copy_sparse_to_sparse_(self, src, non_blocking=False):
    _ = non_blocking
    self.copy_(src)
    return self

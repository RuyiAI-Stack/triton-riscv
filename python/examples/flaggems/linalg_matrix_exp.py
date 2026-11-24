import torch


def _batched_mm(a, b):
    from .mm import mm

    if a.ndim == 2 and b.ndim == 2:
        return mm(a, b)
    if b.ndim == 2:
        b = b.expand(a.shape[0], *b.shape)
    if a.ndim == 2:
        a = a.expand(b.shape[0], *a.shape)
    batch = a.reshape(-1, a.shape[-2], a.shape[-1])
    rhs = b.reshape(-1, b.shape[-2], b.shape[-1])
    if rhs.shape[0] == 1 and batch.shape[0] != 1:
        rhs = rhs.expand(batch.shape[0], -1, -1)
    if batch.shape[0] == 1 and rhs.shape[0] != 1:
        batch = batch.expand(rhs.shape[0], -1, -1)
    out = torch.empty(
        batch.shape[0],
        batch.shape[1],
        rhs.shape[-1],
        dtype=a.dtype,
        device=a.device,
    )
    for i in range(batch.shape[0]):
        out[i] = mm(batch[i], rhs[i])
    return out.reshape(a.shape[:-1] + (rhs.shape[-1],))


def _eye_like_batch(A):
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    return eye.expand(A.shape).contiguous()


def linalg_matrix_exp(A):
    fro = torch.linalg.matrix_norm(A, ord="fro")
    scale = torch.ceil(torch.log2(fro.clamp(min=1.0))).to(torch.int32)
    steps = int(scale.max().item()) if scale.numel() else 0
    mat = A / float(2**steps)
    term = _eye_like_batch(mat)
    acc = term.clone()
    for k in range(1, 12):
        term = _batched_mm(term, mat) / float(k)
        acc = acc + term
    for _ in range(steps):
        acc = _batched_mm(acc, acc)
    return acc

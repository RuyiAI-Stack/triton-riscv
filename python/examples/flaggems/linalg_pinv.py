import torch

from .linalg_eigh import linalg_eigh


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


def linalg_pinv(A, *, atol=None, rtol=None, hermitian=False):
    _ = hermitian
    if A.shape[-1] > A.shape[-2]:
        return linalg_pinv(A.transpose(-1, -2), atol=atol, rtol=rtol).transpose(-1, -2)
    gram = _batched_mm(A.transpose(-1, -2), A)
    evals, vecs = linalg_eigh(gram)
    evals = evals.clamp(min=0)
    sv = torch.sqrt(evals)
    if rtol is None and atol is None:
        tol = sv[..., -1] * max(A.shape[-2:]) * torch.finfo(A.dtype).eps
    else:
        tol = torch.zeros(sv.shape[:-1], dtype=sv.dtype, device=sv.device)
        if rtol is not None:
            tol = tol + rtol * sv[..., -1]
        if atol is not None:
            tol = tol + atol
    inv_eig = torch.where(
        sv > tol.unsqueeze(-1), 1.0 / evals.clamp(min=1e-20), torch.zeros_like(evals)
    )
    scaled = vecs * inv_eig.unsqueeze(-2)
    return _batched_mm(_batched_mm(scaled, vecs.transpose(-1, -2)), A.transpose(-1, -2))


__all__ = ["linalg_pinv"]

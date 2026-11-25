"""Independent PyTorch and analytic references for B13 CPU examples."""

import importlib
import torch
import torch.nn.functional as F


def _lanczos_reference(x, size, align):
    def weights(n, m):
        scale = (n - 1) / (m - 1) if align and m > 1 else n / m
        distance = (
            torch.arange(n)[None, :] + 0.5 - scale * (torch.arange(m)[:, None] + 0.5)
        ) / max(scale, 1)
        w = torch.sinc(distance) * torch.sinc(distance / 3)
        w = torch.where(distance.abs() < 3, w, 0.0)
        return w / w.sum(1, keepdim=True)

    h, w = weights(x.shape[-2], size[0]), weights(x.shape[-1], size[1])
    return torch.einsum("ah,nchw,bw->ncab", h, x, w)


def run_case(name, variant):
    torch.manual_seed(2300 + variant)
    package = importlib.import_module(__package__)
    alias = {
        "embedding_bag_backward": "_embedding_bag_backward",
        "upsample_lanczos2d_aa": "_upsample_lanczos2d_aa",
        "_upsample_lanczos2d_aa_backward": "upsample_lanczos2d_aa_backward",
    }
    export_name = alias.get(name, name)
    assert export_name in package.__all__, f"{export_name} is missing from __all__"
    fn = getattr(package, export_name)
    align = bool(variant)
    if name in {
        "affine_grid_generator_backward",
        "cudnn_affine_grid_generator_backward",
    }:
        size = (
            (2, 3, 4, 5) if not variant or name.startswith("cudnn") else (2, 3, 3, 4, 5)
        )
        theta = torch.randn(2, len(size) - 2, len(size) - 1, requires_grad=True)
        al = True if name.startswith("cudnn") else align
        grid = F.affine_grid(theta, size, al)
        grad = torch.randn_like(grid)
        ref = torch.autograd.grad(grid, theta, grad)[0]
        actual = fn(grad, *size) if name.startswith("cudnn") else fn(grad, size, al)
        return actual, ref
    if name == "as_strided_scatter":
        x = torch.randn(20)
        src = torch.randn(2, 3)
        args = (x, src, [2, 3], [5, 1], variant + 1)
        return fn(*args), torch.as_strided_scatter(*args)
    if name.startswith("batch_norm_gather_stats"):
        counts = [5, 5, 5] if name == "batch_norm_gather_stats" else [3, 5, 7]
        shards = [torch.randn(n, 4) + i for i, n in enumerate(counts)]
        means = torch.stack([s.mean(0) for s in shards])
        eps = 1e-5
        inv = torch.stack([torch.rsqrt(s.var(0, unbiased=False) + eps) for s in shards])
        full = torch.cat(shards)
        rm = torch.randn(4)
        rv = torch.rand(4) + 0.5
        ref_mean = full.mean(0)
        ref_inv = torch.rsqrt(full.var(0, unbiased=False) + eps)
        ref_rm = rm * 0.8 + ref_mean * 0.2
        ref_rv = rv * 0.8 + full.var(0, unbiased=True) * 0.2
        ct = (
            5
            if name == "batch_norm_gather_stats"
            else torch.tensor(counts, dtype=torch.float32)
        )
        actual = fn(
            full.T,
            means,
            inv,
            rm if variant else None,
            rv if variant else None,
            0.2,
            eps,
            ct,
        )
        return (actual, rm, rv), (
            (ref_mean, ref_inv),
            ref_rm if variant else rm,
            ref_rv if variant else rv,
        )
    if name == "cosine_embedding_loss":
        x, y = torch.randn(7, 5), torch.randn(7, 5)
        target = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        red = "sum" if variant else "none"
        return fn(x, y, target, 0.3, red), F.cosine_embedding_loss(
            x, y, target, margin=0.3, reduction=red
        )
    if name == "hinge_embedding_loss":
        x = torch.randn(3, 7)
        target = torch.where(x > 0, 1.0, -1.0)
        red = "mean" if variant else "none"
        return fn(x, target, 0.7, red), F.hinge_embedding_loss(
            x, target, margin=0.7, reduction=red
        )
    if name == "diagonal_scatter":
        x = torch.randn(3, 4, 5)
        src = torch.randn(3, 4 if variant else 3)
        offset = 0 if variant else 1
        # Explicit dimensions keep the untouched leading dimension.
        src = torch.randn_like(x.diagonal(offset, 1, 2))
        return fn(x, src, offset, 1, 2), torch.diagonal_scatter(x, src, offset, 1, 2)
    if name.startswith("grid_sampler") or name in {
        "cudnn_grid_sampler",
        "cudnn_grid_sampler_backward",
    }:
        d = 3 if "3d" in name or (name == "grid_sampler" and variant) else 2
        x = torch.randn((2, 2) + ((3, 4, 5) if d == 3 else (4, 5)), requires_grad=True)
        grid = (
            torch.rand((2,) + ((2, 3, 4) if d == 3 else (3, 4)) + (d,)) * 2.6 - 1.3
        ).requires_grad_()
        cudnn = name.startswith("cudnn")
        al = True if cudnn else align
        padding = "zeros" if cudnn or not variant else "reflection"
        mode = "bilinear"
        ref = F.grid_sample(x, grid, mode=mode, padding_mode=padding, align_corners=al)
        if name.endswith("backward"):
            go = torch.randn_like(ref)
            expected = torch.autograd.grad(ref, (x, grid), go)
            actual = (
                fn(x, grid, go)
                if cudnn
                else fn(go, x, grid, 0, 2 if variant else 0, al, [True, True])
            )
        else:
            actual = fn(x, grid) if cudnn else fn(x, grid, 0, 2 if variant else 0, al)
            expected = ref
        return actual, expected
    if name == "index_fill":
        x = torch.randn(3, 5)
        idx = torch.tensor([0, 2])
        val = torch.tensor(0.25) if variant else 0.25
        return fn(x, variant, idx, val), torch.index_fill(x, variant, idx, val)
    if name == "index_reduce":
        x = torch.randn(4, 5)
        idx = torch.tensor([2, 0, 2])
        src = torch.randn(3, 5)
        red = "mean" if variant else "amax"
        return fn(x, 0, idx, src, red, include_self=bool(variant)), torch.index_reduce(
            x, 0, idx, src, red, include_self=bool(variant)
        )
    if name == "index_select_backward":
        x = torch.randn(3, 5, requires_grad=True)
        idx = torch.tensor([2, 0, 2])
        y = torch.index_select(x, variant, idx)
        go = torch.randn_like(y)
        return fn(go, x.shape, variant, idx), torch.autograd.grad(y, x, go)[0]
    if name == "masked_scatter_backward":
        grad = torch.randn(3, 5)
        mask = grad > 0
        sizes = [mask.sum().item() + variant + 2]
        return fn(grad, mask, sizes), torch.ops.aten.masked_scatter_backward(
            grad, mask, sizes
        )
    if name in {"put", "put_"}:
        x = torch.randn(3, 5)
        idx = torch.tensor([0, 3, 7, 3] if variant else [0, 3, 7, 9])
        src = torch.randn(4)
        expected = x.clone().put_(idx, src, accumulate=bool(variant))
        actual = fn(x, idx, src, accumulate=bool(variant))
        if name == "put_":
            assert actual.data_ptr() == x.data_ptr()
        return actual, expected
    if name == "take":
        x = torch.randn(3, 5).T if variant else torch.randn(3, 5)
        idx = torch.tensor([0, -1, 5, 7])
        return fn(x, idx), torch.take(x, idx)
    if name == "take_along_dim":
        x = torch.randn(3, 5)
        idx = torch.tensor([[0, 3, 2]]) if variant else torch.tensor([0, 3, 7])
        dim = 1 if variant else None
        return fn(x, idx, dim), torch.take_along_dim(x, idx, dim)
    if "upsample" in name:
        d = 1 if "1d" in name else 3 if "3d" in name else 2
        ins = (2, 2) + {1: (5,), 2: (5, 7), 3: (3, 4, 5)}[d]
        size = {1: (8,), 2: ((3, 4) if variant else (7, 9)), 3: (4, 6, 7)}[d]
        x = torch.randn(ins, requires_grad=True)
        lanczos = "lanczos" in name
        nearest = "nearest" in name
        mode = (
            ("nearest-exact" if "exact" in name else "nearest")
            if nearest
            else (
                "bicubic"
                if "bicubic" in name
                else {1: "linear", 2: "bilinear", 3: "trilinear"}[d]
            )
        )
        ref = (
            _lanczos_reference(x, size, align)
            if lanczos
            else F.interpolate(
                x,
                size=size,
                mode=mode,
                **({} if nearest else {"align_corners": align}),
                **({"antialias": True} if "_aa" in name else {}),
            )
        )
        if "backward" in name:
            go = torch.randn_like(ref)
            args = [go, size, ins] + ([] if nearest else [align])
            return fn(*args), torch.autograd.grad(ref, x, go)[0]
        return fn(x, size, *([] if nearest else [align])), ref
    if name == "embedding_renorm_":
        w = torch.randn(8, 5)
        idx = torch.tensor([1, 3, 1, 5])
        p = 1.0 if variant else 2.0
        expected = torch.embedding_renorm_(w.clone(), idx, 1.2, p)
        actual = fn(w, idx, 1.2, p)
        assert actual.data_ptr() == w.data_ptr()
        return actual, expected
    if name == "embedding_sparse_backward":
        idx = torch.tensor([[1, 2, 1], [3, 0, 5]])
        go = torch.randn(2, 3, 5)
        args = (go, idx, 7, 0 if variant else -1, False)
        return (
            fn(*args).to_dense(),
            torch.ops.aten.embedding_sparse_backward(*args).to_dense(),
        )
    if name == "_gather_sparse_backward":
        x = torch.randn(3, 5)
        idx = (
            torch.tensor([[1, 2, 1], [0, 2, 0]])
            if not variant
            else torch.tensor([[1, 2], [3, 1], [4, 0]])
        )
        go = torch.randn_like(torch.gather(x, variant, idx))
        args = (x, variant, idx, go)
        return (
            fn(*args).to_dense(),
            torch.ops.aten._gather_sparse_backward(*args).to_dense(),
        )
    if name in {"embedding_spmdm", "embedding_spmdm_grad"}:
        w = torch.randn(8, 5, requires_grad=True)
        idx = torch.tensor([[1, 2, 1], [3, 0, 5]])
        ps = torch.rand(2, 3, requires_grad=True) if variant else None
        ref = (w[idx] * (ps[..., None] if ps is not None else 1)).sum(1)
        if name.endswith("grad"):
            go = torch.randn_like(ref)
            grads = torch.autograd.grad(ref, (w, ps) if ps is not None else (w,), go)
            return fn(go, w, idx, ps), (grads[0], grads[1] if ps is not None else None)
        return fn(w, idx, ps), ref
    if "embedding_bag" in name:
        w = torch.randn(7, 5, requires_grad=True)
        idx = torch.tensor([1, 2, 1, 0, 3, 5])
        off = torch.tensor([0, 3])
        mode = variant
        ps = (
            torch.rand(6, requires_grad=True)
            if name == "_embedding_bag_per_sample_weights_backward"
            else None
        )
        if ps is not None:
            mode = 0
        output, offset2bag, bag_size, maximum = torch.ops.aten._embedding_bag(
            w, idx, off, False, mode, False, ps, False, -1
        )
        if name == "embedding_bag":
            # CPU ATen leaves offset2bag empty for sum; compare the meaningful output and sizes.
            actual = fn(w, idx, off, False, mode, False, ps, False)
            return (actual[0], actual[2]), (output, bag_size)
        if offset2bag.numel() == 0:
            offset2bag = torch.tensor([0, 0, 0, 1, 1, 1])
        if mode == 0:
            bag_size = torch.tensor([3, 3])
        grad = torch.randn_like(output)
        if name == "_embedding_bag_per_sample_weights_backward":
            args = (grad, w, idx, off, offset2bag, mode, -1)
        elif name == "_embedding_bag_dense_backward":
            args = (grad, idx, offset2bag, bag_size, maximum, 7, False, mode, ps, -1)
        elif name == "_embedding_bag_sparse_backward":
            args = (grad, idx, off, offset2bag, bag_size, 7, False, mode, ps, -1)
        else:
            args = (
                grad,
                idx,
                off,
                offset2bag,
                bag_size,
                maximum,
                7,
                False,
                mode,
                bool(variant),
                ps,
                -1,
            )
        actual = fn(*args)
        expected = getattr(torch.ops.aten, alias.get(name, name))(*args)
        return (actual.to_dense() if actual.is_sparse else actual), (
            expected.to_dense() if expected.is_sparse else expected
        )
    if name == "gather_block_quantized":
        q = torch.randint(-50, 50, (37,), dtype=torch.int8)
        scales = torch.rand(5)
        idx = torch.tensor([0, 7, 19, 36]) if variant else None
        ref = q.float() * scales[torch.arange(37) // 8]
        return fn(q, scales, idx, 8), ref if idx is None else ref[idx]
    raise AssertionError("Missing reference for " + name)

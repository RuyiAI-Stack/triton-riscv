"""PyTorch/formula references for the B12 CPU migration."""

import importlib

import torch


def run_case(name, variant):
    torch.manual_seed(1300 + variant)
    mod = importlib.import_module("." + name, __package__)
    aliases = {
        "cartesian_prod": "run",
        "histogram": "histogram_bin_ct",
        "trapezoid": "trapezoid_x",
        "_batch_norm_impl_index": "batch_norm_impl_index",
        "_weight_norm_differentiable_backward": "weight_norm_differentiable_backward",
        "te_rmsnorm": "te_rmsnorm_fwd",
    }
    fn = getattr(mod, aliases.get(name, name))
    eps = 1e-5
    if (
        name.startswith("batch_norm_")
        or name == "cudnn_batch_norm_backward"
        or name.startswith("_batch_norm")
        or name.startswith("_native_batch_norm")
    ):
        x = torch.randn(2, 3, 5 + variant)
        w, b = torch.randn(3), torch.randn(3)
        mean = x.mean((0, 2))
        var = x.var((0, 2), unbiased=False)
        inv = torch.rsqrt(var + eps)
        rm, rv = torch.randn(3), torch.rand(3) + 0.5
        go = torch.randn_like(x)
        n = x.numel() // 3
        sumdy = go.sum((0, 2))
        sumdx = (go * (x - mean[None, :, None])).sum((0, 2))
        if name == "batch_norm_stats":
            return fn(x, eps), (mean, inv)
        if name == "batch_norm_elemt":
            return (
                fn(x, w, b, mean, inv, eps),
                (x - mean[None, :, None]) * inv[None, :, None] * w[None, :, None]
                + b[None, :, None],
            )
        if name == "batch_norm_backward_reduce":
            return fn(go, x, mean, inv, w, True, True, True), (
                sumdy,
                sumdx,
                sumdx * inv,
                sumdy,
            )
        dx = (
            w[None, :, None]
            * inv[None, :, None]
            * (
                go
                - (
                    sumdy[None, :, None]
                    + (x - mean[None, :, None])
                    * inv[None, :, None] ** 2
                    * sumdx[None, :, None]
                )
                / n
            )
        )
        if name == "batch_norm_backward_elemt":
            return (
                fn(
                    go,
                    x,
                    mean,
                    inv,
                    w,
                    sumdy,
                    sumdx,
                    torch.tensor([n], dtype=torch.int32),
                ),
                dx,
            )
        if name == "cudnn_batch_norm_backward":
            return fn(x, go, w, rm, rv, mean, inv, eps), (dx, sumdx * inv, sumdy)
        if name == "batch_norm_update_stats":
            old_rm, old_rv = rm.clone(), rv.clone()
            actual = fn(x, rm, rv, 0.2)
            expected = torch.batch_norm_update_stats(x, old_rm, old_rv, 0.2)
            return (actual, rm, rv), (expected, old_rm, old_rv)
        args = [x, w, b, rm.clone(), rv.clone()]
        refargs = [x, w, b, rm.clone(), rv.clone()]
        if name == "_batch_norm_impl_index_backward":
            actual = fn(
                0,
                x,
                go,
                w,
                rm,
                rv,
                mean,
                inv,
                True,
                eps,
                [True] * 3,
                torch.empty(0, dtype=torch.uint8),
            )
            return actual, (dx, sumdx * inv, sumdy)
        if name in {"_batch_norm_impl_index", "_native_batch_norm_legit_functional"}:
            args += [True, 0.2, eps]
            refargs += [True, 0.2, eps]
        else:
            args += [0.2, eps]
            refargs += [0.2, eps]
        if name == "_batch_norm_impl_index":
            args.append(True)
            refargs.append(True)
        ref = getattr(torch.ops.aten, name)(*refargs)
        actual = fn(*args)
        return (actual, args[3], args[4]), (ref, refargs[3], refargs[4])
    if name == "cartesian_prod":
        args = (torch.arange(3), torch.arange(2 + variant), torch.tensor([4, 7]))
        return fn(*args), torch.cartesian_prod(*args)
    if name == "combinations":
        x = torch.arange(5)
        return fn(x, 2, bool(variant)), torch.combinations(x, 2, bool(variant))
    if name in {"cdist", "pdist", "dist"}:
        x = torch.randn(4, 5 + variant)
        y = torch.randn(3, 5 + variant)
        p = 1.0 if variant else 2.0
        if name == "cdist":
            return fn(x, y, p), torch.cdist(x, y, p)
        if name == "pdist":
            return fn(x, p), torch.pdist(x, p)
        return fn(x, y[:1], p), torch.dist(x, y[:1], p)
    if name in {"corrcoef", "cov"}:
        x = torch.randn(3, 11 + variant)
        return fn(x), getattr(torch, name)(x)
    if name in {"cumulative_trapezoid", "trapz", "trapezoid"}:
        y = torch.randn(3, 7 + variant)
        if name == "trapezoid":
            x = torch.linspace(0, 3, y.shape[1]) ** 2
            return fn(y, x, dim=1), torch.trapezoid(y, x, dim=1)
        return fn(y, dx=0.7, dim=1), getattr(torch, name)(y, dx=0.7, dim=1)
    if name == "histogram":
        x = torch.randn(31)
        w = torch.rand(31) if variant else None
        return fn(
            x, 5, range=(-3, 3), weight=w, density=bool(variant)
        ), torch.histogram(x, 5, range=(-3, 3), weight=w, density=bool(variant))
    if name in {
        "histogramdd",
        "_histogramdd_from_bin_cts",
        "_histogramdd_from_bin_tensors",
    }:
        x = torch.randn(17, 2)
        w = torch.rand(17) if variant else None
        bins = [torch.linspace(-3, 3, 4), torch.linspace(-2, 2, 5)]
        ref = torch.histogramdd(x, bins=bins, weight=w, density=bool(variant))
        if name == "histogramdd":
            return fn(x, bins, weight=w, density=bool(variant)), ref
        if name == "_histogramdd_from_bin_cts":
            return (
                fn(x, [3, 4], range=[-3, 3, -2, 2], weight=w, density=bool(variant)),
                ref[0],
            )
        return fn(x, bins, weight=w, density=bool(variant)), ref[0]
    if name in {"linalg_norm", "native_norm", "norm_except_dim"}:
        x = torch.randn(3, 5 + variant)
        if name == "linalg_norm":
            return fn(x, ord=2, dim=1), torch.linalg.vector_norm(x, ord=2, dim=1)
        if name == "native_norm":
            return fn(x, 2), torch.norm(x, 2)
        return fn(x, 2, variant), torch.norm_except_dim(x, 2, variant)
    if name in {"logcumsumexp", "_logcumsumexp"}:
        x = torch.randn(3, 11)
        return fn(x, variant), torch.logcumsumexp(x, variant)
    if name == "mode":
        x = torch.tensor([[3.0, 1.0, 3.0, 2.0], [2.0, 2.0, 4.0, 1.0]])
        return fn(x, dim=1, keepdim=bool(variant)), torch.mode(
            x, dim=1, keepdim=bool(variant)
        )
    if name in {"nanmedian", "nanquantile"}:
        x = torch.randn(3, 9)
        x[0, 2] = float("nan")
        if name == "nanmedian":
            return fn(x), torch.nanmedian(x)
        return fn(x, 0.25 + variant * 0.5, dim=1), torch.nanquantile(
            x, 0.25 + variant * 0.5, dim=1
        )
    if name == "native_group_norm":
        x = torch.randn(2, 4, 5 + variant)
        w = torch.randn(4)
        b = torch.randn(4)
        args = (x, w, b, 2, 4, 5 + variant, 2, eps)
        return fn(*args), torch.native_group_norm(*args)
    if name == "quantized_batch_norm":
        x = torch.quantize_per_tensor(torch.randn(2, 3, 4, 5), 0.05, 7, torch.quint8)
        w, b, m, v = torch.randn(3), torch.randn(3), torch.randn(3), torch.rand(3) + 0.5
        ref = (x.dequantize() - m[None, :, None, None]) * torch.rsqrt(v + eps)[
            None, :, None, None
        ] * w[None, :, None, None] + b[None, :, None, None]
        return fn(x, w, b, m, v, eps, 0.1, 11), torch.quantize_per_tensor(
            ref, 0.1, 11, torch.quint8
        )
    if name in {"renorm", "renorm_"}:
        x = torch.randn(4, 7)
        ref = torch.renorm(x, 2, variant, 0.8)
        return fn(x, 2, variant, 0.8), ref
    if name == "segment_reduce":
        x = torch.randn(7, 3)
        lengths = torch.tensor([2, 0, 3, 2])
        reduce = "sum" if variant else "mean"
        return fn(x, reduce, lengths=lengths), torch.segment_reduce(
            x, reduce, lengths=lengths
        )
    if name == "sum_to_size":
        x = torch.randn(2, 3, 4)
        shape = (1, 3, 1) if variant else (3, 4)
        return fn(x, shape), x.sum_to_size(shape)
    if name == "vander":
        x = torch.tensor([-2.0, 0.0, 1.0, 3.0])
        return fn(x, 5, bool(variant)), torch.vander(x, 5, bool(variant))
    if name == "_aminmax":
        x = torch.randn(3, 7)
        return fn(x), torch.aminmax(x)
    if name == "_is_any_true":
        x = torch.tensor([False, bool(variant), False])
        return fn(x), torch.any(x)
    if name in {
        "_fused_rms_norm",
        "gemma_rms_norm",
        "rms_norm_w8a16_fp8",
        "te_rmsnorm",
    }:
        x = torch.randn(3, 16)
        w = torch.randn(16)
        inv = torch.rsqrt(x.square().mean(-1) + eps)
        if name == "gemma_rms_norm":
            return fn(x, w, eps), x * inv[:, None] * (1 + w)
        if name == "_fused_rms_norm":
            actual = fn(x, [16], w, eps)
            return actual, (x * inv[:, None] * w, inv.reshape(actual[1].shape))
        if name == "te_rmsnorm":
            actual = fn(x, w, eps, zero_centered_gamma=bool(variant))
            return actual, (
                x * inv[:, None] * (w + variant),
                None,
                inv.reshape(actual[2].shape),
            )
        qw = w.to(torch.float8_e4m3fn)
        scales = torch.ones(2) * 0.5
        return (
            fn(x, [16], qw, scales, eps, group_size=8),
            x * inv[:, None] * qw.float() * 0.5,
        )
    if name in {"_weight_norm", "_weight_norm_differentiable_backward"}:
        v = torch.randn(3, 4)
        dim = 0 if variant == 0 else 1
        norms = torch.linalg.vector_norm(v, dim=1 - dim, keepdim=True)
        g = torch.randn_like(norms)
        if name == "_weight_norm":
            return fn(v, g, dim), v * g / norms
        go = torch.randn_like(v)
        vv = v.clone().requires_grad_()
        gg = g.clone().requires_grad_()
        y = vv * gg / torch.linalg.vector_norm(vv, dim=1 - dim, keepdim=True)
        return fn(go, v, g, norms, dim), torch.autograd.grad(y, (vv, gg), go)
    if name == "_nested_sum_backward":
        x = torch.randn(2, 3, 4)
        dim = 1 + variant
        go = torch.randn_like(x.sum(dim))
        return fn(go, x, [dim]), go.unsqueeze(dim).expand_as(x).contiguous()
    if name == "sparse_csr_sum":
        x = torch.tensor(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
        ).to_sparse_csr()
        actual = fn(x, [variant], keepdim=True)
        return actual.to_dense(), x.to_dense().sum(variant, keepdim=True)
    if name == "_sparse_sum_backward":
        x = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]).to_sparse().coalesce()
        go = torch.randn(())
        actual = fn(go, x, [0, 1])
        expected = torch.sparse_coo_tensor(
            x.indices(), torch.full_like(x.values(), go.item()), x.shape
        )
        return actual.to_dense(), expected.to_dense()
    if name == "_philox_normal_":
        from ._port_rng_reference import normal

        x = torch.empty(257)
        key = torch.tensor([123 + variant, 7], dtype=torch.uint64)
        return fn(x, key, 0.3, 0.7), normal(257, 123 + variant, 7, 0.3, 0.7)
    if name == "_unsafe_masked_index_put_accumulate":
        x = torch.randn(7)
        idx = torch.tensor([1, 2, 1, 4])
        mask = torch.tensor([True, False, True, True])
        values = torch.randn(4)
        ref = x.clone()
        ref.index_add_(0, idx[mask], values[mask])
        return fn(x, mask, [idx], values), ref
    raise AssertionError(f"No reference case registered for {name}")

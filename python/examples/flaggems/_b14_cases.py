"""Independent PyTorch references for B14 CPU operator examples."""

import importlib

import pytest
import torch
import torch.nn.functional as F


def run_case(name, variant):
    torch.manual_seed(1200 + variant)
    package = importlib.import_module(__package__)
    aliases = {
        "adaptive_avg_pool3d_backward": "_adaptive_avg_pool3d_backward",
        "convert_indices_from_coo_to_csr": "_convert_indices_from_coo_to_csr",
        "convert_indices_from_csr_to_coo": "_convert_indices_from_csr_to_coo",
    }
    export_name = aliases.get(name, name)
    assert export_name in package.__all__, f"{export_name} is missing from __all__"
    fn = getattr(package, export_name)
    size = 5 + variant
    if name in {
        "adaptive_avg_pool2d",
        "_adaptive_avg_pool3d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
    }:
        rank = 3 if "3d" in name else 2
        x = torch.randn((2, 2) + (size,) * rank)
        output = (2 + variant,) * rank
        if "max" in name:
            ref = getattr(F, f"adaptive_max_pool{rank}d")(
                x, output, return_indices=True
            )
        else:
            ref = getattr(F, f"adaptive_avg_pool{rank}d")(x, output)
        return fn(x, output), ref
    if name in {
        "adaptive_avg_pool3d_backward",
        "_adaptive_avg_pool2d_backward",
        "adaptive_max_pool2d_backward",
        "adaptive_max_pool3d_backward",
    }:
        rank = 3 if "3d" in name else 2
        x = torch.randn((2, 2) + (size,) * rank, requires_grad=True)
        output = (2 + variant,) * rank
        if "max" in name:
            y, indices = getattr(F, f"adaptive_max_pool{rank}d")(
                x, output, return_indices=True
            )
            go = torch.randn_like(y)
            actual = fn(go, x.detach(), indices)
        else:
            y = getattr(F, f"adaptive_avg_pool{rank}d")(x, output)
            go = torch.randn_like(y)
            actual = fn(go, x.detach())
        return actual, torch.autograd.grad(y, x, go)[0]
    if name in {"max_pool1d", "max_pool1d_with_indices"}:
        x = torch.randn(2, 3, 13 + variant)
        args = (x, [3], [2], [1], [1], bool(variant))
        return fn(*args), F.max_pool1d(*args, return_indices="indices" in name)
    if name == "max_pool3d_with_indices_backward":
        x = torch.randn(1, 2, size, size, size, requires_grad=True)
        y, idx = F.max_pool3d(x, 2, 2, return_indices=True)
        go = torch.randn_like(y)
        return (
            fn(go, x.detach(), [2] * 3, [2] * 3, [0] * 3, [1] * 3, False, idx),
            torch.autograd.grad(y, x, go)[0],
        )
    if name in {"max_unpool2d", "max_unpool3d"}:
        rank = 3 if name.endswith("3d") else 2
        x = torch.randn((1, 2) + (6 + 2 * variant,) * rank)
        y, idx = getattr(F, f"max_pool{rank}d")(x, 2, 2, return_indices=True)
        ref = getattr(F, name)(y, idx, 2, stride=2, output_size=x.shape)
        actual = (
            fn(y, idx, list(x.shape[-2:]))
            if rank == 2
            else fn(y, idx, 2, stride=2, output_size=x.shape[-3:])
        )
        return actual, ref
    if name.startswith("fractional_max_pool"):
        rank = 2 if "2d" in name else 3
        x = torch.randn(
            (1, 2) + (size,) * rank, requires_grad=name.endswith("backward")
        )
        samples = torch.rand(1, 2, rank)
        y, idx = getattr(F, f"fractional_max_pool{rank}d")(
            x, 2, output_size=(3,) * rank, return_indices=True, _random_samples=samples
        )
        if name.endswith("backward"):
            go = torch.randn_like(y)
            return (
                fn(go, x.detach(), [2] * rank, [3] * rank, idx),
                torch.autograd.grad(y, x, go)[0],
            )
        return fn(x, [2] * rank, output_size=[3] * rank, _random_samples=samples), (
            y,
            idx,
        )
    if name.startswith("quantized_max_pool"):
        rank = int(name[-2])
        x = torch.quantize_per_tensor(
            torch.randn((1, 2) + (size,) * rank), 0.05, 7, torch.quint8
        )
        args = (x, [2] * rank, [2] * rank, [0] * rank, [1] * rank, bool(variant))
        # Integer max commutes with a positive affine quantizer.
        ref_float = getattr(F, f"max_pool{rank}d")(x.dequantize(), *args[1:])
        ref = torch.quantize_per_tensor(
            ref_float, x.q_scale(), x.q_zero_point(), x.dtype
        )
        return fn(*args), ref
    if name == "im2col":
        x = torch.randn(2, 3, size, size + 1)
        args = (x, [3, 2], [1, 1], [1, 0], [1 + variant, 1])
        return fn(*args), F.unfold(
            x, args[1], dilation=args[2], padding=args[3], stride=args[4]
        )
    if name == "convert_indices_from_coo_to_csr":
        rows = torch.tensor([0, 0, 2, 3, 3, 3], dtype=torch.int64)
        kwargs = {"out_int32": bool(variant)}
        return fn(rows, 5, **kwargs), torch._convert_indices_from_coo_to_csr(
            rows, 5, **kwargs
        )
    if name == "convert_indices_from_csr_to_coo":
        rows = torch.tensor([0, 2, 2, 3, 6], dtype=torch.int64)
        cols = torch.tensor([0, 3, 1, 0, 2, 4], dtype=torch.int64)
        kwargs = {"out_int32": bool(variant), "transpose": bool(variant)}
        return fn(rows, cols, **kwargs), torch._convert_indices_from_csr_to_coo(
            rows, cols, **kwargs
        )
    if name == "_convert_weight_to_int4pack":
        # This FlagGems entry packs adjacent int32 nibbles into uint8 pairs;
        # it does not expose ATen's hardware-specific blocked weight layout.
        x = torch.randint(0, 16, (16, 64 + variant * 64), dtype=torch.int32)
        ref = ((x[:, 0::2] & 15) | ((x[:, 1::2] & 15) << 4)).to(torch.uint8)
        return fn(x, 2 + 2 * variant), ref
    if name == "mkldnn_reorder_conv2d_weight":
        if not torch.backends.mkldnn.is_available():
            pytest.skip("PyTorch on this CPU has no oneDNN/MKLDNN tensor layout")
        x = torch.randn(4, 2, 3, 3).to_mkldnn()
        return (
            fn(x).to_dense(),
            torch.ops.aten.mkldnn_reorder_conv2d_weight(x).to_dense(),
        )
    if name in {"conv_tbc", "conv_tbc_backward"}:
        x = torch.randn(7 + variant, 2, 3, requires_grad=True)
        w = torch.randn(3, 3, 4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        ref = torch.conv_tbc(x, w, b, variant)
        if name == "conv_tbc":
            return fn(x.detach(), w.detach(), b.detach(), variant), ref
        go = torch.randn_like(ref)
        return fn(go, x.detach(), w.detach(), b.detach(), variant), torch.autograd.grad(
            ref, (x, w, b), go
        )
    if name == "_convolution_double_backward":
        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        w = torch.randn(3, 2, 2, 2, requires_grad=True)
        b = torch.randn(3, requires_grad=True)
        y = F.conv2d(x, w, b)
        go = torch.randn_like(y, requires_grad=True)
        g = torch.autograd.grad(y, (x, w, b), go, create_graph=True)
        gg = tuple(torch.randn_like(t) for t in g)
        loss = sum((a * z).sum() for a, z in zip(g, gg))
        ref = torch.autograd.grad(loss, (go, x, w))
        actual = fn(
            *gg,
            go.detach(),
            w.detach(),
            x.detach(),
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
            [True] * 3,
        )
        return actual, ref
    # Convolution families share a PyTorch autograd oracle; their launchers and
    # layouts remain independent of that oracle.
    rank = 3 if "3d" in name else 2
    transposed = "transpose" in name
    groups = 2 if name == "conv_depthwise3d" else 1
    x = torch.randn((1, 2) + (size,) * rank, requires_grad=True)
    wshape = (2, 3) if transposed else ((2, 1) if groups == 2 else (3, 2))
    w = torch.randn(wshape + (2,) * rank, requires_grad=True)
    b = torch.randn(2 if groups == 2 else 3, requires_grad=True)
    stride, padding, dilation = [1 + variant] * rank, [variant] * rank, [1] * rank
    outpad = [0] * rank
    conv = getattr(F, ("conv_transpose" if transposed else "conv") + str(rank) + "d")
    kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    ref = conv(x, w, b, **kwargs)
    xd, wd, bd = x.detach(), w.detach(), b.detach()
    if name in {"convolution", "convolution_overrideable"}:
        return fn(xd, wd, bd, stride, padding, dilation, False, outpad, groups), ref
    if name in {
        "convolution_backward",
        "convolution_backward_overrideable",
        "_slow_conv2d_backward",
    }:
        go = torch.randn_like(ref)
        expected = torch.autograd.grad(ref, (x, w, b), go)
        args = [go, xd, wd]
        if name == "convolution_backward":
            args.append(list(b.shape))
        if name == "_slow_conv2d_backward":
            actual = fn(*args, [2] * rank, stride, padding, [True] * 3)
        else:
            actual = fn(
                *args, stride, padding, dilation, False, outpad, groups, [True] * 3
            )
        return actual, expected
    if name == "conv_depthwise3d":
        return fn(xd, wd, [2] * rank, bd, stride, padding, dilation), ref
    if name == "conv_transpose3d":
        return fn(xd, wd, bd, **kwargs), ref
    if name == "cudnn_convolution_transpose":
        return fn(
            xd,
            wd,
            padding=padding,
            output_padding=outpad,
            stride=stride,
            dilation=dilation,
        ), conv(x, w, None, **kwargs)
    if name.startswith("slow_conv_dilated"):
        return fn(xd, wd, [2] * rank, bd, stride, padding, dilation), ref
    if name.startswith("slow_conv_transpose"):
        return fn(xd, wd, [2] * rank, bd, stride, padding, outpad, dilation), ref
    if name in {"thnn_conv2d", "_slow_conv2d_forward"}:
        return fn(xd, wd, [2] * rank, bd, stride, padding), ref
    raise AssertionError(f"No reference case registered for {name}")

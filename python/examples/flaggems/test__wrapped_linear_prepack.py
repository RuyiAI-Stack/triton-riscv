import importlib

import pytest
import torch

from ._wrapped_linear_prepack import _wrapped_linear_prepack, unpack_linear_weight

# aten::::_wrapped_linear_prepack is CPU-callable but returns an opaque
# 8-byte handle blob, so packed values cannot be compared against it; only
# its error contract is cross-checked (see the guard tests).
torch_wlp = torch._wrapped_linear_prepack

# (out_channel N, in_channel K)
SHAPES = [
    (1, 1),
    (4, 8),
    (16, 32),
    (7, 33),  # odd sizes, tail lanes masked
    (64, 128),  # multi-block on the weight span
    (3, 1025),  # N*K not divisible by the block
]


def quantize_reference(weight, scale, zero_point):
    # round-half-to-even (torch.round) + zero point, saturated to int8 -
    # the math the kernel performs in float32.
    q = torch.round(weight / scale) + zero_point
    return q.clamp(-128.0, 127.0).to(torch.int8)


def check_pack_layout(packed, weight, scale, zero_point, bias):
    N, K = weight.shape
    assert packed.dtype == torch.uint8
    assert packed.dim() == 1
    assert packed.numel() == 16 + 4 * N + N * K

    w, metadata, packed_bias = unpack_linear_weight(packed, N, K)
    assert w.dtype == torch.int8
    assert w.shape == (N, K)
    assert metadata.shape == (4,)
    assert packed_bias.shape == (N,)

    torch.testing.assert_close(metadata[0], torch.tensor(scale), rtol=0, atol=0)
    torch.testing.assert_close(metadata[1], torch.tensor(zero_point), rtol=0, atol=0)
    torch.testing.assert_close(metadata[2], torch.tensor(260821.0), rtol=0, atol=0)
    torch.testing.assert_close(metadata[3], torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(packed_bias, bias, rtol=0, atol=0)
    assert torch.equal(w, quantize_reference(weight, scale, zero_point))


@pytest.mark.parametrize("N, K", SHAPES)
@pytest.mark.parametrize("zero_point", [0.0, 3.0, -7.0])
def test_wrapped_linear_prepack_layout_and_values(N, K, zero_point):
    torch.manual_seed(0)
    weight = torch.randn(N, K)
    scale = torch.tensor(0.05)
    zp = torch.tensor(zero_point)
    bias = torch.randn(N)

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    check_pack_layout(packed, weight, 0.05, zero_point, bias)


def test_wrapped_linear_prepack_round_ties_half_to_even():
    # scale chosen so weight/scale lands exactly on .5 ties: 0.75/0.25 = 3,
    # so w = 0.125 -> 0.5 (rounds to 0), w = 0.375 -> 1.5 (rounds to 2).
    weight = torch.tensor([[0.125, 0.375, -0.125, -0.375, 0.625, 2.625]])
    scale = torch.tensor(0.25)
    zp = torch.tensor(0.0)
    bias = torch.zeros(1)

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    w, _, _ = unpack_linear_weight(packed, 1, 6)
    assert w.flatten().tolist() == [0, 2, 0, -2, 2, 10]  # half-to-even on every tie


def test_wrapped_linear_prepack_saturates():
    weight = torch.tensor([[100.0, -100.0, 6.0, -6.0]])
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    bias = torch.zeros(1)

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    w, _, _ = unpack_linear_weight(packed, 1, 4)
    assert w.flatten().tolist() == [127, -128, 60, -60]


def test_wrapped_linear_prepack_zero_weight():
    # All-zero weight quantizes to the zero point everywhere.
    weight = torch.zeros(5, 9)
    scale = torch.tensor(0.5)
    zp = torch.tensor(11.0)
    bias = torch.arange(5, dtype=torch.float32)

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    check_pack_layout(packed, weight, 0.5, 11.0, bias)


def test_wrapped_linear_prepack_non_contiguous_weight():
    torch.manual_seed(1)
    weight = torch.randn(8, 6).t()  # (6, 8) non-contiguous
    scale = torch.tensor(0.1)
    zp = torch.tensor(1.0)
    bias = torch.randn(6)

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    check_pack_layout(packed, weight.contiguous(), 0.1, 1.0, bias)


def test_wrapped_linear_prepack_non_contiguous_bias():
    torch.manual_seed(1)
    weight = torch.randn(6, 8)
    scale = torch.tensor(0.1)
    zp = torch.tensor(1.0)
    bias = torch.randn(12)[::2]
    assert not bias.is_contiguous()

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    check_pack_layout(packed, weight, 0.1, 1.0, bias.contiguous())


@pytest.mark.parametrize("N, K", [(0, 8), (4, 0), (0, 0)])
def test_wrapped_linear_prepack_empty_dimensions(N, K):
    # The empty kernel writes header + bias only; no weight payload.
    weight = torch.randn(N, K)
    scale = torch.tensor(0.05)
    zp = torch.tensor(2.0)
    bias = torch.arange(N, dtype=torch.float32)

    packed = _wrapped_linear_prepack(weight, scale, zp, bias)
    assert packed.numel() == 16 + 4 * N + N * K
    w, metadata, packed_bias = unpack_linear_weight(packed, N, K)
    assert metadata[0].item() == pytest.approx(0.05, abs=1e-7)
    assert metadata[1].item() == 2.0
    torch.testing.assert_close(packed_bias, bias, rtol=0, atol=0)


def test_wrapped_linear_prepack_rejects_non_float32_weight():
    # aten raises the same message ("Quantize only works on Float Tensor,
    # got Double"); the port keeps the upstream guard text.
    weight = torch.randn(4, 8, dtype=torch.float64)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    bias = torch.randn(4)

    with pytest.raises(RuntimeError, match="Quantize only works on Float"):
        _wrapped_linear_prepack(weight, scale, zp, bias)
    # FBGEMM-less builds (the riscv64 CI torch) reject at dispatch with a
    # build message before aten's dtype/shape guards can run.
    with pytest.raises(
        RuntimeError, match="Quantize only works on Float|not built with FBGEMM"
    ):
        torch_wlp(weight, scale, zp, bias)


def test_wrapped_linear_prepack_rejects_vector_weight():
    # aten: "fbgemm weight packing only packs matrices not vectors."
    weight = torch.randn(8)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    bias = torch.randn(1)

    with pytest.raises(RuntimeError, match="only packs matrices"):
        _wrapped_linear_prepack(weight, scale, zp, bias)
    # Same FBGEMM-less dispatch rejection as the dtype test above.
    with pytest.raises(RuntimeError, match="only packs matrices|not built with FBGEMM"):
        torch_wlp(weight, scale, zp, bias)


@pytest.mark.parametrize(
    "scale, zp",
    [
        (torch.randn(2), torch.tensor(0.0)),
        (torch.tensor(0.1), torch.randn(2)),
    ],
)
def test_wrapped_linear_prepack_rejects_multi_element_qparams(scale, zp):
    weight = torch.randn(4, 8)
    bias = torch.randn(4)
    with pytest.raises(RuntimeError, match="one element"):
        _wrapped_linear_prepack(weight, scale, zp, bias)


@pytest.mark.parametrize(
    "bias",
    [
        torch.randn(4, dtype=torch.float64),  # wrong dtype
        torch.randn(3),  # wrong length
        torch.randn(4, 4),  # not a vector
    ],
)
def test_wrapped_linear_prepack_rejects_bad_bias(bias):
    weight = torch.randn(4, 8)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="bias"):
        _wrapped_linear_prepack(weight, scale, zp, bias)


def test_unpack_linear_weight_rejects_bad_layout():
    weight = torch.randn(4, 8)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    bias = torch.randn(4)
    packed = _wrapped_linear_prepack(weight, scale, zp, bias)

    with pytest.raises(RuntimeError, match="uint8"):
        unpack_linear_weight(packed.view(torch.float32)[:8], 4, 8)
    with pytest.raises(RuntimeError, match="one-dimensional"):
        unpack_linear_weight(packed.view(8, 8), 4, 8)
    with pytest.raises(RuntimeError, match="non-negative"):
        unpack_linear_weight(packed, -1)
    with pytest.raises(RuntimeError, match="invalid FlagGems linear layout"):
        # header + partial bias: the weight payload goes negative
        unpack_linear_weight(packed[:20], 4, 8)
    with pytest.raises(RuntimeError, match="invalid FlagGems linear layout"):
        # payload size not a multiple of out_channel
        unpack_linear_weight(packed[:-1], 4, 8)
    with pytest.raises(RuntimeError, match="do not match"):
        unpack_linear_weight(packed, 4, 7)  # wrong in_channel hint


def test_wrapped_linear_prepack_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._wrapped_linear_prepack")

    launches = []
    real_run = mod._wrapped_linear_prepack_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._wrapped_linear_prepack_kernel, "run", spy_run)
    torch.manual_seed(0)
    weight = torch.randn(4, 8)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    bias = torch.randn(4)
    packed = mod._wrapped_linear_prepack(weight, scale, zp, bias)

    assert len(launches) == 1
    assert launches[0] == (1,)  # cdiv(max(32, 4), 1024)
    check_pack_layout(packed, weight, 0.1, 0.0, bias)


def test_wrapped_linear_prepack_empty_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._wrapped_linear_prepack")

    launches = []
    real_run = mod._wrapped_linear_prepack_empty_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._wrapped_linear_prepack_empty_kernel, "run", spy_run)
    weight = torch.randn(4, 0)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    bias = torch.randn(4)
    packed = mod._wrapped_linear_prepack(weight, scale, zp, bias)

    assert len(launches) == 1
    w, _, packed_bias = unpack_linear_weight(packed, 4, 0)
    torch.testing.assert_close(packed_bias, bias, rtol=0, atol=0)

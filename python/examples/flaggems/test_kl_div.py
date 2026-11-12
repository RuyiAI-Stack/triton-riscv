import pytest
import torch

from .kl_div import kl_div

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


def _make_probs(size):
    torch.manual_seed(0)
    logits = torch.randn(size)
    log_p = torch.log_softmax(logits, dim=0)
    target = torch.softmax(torch.randn(size), dim=0)
    return log_p, target


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_kl_div(size, reduction):
    log_p, target = _make_probs(size)

    ref = torch.nn.functional.kl_div(log_p, target, reduction=reduction)
    tri = kl_div(log_p, target, reduction=reduction)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_kl_div_zero_targets():
    # Zero targets contribute exactly zero (xlogy semantics).
    log_p = torch.randn(64)
    target = torch.softmax(torch.randn(64), dim=0)
    target[:16] = 0.0

    ref = torch.nn.functional.kl_div(log_p, target, reduction="none")
    tri = kl_div(log_p, target, reduction="none")

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_kl_div_log_target():
    torch.manual_seed(0)
    log_p = torch.log_softmax(torch.randn(256), dim=0)
    target_log = torch.log_softmax(torch.randn(256), dim=0)

    ref = torch.nn.functional.kl_div(
        log_p, target_log, reduction="none", log_target=True
    )
    tri = kl_div(log_p, target_log, reduction="none", log_target=True)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_kl_div_fp64_fallback():
    torch.manual_seed(0)
    log_p = torch.log_softmax(torch.randn(256, dtype=torch.float64), dim=0)
    target = torch.softmax(torch.randn(256, dtype=torch.float64), dim=0)

    ref = torch.nn.functional.kl_div(log_p, target)
    tri = kl_div(log_p, target)

    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_kl_div_bf16():
    torch.manual_seed(0)
    log_p = torch.log_softmax(torch.randn(128), dim=0).to(torch.bfloat16)
    target = torch.softmax(torch.randn(128), dim=0).to(torch.bfloat16)

    ref = torch.nn.functional.kl_div(log_p.float(), target.float()).to(torch.bfloat16)
    tri = kl_div(log_p, target)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_kl_div_fp16():
    torch.manual_seed(0)
    log_p = torch.log_softmax(torch.randn(512), dim=0).to(torch.float16)
    target = torch.softmax(torch.randn(512), dim=0).to(torch.float16)
    torch.testing.assert_close(
        kl_div(log_p, target),
        torch.nn.functional.kl_div(log_p.float(), target.float()).to(torch.float16),
        rtol=1e-2,
        atol=1e-2,
    )


def test_kl_div_bad_reduction_raises():
    log_p = torch.randn(16)
    t = torch.softmax(torch.randn(16), dim=0)
    with pytest.raises(ValueError):
        kl_div(log_p, t, reduction="tot")


def test_kl_div_broadcast():
    torch.manual_seed(0)
    log_p = torch.log_softmax(torch.randn(2, 3), dim=-1)
    t = torch.softmax(torch.randn(1, 3), dim=-1)
    torch.testing.assert_close(kl_div(log_p, t), torch.nn.functional.kl_div(log_p, t))


def test_kl_div_int_raises_like_torch():
    x = torch.randn(8)
    t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(RuntimeError):
        kl_div(x, t)

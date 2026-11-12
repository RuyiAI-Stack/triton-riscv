import pytest
import torch

from .hamming_window import hamming_window

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [17, 128, 1023])
@pytest.mark.parametrize("periodic", [True, False])
def test_hamming_window(size, periodic):
    ref = torch.hamming_window(size, periodic=periodic)
    tri = hamming_window(size, periodic=periodic)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("alpha, beta", [(0.54, 0.46), (0.75, 0.25)])
def test_hamming_window_alpha_beta(alpha, beta):
    ref = torch.hamming_window(128, periodic=True, alpha=alpha, beta=beta)
    tri = hamming_window(128, periodic=True, alpha=alpha, beta=beta)
    torch.testing.assert_close(tri, ref)


def test_hamming_window_fp16():
    ref = torch.hamming_window(128, dtype=torch.float16)
    tri = hamming_window(128, dtype=torch.float16)
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_hamming_window_fp64_fallback():
    ref = torch.hamming_window(128, dtype=torch.float64)
    tri = hamming_window(128, dtype=torch.float64)
    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_hamming_window_bf16():
    ref = torch.hamming_window(128, dtype=torch.bfloat16)
    tri = hamming_window(128, dtype=torch.bfloat16)
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("periodic", [True, False])
def test_hamming_window_length_one(periodic):
    # torch special-cases length-1 windows to [1.0] for both forms; the
    # symmetric path additionally trips Triton's ==1 argument specialization
    # on formula_len - 1 == 0.
    ref = torch.hamming_window(1, periodic=periodic)
    tri = hamming_window(1, periodic=periodic)
    assert tri.shape == (1,)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_hamming_window_tiny_lengths(n, periodic):
    torch.testing.assert_close(
        hamming_window(n, periodic=periodic), torch.hamming_window(n, periodic=periodic)
    )


def test_hamming_window_torch_kwargs():
    # torch-legal kwargs must be accepted; unsupported values must raise.
    import torch as _t

    out = hamming_window(8, layout=_t.strided, device="cpu", requires_grad=False)
    assert out.shape == (8,)
    with pytest.raises(RuntimeError):
        hamming_window(8, dtype=_t.int32)
    with pytest.raises(NotImplementedError):
        hamming_window(8, requires_grad=True)

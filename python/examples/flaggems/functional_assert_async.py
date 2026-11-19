import torch
import triton
import triton.language as tl


@triton.jit
def _functional_assert_async_kernel(x_ptr, MSG: tl.constexpr):
    val = tl.load(x_ptr)
    tl.device_assert(val != 0, MSG)


def _functional_assert_async(tensor, assert_msg, dep_token):
    """Functional _assert_async that takes a dependency token and returns a
    new one.

    A single-element non-zero tensor passes (no-op) and a fresh empty token
    with dep_token's dtype/device is returned; a zero value raises
    RuntimeError(assert_msg), matching torch.ops.aten._functional_assert_async
    on CPU. tl.device_assert is advisory on this CPU backend (it never fires),
    so the value is checked on the host before the kernel launch; the kernel
    is kept from the upstream file for backend parity.
    """
    if tensor.numel() != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with shape {list(tensor.shape)} is ambiguous"
        )
    if tensor.item() == 0:
        raise RuntimeError(assert_msg)

    _functional_assert_async_kernel[(1,)](tensor, MSG=assert_msg)
    return torch.empty(0, dtype=dep_token.dtype, device=dep_token.device)

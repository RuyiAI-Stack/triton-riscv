import torch

import triton
import triton.language as tl


@triton.jit
def block_copy_kernel(a_ptr, b_ptr):
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=(4, 4),
        strides=(4, 1),
        block_shape=(1, 4),
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=(1, 4),
        strides=(4, 1),
        block_shape=(1, 4),
    )
    a = a_desc.load((2, 0))
    b_desc.store((0, 0), a)


def test(device):
    input = torch.arange(0, 16, device=device, dtype=torch.float32)
    output = torch.full((4,), -1, device=device, dtype=torch.float32)
    expected = torch.arange(8, 12, device=device, dtype=torch.float32)

    def grid(meta):
        return (1,)

    block_copy_kernel[grid](input, output)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)

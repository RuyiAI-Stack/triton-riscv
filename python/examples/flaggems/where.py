import torch
import triton
import triton.language as tl


@triton.jit
def where_kernel(
    cond_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    cond = tl.load(cond_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = tl.where(cond, x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


def where_self(condition, self, other):
    return where_self_out(condition, self, other)


def where_self_out(condition, self, other, out=None):
    result_type = torch.result_type(self, other)
    if out is not None:
        assert out.dtype == result_type, (
            f"Expected out type to be {result_type}, but got {out.dtype}."
        )

    c, a, b = list(
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            (condition, self, other),
        )
    )

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    assert c.dtype == torch.bool, (
        f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"
    )

    if out is None:
        out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
        out = torch.empty(out_shape, dtype=result_type, device="cpu")

    c = c.expand(out.shape).contiguous()
    a = a.expand(out.shape).contiguous()
    b = b.expand(out.shape).contiguous()
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    where_kernel[grid](c, a, b, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def where(condition, self, other):
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    return where_self_out(condition, self, other)

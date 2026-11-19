import torch
import triton
import triton.language as tl

# _fused_dropout: returns (output, mask) where mask is a uint8 Bernoulli
# (1 - p) draw and output = input * mask / (1 - p). ATen has no CPU
# implementation for this operator (NotImplementedError), so the contract
# here is the one measured against the CUDA semantics: p == 0 returns the
# input values with an all-ones mask, p == 1 returns zeros with an all-
# zeros mask, and an empty input returns empty tensors. Only floating-point
# inputs are accepted; a non-Generator object is a TypeError.
#
# The mask is drawn from the torch RNG (or the caller's Generator) and the
# rescale runs in the Triton kernel.

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def _fused_dropout_kernel(
    x_ptr,
    keep_ptr,
    out_ptr,
    n_elements,
    one_minus_p,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Dividing by (1 - p) in-kernel (instead of multiplying by a host
    # precomputed scale) keeps full precision on the fp64 path: python-float
    # scalar arguments are rounded to fp32 at kernel launch.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0)
    if IS64:
        xc = x.to(tl.float64)
    else:
        xc = x.to(tl.float32)
    res = tl.where(keep != 0, xc / one_minus_p, 0.0 * xc)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _fused_dropout(self, p, generator=None):
    """Dropout returning (output, uint8 mask); see the module comment."""
    if self.dtype not in _FLOAT_DTYPES:
        raise RuntimeError("_fused_dropout supports floating-point inputs only")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("_fused_dropout generator must be a torch.Generator")
        if generator.device != self.device:
            raise RuntimeError("_fused_dropout generator device does not match input")
    n = self.numel()
    mask = torch.empty(self.shape, dtype=torch.uint8, device=self.device)
    if n == 0:
        return self.clone(), mask
    if p == 0.0:
        mask.fill_(1)
        return self.clone(), mask
    if p == 1.0:
        return torch.zeros_like(self), mask.zero_()
    if generator is None:
        u = torch.rand(self.shape, dtype=torch.float32, device=self.device)
    else:
        u = torch.rand(
            self.shape, dtype=torch.float32, device=self.device, generator=generator
        )
    keep = (u >= p).to(torch.uint8)
    mask.copy_(keep)
    # Allocate a contiguous result: the kernel writes flat offsets, so an
    # empty_like(self) that inherited strided strides from a non-contiguous
    # input would be filled in the wrong layout.
    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    _fused_dropout_kernel[(triton.cdiv(n, 1024),)](
        self.contiguous(),
        keep,
        output,
        n,
        1.0 - float(p),
        IS64=self.dtype == torch.float64,
        BLOCK=1024,
    )
    return output, mask

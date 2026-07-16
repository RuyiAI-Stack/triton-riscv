import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel

triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def _layer_norm_inference_cpu(
    X,
    Y,
    W,
    B,
    EPS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    # A single structured kernel avoids per-row tensor staging and the mean /
    # rstd allocations required by the training-oriented autograd wrapper.
    for row in tl.range(0, M):
        base = row * N
        mean = 0.0
        for col in tl.range(0, N):
            mean += tl.load(X + base + col).to(tl.float32)
        mean /= N

        variance = 0.0
        for col in tl.range(0, N):
            centered = tl.load(X + base + col).to(tl.float32) - mean
            variance += centered * centered
        rstd = 1.0 / tl.sqrt(variance / N + EPS)

        for col in tl.range(0, N):
            x = tl.load(X + base + col).to(tl.float32)
            weight = tl.load(W + col).to(tl.float32)
            bias = tl.load(B + col).to(tl.float32)
            tl.store(Y + base + col, (x - mean) * rstd * weight + bias)


def layer_norm_inference(x, weight, bias, eps):
    x_2d = x.reshape(-1, x.shape[-1])
    output = torch.empty_like(x)
    m, n = x_2d.shape
    key = (m, n, eps, x.dtype, weight.dtype, bias.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            _layer_norm_inference_cpu,
            (1,),
            x_2d,
            output,
            weight,
            bias,
            EPS=eps,
            M=m,
            N=n,
            allow_fp_reassoc=True,
        )
        _prepared_kernels[key] = runner
    runner(x_2d, output, weight, bias)
    return output


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, device):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=device)
        rstd = torch.empty((M,), dtype=torch.float32, device=device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M,)](  #
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,  #
            x_arg.stride(0),
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_ctas=1,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y


def bench_layernorm(size):
    layer_norm = LayerNorm.apply
    device = "cpu"
    eps = 1e-5
    dtype = torch.float16
    try:
        F.layer_norm(
            torch.randn((1, 8), dtype=dtype, device=device),
            (8,),
            torch.rand((8,), dtype=dtype, device=device),
            torch.rand((8,), dtype=dtype, device=device),
            eps,
        )
    except RuntimeError:
        dtype = torch.float32
    x_shape = (size, size)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(False)
    benchmark.compare_providers(
        f"bench_layernorm(size={size})",
        {
            "torch": lambda: F.layer_norm(x, w_shape, weight, bias, eps),
            "triton-riscv": lambda: layer_norm_inference(x, weight, bias, eps),
        },
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for X in [2**i for i in range(6, 8, 1)]:
        bench_layernorm(X)

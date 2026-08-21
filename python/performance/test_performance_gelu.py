import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from flaggems.gelu import gelu


def bench_gelu(size):
    torch.manual_seed(0)
    x = torch.randn((size,), device="cpu", dtype=torch.float32)

    benchmark.compare_providers(
        f"bench_gelu(size={size})",
        {
            "torch": lambda: F.gelu(x, approximate="tanh"),
            "triton-riscv": lambda: gelu(x, approximate="tanh"),
        },
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**18, 2**20, 2**22]:
        bench_gelu(size)

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from flaggems.rms_norm import rms_norm_forward


def bench_rms_norm(rows, cols):
    torch.manual_seed(0)
    x = torch.randn((rows, cols), device="cpu", dtype=torch.float32)
    weight = torch.randn((cols,), device="cpu", dtype=torch.float32)
    normalized_shape = (cols,)
    eps = 1e-5

    benchmark.compare_providers(
        f"bench_rms_norm(rows={rows}, cols={cols})",
        {
            "torch": lambda: F.rms_norm(x, normalized_shape, weight, eps),
            "triton-riscv": lambda: rms_norm_forward(x, normalized_shape, weight, eps)[
                0
            ],
        },
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(64, 512), (128, 1024), (256, 2048)]:
        bench_rms_norm(rows, cols)

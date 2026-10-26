# RUN: %PYTHON %s %t
"""Execute the ordinary CPU CAS/FP8 paths after the frontend split."""

import os
import shutil
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def cas_kernel(X, Old, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    cmp = tl.full((BLOCK,), 0, tl.int32)
    value = tl.full((BLOCK,), 7, tl.int32)
    old = tl.atomic_cas(X + i, cmp, value)
    tl.store(Old + i, old)


@triton.jit
def fp8_kernel(X, Bytes, Roundtrip, TYPE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.load(X + i).to(TYPE)
    tl.store(Bytes + i, value.to(tl.uint8, bitcast=True))
    tl.store(Roundtrip + i, value.to(tl.float32))


def check_default_path(compiled):
    assert compiled.asm["obj"][:4] == b"\x7fELF"
    for stage in ("ttsharedir", "llir"):
        assert "contract.version" not in compiled.asm[stage]
        assert "bosc_ame" not in compiled.asm[stage]


work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(work / "cache")
os.environ["BUDDY_MLIR_BINARY_DIR"] = str(Path(os.environ["BUDDY_OPT"]).parent)
os.environ["LLVM_BINARY_DIR"] = str(Path(shutil.which("llc")).parent)
for name in ("TRITON_RISCV_USE_IME", "TRITON_RISCV_CROSS_COMPILE"):
    os.environ.pop(name, None)
triton.runtime.driver.set_active(CPUDriver())

x = torch.zeros(64, dtype=torch.int32)
old = torch.empty_like(x)
first = cas_kernel[(1,)](x, old, 64)
assert torch.equal(x, torch.full_like(x, 7))
assert torch.equal(old, torch.zeros_like(old))
cas_kernel[(1,)](x, old, 64)
assert torch.equal(x, torch.full_like(x, 7))
assert torch.equal(old, torch.full_like(old, 7))
check_default_path(first)

for limit in (4, 448):
    values = torch.linspace(-limit, limit, 256, dtype=torch.float32)
    values[:8] = torch.tensor(
        [0.0, -0.0, 0.5, -0.5, 1.0625, -1.0625, 1.1875, -1.1875]
    )
    bits = torch.empty(256, dtype=torch.uint8)
    roundtrip = torch.empty_like(values)
    compiled = fp8_kernel[(1,)](values, bits, roundtrip, tl.float8e4nv, 256)
    expected = values.to(torch.float8_e4m3fn)
    assert torch.equal(bits, expected.view(torch.uint8))
    assert torch.equal(
        roundtrip.view(torch.int32), expected.float().view(torch.int32)
    )
    check_default_path(compiled)

print(
    "CPU CAS success/failure and both FP8 conversions: exact bits, default non-Contract path"
)

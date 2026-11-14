import subprocess

import pytest
import torch

import triton
import triton.language as tl

from test_utils import get_llvm_bin_path
from triton.backends.triton_shared.riscv import DEFAULT_LLC_FEATURES, DEFAULT_TRIPLE

BLOCK = 64


@triton.jit
def convert_to_bf16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, x.to(tl.bfloat16), mask=m)


@triton.jit
def arith_to_bf16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, (x * 3.0 + 1.0).to(tl.bfloat16), mask=m)


@triton.jit
def bf16_to_fp32(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, x.to(tl.float32) * 2.0, mask=m)


def run_to_bf16(kernel, x):
    n = x.numel()
    y = torch.empty(n, dtype=torch.bfloat16)
    kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
    return y


# The conversions must match torch bit-for-bit. On the CI hosts llc lowers
# these scalar fp32<->bf16 conversions to the __truncsfbf2/__extendbfsf2
# libcalls defined in the launcher .so (x86-64 has no native bf16, and the
# native riscv64 path keeps kernels scalar at llc -O0), so these tests pin
# the launcher helpers' ABI and rounding. The register-class mismatch this
# guards against is RuyiAI-Stack/triton-riscv#114: a caller reading the
# bfloat return from xmm0/fa0 while the helper wrote eax/a0 silently
# produced zeros and garbage.
def test_arith_to_bf16(device):
    torch.manual_seed(0)
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], device=device, dtype=torch.float32)
    y = run_to_bf16(arith_to_bf16, x)
    ref = (x * 3.0 + 1.0).to(torch.bfloat16)
    assert torch.equal(y.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("n", [4, 1023])
def test_arith_to_bf16_sized(device, n):
    torch.manual_seed(0)
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = run_to_bf16(arith_to_bf16, x)
    ref = (x * 3.0 + 1.0).to(torch.bfloat16)
    assert torch.equal(y.view(torch.int16), ref.view(torch.int16))


def test_fp32_to_bf16_round_to_nearest_even(device):
    # fp32 bit patterns chosen at the rounding boundaries: below half
    # (0x7fff), tie with even/odd retained lsb (0x8000), above half
    # (0x8001), and float32 max rounding to infinity.
    bits = [
        0x3F800000,  # 1.0, exact
        0x3F807FFF,  # below half -> down
        0x3F808000,  # tie, retained lsb 0 -> stay even
        0x3F818000,  # tie, retained lsb 1 -> up to even
        0x3F808001,  # above half -> up
        0xBF807FFF,  # negative, below half -> down
        0xBF818000,  # negative tie, odd -> up to even
        0x7F7FFFFF,  # float32 max -> bf16 inf
    ]
    signed = [b - (1 << 32) if b >= (1 << 31) else b for b in bits]
    x = torch.tensor(signed, device=device, dtype=torch.int32).view(torch.float32)
    y = run_to_bf16(convert_to_bf16, x)
    ref = x.to(torch.bfloat16)
    got = [hex(v & 0xFFFF) for v in y.view(torch.int16).tolist()]
    exp = [hex(v & 0xFFFF) for v in ref.view(torch.int16).tolist()]
    assert got == exp, f"bf16 bits mismatch: got {got}, expected {exp}"


@pytest.mark.parametrize("n", [3, 130])
def test_bf16_to_fp32(device, n):
    torch.manual_seed(0)
    x = torch.randn(n, device=device, dtype=torch.float32).to(torch.bfloat16)
    y = torch.empty(n, device=device, dtype=torch.float32)
    bf16_to_fp32[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
    ref = x.to(torch.float32) * 2.0
    assert torch.equal(y, ref)


FEATS_WITHOUT_ZVFBFMIN = DEFAULT_LLC_FEATURES.replace(",+zvfbfmin", "")


def dump_relocations(obj_bytes, tmp_path, name):
    obj_path = tmp_path / name
    obj_path.write_bytes(obj_bytes)
    return subprocess.check_output(
        [get_llvm_bin_path("llvm-objdump"), "-r", str(obj_path)], text=True
    )


def test_host_target_fp32_to_bf16_uses_libcall(device, tmp_path):
    # The numeric tests above only exercise the launcher helper if llc
    # actually emits the libcall for the host target. On riscv64 the bare
    # runtime path stays scalar (no opt, no vector passes), so the
    # conversion must go through __truncsfbf2 even though +zvfbfmin is in
    # the default features since #117. Fail loudly if a pipeline change
    # ever replaces the libcall with vector bf16 instructions: the
    # launcher ABI of #114 would silently stop being tested.
    x = torch.randn(64, device=device, dtype=torch.float32)
    y = torch.empty(64, device=device, dtype=torch.bfloat16)
    compiled = arith_to_bf16.warmup(x, y, 64, BLOCK=BLOCK, grid=(1,))
    relocs = dump_relocations(compiled.asm["obj"], tmp_path, "host_fp32_to_bf16.o")
    assert "__truncsfbf2" in relocs


def test_riscv_without_zvfbfmin_uses_libcall(device, tmp_path):
    # With the feature string stripped of +zvfbfmin the same conversion
    # must lower to __truncsfbf2, not to vfncvtbf16 vector instructions,
    # regardless of the host.
    x = torch.randn(64, device=device, dtype=torch.float32)
    y = torch.empty(64, device=device, dtype=torch.bfloat16)
    compiled = arith_to_bf16.warmup(
        x,
        y,
        64,
        BLOCK=BLOCK,
        grid=(1,),
        target_triple=DEFAULT_TRIPLE,
        target_features=FEATS_WITHOUT_ZVFBFMIN,
    )
    relocs = dump_relocations(compiled.asm["obj"], tmp_path, "riscv_no_zvfbfmin.o")
    assert "__truncsfbf2" in relocs

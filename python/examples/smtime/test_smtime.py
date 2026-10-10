from pathlib import Path
import os
import re
import subprocess
import sys

from test_utils import get_llvm_bin_path


ROOT = Path(__file__).resolve().parents[3]
TRITON_DIR = Path(os.environ.get("TRITON_DIR", ROOT.parent / "triton"))


USER_KERNEL = r"""
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def smtime_i8_matmul(a, b, c):
    offs_m = tl.arange(0, 4)
    offs_n = tl.arange(0, 4)
    offs_k = tl.arange(0, 8)
    av = tl.load(a + offs_m[:, None] * 8 + offs_k[None, :])
    bv = tl.load(b + offs_k[:, None] * 4 + offs_n[None, :])
    acc = tl.dot(av, bv, out_dtype=tl.int32)
    tl.store(c + offs_m[:, None] * 4 + offs_n[None, :], acc)


a = torch.randint(-8, 8, (4, 8), dtype=torch.int8, device="cpu")
b = torch.randint(-8, 8, (8, 4), dtype=torch.int8, device="cpu")
c = torch.empty((4, 4), dtype=torch.int32, device="cpu")
compiled = smtime_i8_matmul.warmup(a, b, c, grid=(1,))
Path(sys.argv[1]).write_bytes(compiled.asm["obj"])
"""


def test_user_kernel_warmup_emits_riscv_ime_object(tmp_path):
    kernel_path = tmp_path / "smtime_user_kernel.py"
    obj_path = tmp_path / "smtime_user_kernel.o"
    kernel_path.write_text(USER_KERNEL)

    env = os.environ.copy()
    env.update({"TRITON_RISCV_USE_IME": "1"})
    subprocess.check_call(
        [sys.executable, str(kernel_path), str(obj_path)],
        cwd=TRITON_DIR,
        env=env,
    )
    assert obj_path.stat().st_size > 0

    header = subprocess.check_output(
        [get_llvm_bin_path("llvm-readelf"), "-h", str(obj_path)], text=True
    )
    assert "Class:                             ELF64" in header
    assert "Type:                              REL (Relocatable file)" in header
    assert "Machine:                           RISC-V" in header

    symbols = subprocess.check_output(
        [get_llvm_bin_path("llvm-readelf"), "-s", str(obj_path)], text=True
    )
    assert "FUNC    GLOBAL DEFAULT" in symbols
    assert "smtime_i8_matmul" in symbols
    assert "xsmtime" in symbols

    asm = subprocess.check_output(
        [get_llvm_bin_path("llvm-objdump"), "-d", str(obj_path)], text=True
    )
    assert re.search(r"<smtime_i8_matmul>:", asm)
    assert len(re.findall(r"\bvmadot\b", asm)) == 1
    assert "vsetvli" in asm
    assert "vle8.v" in asm
    assert "vle32.v" in asm
    assert "vse32.v" in asm

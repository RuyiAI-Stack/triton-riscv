from pathlib import Path
import ctypes
import errno
import os
import platform
import re
import shutil

import pytest


def get_llvm_bin_path(bin_name):
    llvm_binary_dir = os.environ.get("LLVM_BINARY_DIR")
    if llvm_binary_dir:
        return str(Path(llvm_binary_dir) / bin_name)

    path = shutil.which(bin_name)
    if path:
        return path

    raise RuntimeError(f"Unable to locate '{bin_name}' via LLVM_BINARY_DIR or PATH.")


class _RiscvHwprobe(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_int64),
        ("value", ctypes.c_uint64),
    ]


SYS_RISCV_HWPROBE = 258
RISCV_HWPROBE_KEY_IMA_EXT_0 = 4
RISCV_HWPROBE_IMA_V = 1 << 2


def _hwprobe_has_rvv():
    pairs = (_RiscvHwprobe * 1)(_RiscvHwprobe(RISCV_HWPROBE_KEY_IMA_EXT_0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    ret = libc.syscall(
        SYS_RISCV_HWPROBE,
        pairs,
        ctypes.c_size_t(1),
        ctypes.c_size_t(0),
        ctypes.c_void_p(0),
        ctypes.c_uint(0),
    )
    if ret == 0 and pairs[0].key == RISCV_HWPROBE_KEY_IMA_EXT_0:
        return bool(pairs[0].value & RISCV_HWPROBE_IMA_V)

    err = ctypes.get_errno()
    if err in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        return None

    return False


def _riscv_isa_has_v(isa):
    for ext in re.split(r"[_\s]+", isa.lower()):
        if ext == "v":
            return True
        if (ext.startswith("rv32") or ext.startswith("rv64")) and "v" in ext[4:]:
            return True

    return False


def supports_rvv_execution():
    if platform.machine() != "riscv64":
        return False

    hwprobe_result = _hwprobe_has_rvv()
    if hwprobe_result is not None:
        return hwprobe_result

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False

    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in {"isa", "riscv isa", "hart isa"}:
            return _riscv_isa_has_v(value)

    return False


requires_rvv_execution = pytest.mark.skipif(
    not supports_rvv_execution(),
    reason="requires native RVV support to execute generated RISC-V kernels",
)

# Native RISC-V llc often emits whole-register loads/stores (vl1re32.v / vs4r.v)
# instead of unit-stride vle32.v / vse32.v used by x86 cross-compiled objects.
RVV_VECTOR_LOAD = re.compile(
    r"\b(?:vle(?:8|16|32|64)\.v|vl\d+re(?:8|16|32|64)\.v|vl\d+r\.v)\b"
)
RVV_VECTOR_STORE = re.compile(r"\b(?:vse(?:8|16|32|64)\.v|vs\d+r\.v)\b")

# triton-riscv

Triton compiler for RISC-V platforms.

This repository is forked from [triton-shared](https://github.com/microsoft/triton-shared) and provides a Triton compiler backend for RISC-V. The upstream triton-shared repo is no longer maintained, so this project is developed independently under the name triton-riscv.

Triton acts as a frontend (Python AST → TTIR); triton-riscv handles the rest of the pipeline (TTIR → Linalg → LLVM IR → native object). No NVIDIA or AMD toolchain is required.

**Compilation pipeline:**

```
Python @triton.jit
  └─► TTIR  (triton: ast_to_ttir + make_ttir passes)
        └─► Linalg MLIR  (triton-shared-opt --triton-to-linalg-experimental)
              └─► LLVM MLIR  (Buddy Compiler lowering passes)
                    └─► LLVM IR  (mlir-translate --mlir-to-llvmir)
                          └─► native .o  (llc -filetype=obj)
```

## Install

### From binary

```sh
python3 -m venv .venv && . .venv/bin/activate
# `triton` and the buddy tools come from the index that carries them; PyPI only
# holds upstream Triton. Until that index has them, both projects attach the same
# wheels to their GitHub releases.
pip install --index-url https://forgestash.artifact.yuantu.ac.cn/repository/pypi-riscv-local-ruyiai/simple/ triton buddy
pip install torch --index-url https://community-ci.openruyi.cn/pypi/riscv64/dev/+simple/torch  # optional

# Get prebuilt LLVM from https://github.com/buddy-compiler/buddy-mlir/releases
mkdir -p /path/to/llvm
tar xf llvm-*.tar.gz -C /path/to/llvm
export PATH=/path/to/llvm/bin:$PATH
```

### From source

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git
cd triton-riscv
python3 -m venv .venv && . .venv/bin/activate
make setup    # downloads the pinned LLVM and buddy builds, clones Triton, installs the build deps
make build    # builds Triton and installs the backend
make test     # the plugin's lit tests and the example suite
```

### Test the installation

`scripts/env.sh` exports what the backend and the lit tests need to find the toolchains and the
staged `triton-shared-opt`, so source it before running anything by hand.

```sh
source scripts/env.sh
python3 -c "import triton; print(triton.__version__, triton.runtime.driver.active.get_current_target())"
```

On RISC-V it prints `3.9.0 GPUTarget(backend='cpu', arch=0, warp_size=0)`; on RISC-V with an AMD GPU
it prints `3.9.0 GPUTarget(backend='hip', arch='gfx942', warp_size=64)`, with the GPU's own arch.

## Quick start

### RISC-V CPU

```py
# gemm.py
import torch
import triton
import triton.language as tl


@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs[:, None] * BLOCK + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * BLOCK + offs[None, :])
    tl.store(c_ptr + offs[:, None] * BLOCK + offs[None, :], tl.dot(a, b))


device = triton.runtime.driver.active.get_active_torch_device()
assert device.type == "cpu"
BLOCK = 16
a = torch.randn(BLOCK, BLOCK, device=device)
b = torch.randn(BLOCK, BLOCK, device=device)
c = torch.empty(BLOCK, BLOCK, device=device)
matmul[(1,)](a, b, c, BLOCK=BLOCK)
target = triton.runtime.driver.active.get_current_target()
print(f"ok: {torch.allclose(c, a @ b, rtol=1e-3, atol=1e-3)} | {device.type} / {target.backend}")
```

```sh
python gemm.py          # ok: True
```

#### RVV

The vector extension is on by default -- `+v` is in the backend's `llc` feature string
(`+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb`; `TRITON_RISCV_LLC_FEATURES` overrides it) -- so
the GEMM above already comes out vectorized. To see it, ask for the RISC-V target explicitly
and count the instructions in the object:

```py
compiled = matmul.warmup(
    a, b, c, BLOCK=BLOCK, grid=(1,),
    target_triple="riscv64-unknown-linux-gnu",   # the backend targets the host by default
)
open("gemm.o", "wb").write(compiled.asm["obj"])
```

```sh
llvm-objdump -d gemm.o | grep -oE "vsetivli|vle32\.v|vse32\.v|vfmacc\.vf|vfadd\.vf" | sort | uniq -c
```

```
      1 vfadd.vf
      9 vfmacc.vf
     12 vle32.v
     11 vse32.v
      2 vsetivli
```

#### SpacemiT IME

SpacemiT's integrated matrix extension (IME) adds `vmadot` for integers and `vfmadot` for
fp16, and the backend can lower `tl.dot` straight to them. Set `TRITON_RISCV_USE_IME=1` and
`linalg.matmul` goes through buddy-mlir's IME dialect (`--lower-linalg-to-ime`,
`--lower-ime`) with the object built for `+xsmtime`.

```py
# ime.py
import os
import torch
import triton
import triton.language as tl

os.environ["TRITON_RISCV_USE_IME"] = "1"


@triton.jit
def matmul_i8(a_ptr, b_ptr, c_ptr):
    m = tl.arange(0, 4)
    n = tl.arange(0, 4)
    k = tl.arange(0, 8)
    a = tl.load(a_ptr + m[:, None] * 8 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 4 + n[None, :])
    tl.store(c_ptr + m[:, None] * 4 + n[None, :], tl.dot(a, b, out_dtype=tl.int32))


a = torch.randint(-8, 8, (4, 8), dtype=torch.int8)
b = torch.randint(-8, 8, (8, 4), dtype=torch.int8)
c = torch.empty((4, 4), dtype=torch.int32)
open("ime.o", "wb").write(matmul_i8.warmup(a, b, c, grid=(1,)).asm["obj"])
```

```sh
llvm-objdump -d ime.o | grep -E "vmadot|vle8|vsetvli"
```

```
  80: vsetvli  zero, a1, e8, m4, ta, ma
  88: vle8.v   v8, (s1)
 13c: vsetvli  zero, a7, e8, m4, ta, ma
 140: vle8.v   v8, (s6)
 144: vle8.v   v12, (a2)
 148: vsetvli  zero, a7, e32, m4, ta, ma
 150: vmadot   v16, v8, v12
 26a: vsetvli  zero, a0, e8, m4, ta, ma
 26e: vle8.v   v8, (t0)
```

### AMD GPU

```py
# gemm.py
import torch
import triton
import triton.language as tl


@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs[:, None] * BLOCK + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * BLOCK + offs[None, :])
    tl.store(c_ptr + offs[:, None] * BLOCK + offs[None, :], tl.dot(a, b))


device = triton.runtime.driver.active.get_active_torch_device()
assert device.type == "cuda"   # ROCm hands out a cuda device, like CUDA does
BLOCK = 16
a = torch.randn(BLOCK, BLOCK, device=device)
b = torch.randn(BLOCK, BLOCK, device=device)
c = torch.empty(BLOCK, BLOCK, device=device)
matmul[(1,)](a, b, c, BLOCK=BLOCK)
target = triton.runtime.driver.active.get_current_target()
print(f"ok: {torch.allclose(c, a @ b, rtol=1e-3, atol=1e-3)} | {device.type} / {target.backend}")
```

If torch and ROCm are installed, Triton finds the AMD GPU on its own; the wheel ships
neither, so both have to be installed.

To force the CPU backend, set `TRITON_DEFAULT_BACKEND=triton_shared`.

## Generate and run RVV ELF files with QEMU

The backend can compile a Python Triton kernel directly to a self-contained RISC-V Linux ELF, automatically generate its runner, execute it with QEMU, and verify outputs. No user-written C runner is required.

```sh
source scripts/env.sh
python python/examples/rvv_vec_add_elf.py
```

See [docs/06-RISCV-QEMU.md](docs/06-RISCV-QEMU.md) for toolchain setup, the end-to-end API, output verification, and troubleshooting.

See [docs/07-Optimization.md](docs/07-Optimization.md) for the current code-generation bottlenecks, correctness gaps, and optimization priorities identified from the RVV and bufferization analysis.

## FAQ

Building on RISC-V often runs into dependency issues; we're happy to help if you run into trouble.

### Fortran library mismatch when building PyTorch

Preload the correct Fortran library by adding the following to your virtualenv’s `activate` script (e.g. `~/triton/.venv/bin/activate`):

```sh
export LD_PRELOAD=/usr/lib64/libgfortran.so.5:$LD_PRELOAD
```

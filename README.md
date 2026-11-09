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
export LLVM_BINARY_DIR=/path/to/llvm/bin
```

### From source

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git
cd triton-riscv
make setup    # creates .venv, downloads the pinned LLVM and buddy builds, clones the pinned Triton
make build    # builds Triton and installs the backend into that .venv
make test     # the plugin's lit tests and the example suite
```

What you end up with is an editable Triton whose backend is
`triton.backends.triton_shared`. It takes over when no GPU backend is present,
so `import triton` behaves as usual.

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

`triton.jit` reads the kernel's source, so kernels run from files like this one; a heredoc or
a REPL will not do. On `riscv64` the CPU backend selects itself, so no `set_active` call or
extra environment variable is needed. The rest of the suite lives in `python/examples/`.

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
`--lower-ime`) with the object built for `+xsmtime`. The tile it wants is 4x4 with a k of 8
for int8 and 4 for fp16 -- the int8 GEMM in `python/examples/smtime/test_smtime.py` is the
smallest one that hits it:

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

The kernel only *runs* where the hardware implements the extension; elsewhere it is an
illegal instruction.

### AMD GPU

The same program runs on an AMD GPU with nothing to change: the device line asks the active
backend for its device, which is the GPU on a machine with ROCm, where the AMD backend
activates on its own (`is_active()` asks torch for `torch.version.hip`). These are ordinary
Triton kernels -- upstream's tutorials are written the same way.

On a `riscv64` host one thing does need saying out loud: the CPU backend reports itself as
active there too, so choose the GPU explicitly with `TRITON_DEFAULT_BACKEND=amd` (or
`triton.runtime.driver.set_active(HIPDriver())`).

The wheel does not ship the ROCm runtime: it needs a working ROCm installation and an AMD
GPU.

## Clone

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git
cd triton-riscv
make
```

`make setup` creates `.venv`, clones the Triton revision pinned in
`triton-hash.txt`, and downloads the LLVM and buddy build pinned in
`buddy-hash.txt`; `make build` then builds and installs the backend into it.
Triton itself is used **unpatched**: the pinned LLVM is simply built with the
NVPTX and AMDGPU targets upstream Triton expects.

## Prerequisites

`make setup` handles the rest: virtualenv, Python build dependencies and
toolchains. A Python 3.11+ interpreter is the only system requirement; the RVV
cross toolchain and QEMU that `scripts/build-riscv-gnu-toolchain.sh` builds are
only needed by the [QEMU example](docs/06-RISCV-QEMU.md).

## Environment helper

`scripts/env.sh` exports what the backend and the lit tests need in order to find
the toolchains and the staged `triton-shared-opt`, so source it before running
anything by hand:

```sh
source scripts/env.sh
```

Its defaults follow the layout `make` creates -- `triton/`, `.venv`,
`.cache/llvm`, `.cache/buddy` -- and it prepends the venv, LLVM and buddy `bin`
directories to `PATH`. Override `TRITON_DIR`, `TRITON_VENV`, `LLVM_SYSPATH` or
`BUDDY_MLIR_BINARY_DIR` before sourcing to point it somewhere else.

## Build

```sh
cd /path/to/triton-riscv
make setup
make build
make test
```

## Verify the build

`make build` ends by printing the installed Triton version, the
`triton.backends.triton_shared` module it loaded and the version of the
`triton-shared-opt` executable the backend shells out to. To check by hand:

```sh
source scripts/env.sh
python -c "import triton; import triton.backends.triton_shared.compiler as c; print(triton.__version__); print(c.__file__)"
python -c "from triton.backends.triton_shared.paths import _get_triton_shared_opt_path as f; print(f())"
```

`triton-shared-opt` has no console script: the backend runs the copy bundled in
the package, or whatever `TRITON_SHARED_OPT_PATH` points at when set.

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

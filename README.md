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

## Install from source

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git
cd triton-riscv
make
make test
```

What you end up with is an editable Triton whose backend is
`triton.backends.triton_shared`. It takes over when no GPU backend is present,
so `import triton` behaves as usual.

## Quick start

```py
# add.py
import torch
import triton
import triton.language as tl


@triton.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


n = 4096
x, y = torch.randn(n), torch.randn(n)
out = torch.empty_like(x)
add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
print("ok:", torch.allclose(out, x + y))

# The backend targets the host by default; ask for RISC-V and keep the object so
# the instructions it emitted can be inspected.
compiled = add.warmup(
    x, y, out, n, BLOCK=1024, grid=(triton.cdiv(n, 1024),),
    target_triple="riscv64-unknown-linux-gnu",
)
open("add.o", "wb").write(compiled.asm["obj"])
```

```sh
python add.py
llvm-objdump -d add.o | grep -E "vset|vle32|vse32|vfadd"   # or "$LLVM_BINARY_DIR/llvm-objdump"
```

The instructions it prints are the RVV ones:

```
      42: cd287057      vsetivli    zero, 0x10, e32, m4, ta, ma
      46: 020ee407      vle32.v     v8, (t4)
      4e: 020ee607      vle32.v     v12, (t4)
      52: 02861457      vfadd.vv    v8, v8, v12
      60: 020ee427      vse32.v     v8, (t4)
```

## How to

One wheel, three ways to use it, all of them on RISC-V:

- **RISC-V CPU.** On `riscv64` the backend selects itself and the kernels run on the CPU --
  that is what the example above does. The rest of the suite lives in `python/examples/`.
- **RISC-V with the vector and matrix extensions.** RVV is on by default (the backend's
  `llc` feature string is `+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb`), and
  `TRITON_RISCV_USE_IME=1` compiles `tl.dot` to the integrated matrix extension's
  `vmadot` / `vfmadot` (see `python/examples/smtime/test_smtime.py`). Either one only *runs*
  where the hardware has the extension.
- **RISC-V host with an AMD GPU.** The wheel carries the AMD backend and the codegen
  library it needs, so a kernel compiles to `hsaco` here. Running it needs ROCm for
  riscv64, which does not exist yet.

The backend can also emit a self-contained RISC-V ELF to be run elsewhere, without a
toolchain on the target -- see [docs/06-RISCV-QEMU.md](docs/06-RISCV-QEMU.md) and
`python/examples/rvv_vec_add_elf.py`.

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

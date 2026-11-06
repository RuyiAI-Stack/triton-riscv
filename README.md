# triton-riscv

Triton compiler for RISC-V platforms.

This repository is forked from [triton-shared](https://github.com/microsoft/triton-shared) and provides a Triton compiler backend for RISC-V. The upstream triton-shared repo is no longer maintained, so this project is developed independently under the name triton-riscv.

Triton acts as a frontend (Python AST → TTIR); triton-riscv handles the rest of the pipeline (TTIR → Linalg → LLVM IR → native object). No NVIDIA or AMD toolchain is required.

**Compilation pipeline:**

```
Python @triton.jit
  └─► TTIR          (triton: ast_to_ttir + make_ttir passes)
        └─► Linalg MLIR   (triton-shared-opt --triton-to-linalg-experimental)
              └─► LLVM MLIR  (Buddy Compiler lowering passes)
                    └─► LLVM IR  (mlir-translate --mlir-to-llvmir)
                          └─► native .o  (llc -filetype=obj)
```

## Install

Every release publishes `cp312-abi3` wheels for `riscv64` and `x86_64` and
uploads them to our package index: install the `riscv64` one on the machine that
runs the kernels, or the `x86_64` one to try the backend out on a normal CPU. The
wheel carries the backend, its `triton.backends` entry point and
`triton-shared-opt`; the LLVM and buddy executables it compiles with are **not**
bundled.

```sh
python3 -m venv .venv && . .venv/bin/activate
# The name is just `triton`, but it has to come from the index that carries it:
# PyPI only holds upstream Triton. Every release attaches the wheels as well.
pip install --index-url https://forgestash.artifact.yuantu.ac.cn/repository/pypi-riscv-local-ruyiai/simple/ triton
pip install torch --index-url https://community-ci.openruyi.cn/pypi/riscv64/dev/+simple/torch  # optional

# Unpack both archives of the buddy-mlir release pinned in buddy-hash.txt; they
# expand to <dir>/bin, which is the layout the backend expects.
mkdir -p /path/to/llvm /path/to/buddy
tar xf llvm-*.tar.gz  -C /path/to/llvm     # opt, mlir-translate, llc
tar xf buddy-*.tar.gz -C /path/to/buddy    # buddy-opt, buddy-llc, buddy-translate
export LLVM_BINARY_DIR=/path/to/llvm/bin          # tools are also found on PATH
export BUDDY_MLIR_BINARY_DIR=/path/to/buddy/bin   # only needed for the IME path
```

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
```

```sh
python add.py     # ok: True
```

`triton.jit` reads the kernel's source, so the kernel has to live in a file -- a
heredoc or a REPL will not work. On `riscv64` the CPU backend selects itself; no
`set_active` call or extra environment variable is needed.

Requires Python 3.12 to 3.15, in a regular (GIL) build: free-threaded builds
cannot load an `abi3` wheel. Python 3.11 has no wheel.
## Quick start (from source)

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git
cd triton-riscv
make            # virtualenv, toolchains, Triton and the backend; ~15 minutes the first time
make test       # the example suite, the plugin's lit tests and the tutorials
```

What you end up with is an editable Triton whose backend is
`triton.backends.triton_shared`. It takes over when no GPU backend is present,
so `import triton` behaves as usual, and the tutorials in
[`python/tutorials/`](python/tutorials) run anywhere a CPU does.

## Clone

To build from source instead, continue here:

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git triton-riscv
TRITON_RISCV_DIR="$(pwd)/triton-riscv"
```

`make setup` clones the Triton revision pinned in `triton-hash.txt` and
downloads the LLVM and buddy tools published under `buddy-hash.txt`, so neither
needs a manual checkout.

> Note: Ensure PyTorch is available in your virtual environment. For RISC-V, since PyTorch does not officially support RISC-V yet, you can build from source or use third-party builds: https://community-ci.openruyi.cn/pypi/riscv64/dev/+simple/torch . To use the third-party builds, python 3.12 or 3.13 are required.

Triton is used **unpatched**. Building the stock Triton checkout on RISC-V
requires the LLVM it links against to provide the NVPTX and AMDGPU codegen
libraries plus `lld` (upstream Triton hard-codes them), and it requires skipping
the host-specific GPU toolchains Triton downloads at build time (nvcc, AMD
codegen, proton), which do not exist for `riscv64`:

- The buddy-mlir LLVM release published under `buddy-hash.txt` is built with
  those extra targets; `make setup` downloads it into `.cache/llvm` (set
  `LLVM_SYSPATH` to use an existing LLVM instead).
- `scripts/_gpu-stub-env.sh` points every GPU download location at an
  empty stub directory and disables the components that need CUDA/ROCm headers.

## Prerequisites

### Create a virtual environment

Use either a Python venv or an existing conda environment:

```sh
python -m venv "${TRITON_RISCV_DIR}/.venv" --prompt triton-riscv
source "${TRITON_RISCV_DIR}/.venv/bin/activate"
```

Or with conda:

```sh
conda activate ruyiai
```

When a conda environment is active, `scripts/env.sh` automatically uses `CONDA_PREFIX` as `TRITON_VENV`.

### Install dependencies

Nothing to install: `make setup` brings up the virtualenv's build
dependencies (cmake, ninja, nanobind, setuptools, pytest) and downloads the LLVM
and buddy tools Triton's build links against. The only system requirement is a
Python 3.10+ interpreter. The one thing you may have to provide yourself is the
RVV cross toolchain and QEMU used by `scripts/build-riscv-gnu-toolchain.sh`
(see [docs/06-RISCV-QEMU.md](docs/06-RISCV-QEMU.md)).

## Environment helper

[`scripts/env.sh`](scripts/env.sh) centralizes the local environment variables needed by `triton-riscv`.

By default it assumes:

- `TRITON_RISCV_DIR=/path/to/triton-riscv`
- `TRITON_DIR=$TRITON_RISCV_DIR/triton` (where `make setup` clones Triton)
- `TRITON_VENV=$CONDA_PREFIX` when a conda environment is active, otherwise `$TRITON_RISCV_DIR/.venv`
- `LLVM_SYSPATH=$TRITON_RISCV_DIR/.cache/llvm`, the LLVM `scripts/build.sh` downloaded; override it to use another multi-target LLVM
- `BUDDY_MLIR_BINARY_DIR=$TRITON_RISCV_DIR/.cache/buddy/bin`, likewise
- `TRITON_SHARED_OPT_PATH` (optional) overrides where the backend looks for the tool; by default it uses the copy bundled in the package, which `scripts/build.sh` stages from the build tree just like `scripts/release.sh` does for the wheel
- Triton's cache, dump and override directories default to `~/.triton`

It exports:

- `TRITON_RISCV_DIR`, `TRITON_DIR`, `TRITON_VENV`
- `LLVM_SYSPATH`, `LLVM_BINARY_DIR`, `BUDDY_MLIR_BINARY_DIR`
- `TRITON_HOME`, `TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`, `TRITON_OVERRIDE_DIR`
- `TRITON_SHARED_DUMP_PATH`

It also prepends the venv, LLVM and buddy `bin` directories to `PATH`, sets
`PYTHONSAFEPATH=1` so the in-tree `triton/` checkout cannot shadow the installed
package, and disables the host-specific GPU toolchain downloads Triton would
otherwise attempt.

If your local layout differs, override variables before sourcing:

```sh
export TRITON_DIR=/path/to/triton
export TRITON_VENV=/path/to/triton-venv   # or rely on active conda env
source /path/to/triton-riscv/scripts/env.sh
```

This helper removes most of the repetitive local environment setup from the README, but it does not replace the external prerequisites themselves: a virtual environment, a Triton checkout, and the LLVM/buddy toolchains (`make setup` fetches all three).

## Build and rebuild

`make` bootstraps a fresh checkout and builds it; afterwards `make build`
rebuilds after a change:

```sh
cd /path/to/triton-riscv
make
```

What the targets do:

| Target | |
|---|---|
| `make` | `make setup` then `make build` |
| `make setup` | Virtualenv, Python build dependencies, the pinned Triton checkout, and the LLVM and buddy toolchains |
| `make build` | Builds Triton with the triton-shared plugin and installs the `triton.backends.triton_shared` backend |
| `make test` | Runs every suite below |
| `make clean` | Removes the Triton build tree |

`make setup` is idempotent, so running it again only picks up changes to
`buddy-hash.txt` or `triton-hash.txt`; `make build` never touches the Triton
checkout, so a rebuild is cheap.

If you change the buddy-opt pass pipeline or cached compilation behavior, clear
Triton's cache before rerunning tests:

```sh
rm -rf "$TRITON_CACHE_DIR"
```

Build artifacts are placed under `triton/build/{current_cmake_version}/third_party/triton_shared`.

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

## Run the example test suite

```sh
make test   # the example suite, plus the flaggems suite on riscv64
make test-lit    # the plugin's lit tests
```

`make test` runs `pytest python/examples`, skipping `test_core.py`,
`test_annotations.py` and `flaggems`; on riscv64 it then runs the flaggems suite,
which is ~3835 tests. To run a single test:

```sh
bash -c 'source scripts/env.sh && pytest python/examples/test_vec_add.py -v'
```

## Tutorials

[`python/tutorials/`](python/tutorials) holds the notebooks, one per RISC-V
feature:

- `02-rvv-vectorization` compiles one kernel twice -- once with the RISC-V vector
extension and once without -- and disassembles both, so you can see the
`vle32.v` / `vfadd.vv` / `vse32.v` the backend emits;
- `03-ime-matmul` lowers `tl.dot` to the integrated matrix extension's `vmadot`
and `vfmadot` when `TRITON_RISCV_USE_IME=1`.

Both compile RISC-V code on any host and are executed by CI. To run them locally:

```sh
make test-tutorials
```

## Generate and run RVV ELF files with QEMU

The backend can compile a Python Triton kernel directly to a self-contained RISC-V Linux ELF, automatically generate its runner, execute it with QEMU, and verify outputs. No user-written C runner is required.

```sh
source scripts/env.sh
python python/examples/rvv_vec_add_elf.py
```

See [docs/06-RISCV-QEMU.md](docs/06-RISCV-QEMU.md) for toolchain setup, the end-to-end API, output verification, and troubleshooting.

See [docs/07-Optimization.md](docs/07-Optimization.md) for the current code-generation bottlenecks, correctness gaps, and optimization priorities identified from the RVV and bufferization analysis.

## Limitations

- The wheels are not self-contained: they need an external LLVM, and the buddy
tools for the IME path (see [Install](#install)).
- IME kernels compile to `vmadot` / `vfmadot` (`TRITON_RISCV_USE_IME=1`), but they
only run on a CPU that implements the `xsmtime` extension. Elsewhere the process
dies with `Illegal instruction`.
- The generated RVV code assumes `VLEN=128`: the backend vectorizes with a fixed
width (`-riscv-v-vector-bits-min/max=128` in `backend/compiler.py`, matching
`--lower-vir-to-vector=vector-width=4`). That is correct on any `VLEN >= 128`
machine, but a wider one only gets part of the available vector width -- the
RVA23 board used for testing is `VLEN=256`.
- The wheels register upstream's `nvidia` and `amd` backends too, but the
CUDA/ROCm toolchains do not exist for `riscv64`, so only the CPU backend can
activate there.
- The examples and tests live in this repository, not in the wheel.

## FAQ

Building on RISC-V often runs into dependency issues; we're happy to help if you run into trouble.

### Fortran library mismatch when building PyTorch

Preload the correct Fortran library by adding the following to your virtualenv’s `activate` script (e.g. `~/triton/.venv/bin/activate`):

```sh
export LD_PRELOAD=/usr/lib64/libgfortran.so.5:$LD_PRELOAD
```

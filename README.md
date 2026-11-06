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

## Quick start (from binary)

```sh
python3 -m venv .venv && . .venv/bin/activate
# `triton` and the buddy tools come from the index that carries them; PyPI only
# holds upstream Triton. Until that index has them, both projects attach the same
# wheels to their GitHub releases.
pip install --index-url https://forgestash.artifact.yuantu.ac.cn/repository/pypi-riscv-local-ruyiai/simple/ triton buddy
pip install torch --index-url https://community-ci.openruyi.cn/pypi/riscv64/dev/+simple/torch  # optional

# LLVM ships only as an archive (no wheel): it provides opt, mlir-translate and
# llc. The buddy tools are already on the PATH after the install above, so
# BUDDY_MLIR_BINARY_DIR is only needed if you unpack that archive instead.
mkdir -p /path/to/llvm
tar xf llvm-*.tar.gz -C /path/to/llvm
export LLVM_BINARY_DIR=/path/to/llvm/bin          # its tools are also found on PATH
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
make test        # everything: the examples, the plugin's lit tests and the tutorials
make test-lit    # only the plugin's lit tests
```

The examples compare against PyTorch, so it has to be installed in the venv;
on RISC-V the third-party builds work:
`pip install torch --index-url https://community-ci.openruyi.cn/pypi/riscv64/dev/+simple/torch`
(Python 3.12+).

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

## FAQ

Building on RISC-V often runs into dependency issues; we're happy to help if you run into trouble.

### Fortran library mismatch when building PyTorch

Preload the correct Fortran library by adding the following to your virtualenv’s `activate` script (e.g. `~/triton/.venv/bin/activate`):

```sh
export LD_PRELOAD=/usr/lib64/libgfortran.so.5:$LD_PRELOAD
```

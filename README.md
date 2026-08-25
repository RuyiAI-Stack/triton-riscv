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

## Clone

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git triton-riscv
TRITON_RISCV_DIR="$(pwd)/triton-riscv"
git clone https://github.com/triton-lang/triton.git triton
cd triton && git checkout "$(cat "${TRITON_RISCV_DIR}/triton-hash.txt")"
cd "${TRITON_RISCV_DIR}/.."
git clone https://github.com/buddy-compiler/buddy-mlir.git buddy-mlir
cd buddy-mlir && git checkout "$(head -n 1 "${TRITON_RISCV_DIR}/buddy-hash.txt")"
```

> Note: Ensure PyTorch is available in your virtual environment. For RISC-V, since PyTorch does not officially support RISC-V yet, you can build from source or use third-party builds: https://community-ci.openruyi.cn/pypi/riscv64/dev/+simple/torch . To use the third-party builds, python 3.12 or 3.13 are required.

Apply the build-system patches from `triton-riscv/patches/` using the helper script. These patches make the NVIDIA/AMD LLVM codegen libraries conditional on their backends being enabled, so the build succeeds without any GPU toolchain.

```sh
"${TRITON_RISCV_DIR}/scripts/apply_patches.sh" "${TRITON_RISCV_DIR}/../triton"
```

The script is idempotent: re-running it on an already-patched tree prints `SKIPPED (patch already applied)` per patch and exits cleanly.

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

When a conda environment is active, `scripts/triton-riscv-env.sh` automatically uses `CONDA_PREFIX` as `TRITON_VENV`.

### Install dependencies

1. Install the [dependencies](https://github.com/buddy-compiler/buddy-mlir?tab=readme-ov-file#llvmmlir-dependencies) required by the Ruyi Buddy Compiler.

2. Install triton-riscv Python dependencies in the active environment:

   ```sh
   pip install cmake ninja pytest-xdist pybind11 setuptools
   ```

   `cmake`, `ninja`, and `pybind11` are needed because the rebuild script uses `pip install --no-build-isolation` to reuse the active environment instead of creating an isolated build environment.

3. Build the pinned Buddy Compiler checkout — [Getting started](https://github.com/buddy-compiler/buddy-mlir?tab=readme-ov-file#getting-started)

## Environment helper

[`scripts/triton-riscv-env.sh`](scripts/triton-riscv-env.sh) centralizes the local environment variables needed by `triton-riscv`.

By default it assumes:

- `TRITON_RISCV_DIR=/path/to/triton-riscv`
- `TRITON_DIR=$TRITON_RISCV_DIR/../triton`
- `TRITON_VENV=$CONDA_PREFIX` when a conda environment is active, otherwise `$TRITON_RISCV_DIR/.venv`
- `BUDDY_DIR=$TRITON_RISCV_DIR/../buddy-mlir`
- `BUILD_DIR` is auto-detected from `$TRITON_DIR/build/cmake.linux-*-cpython-*`
- runtime cache, dump, and JSON dependency directories default to `~/.triton`

It exports:

- `TRITON_PLUGIN_DIRS`
- `LLVM_SYSPATH`
- `JSON_SYSPATH`
- `LLVM_BINARY_DIR`
- `BUDDY_MLIR_BINARY_DIR`
- `TRITON_SHARED_OPT_PATH`
- `TRITON_HOME`, `TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`, `TRITON_OVERRIDE_DIR`

If your local layout differs, override variables before sourcing:

```sh
export TRITON_DIR=/path/to/triton
export TRITON_VENV=/path/to/triton-venv   # or rely on active conda env
export BUDDY_DIR=/path/to/buddy-mlir
source /path/to/triton-riscv/scripts/triton-riscv-env.sh
```

This helper removes most of the repetitive local environment setup from the README, but it does not replace the external prerequisites themselves: Buddy/LLVM dependencies, a built `buddy-mlir`, a checked-out Triton tree, the virtual environment, and the patch step are still required.

## Build and rebuild

For both the initial build and later source-only rebuilds:

```sh
conda activate ruyiai   # or: source /path/to/triton-riscv/.venv/bin/activate
cd /path/to/triton-riscv
source scripts/triton-riscv-env.sh
scripts/rebuild-triton-riscv.sh
```

[`scripts/rebuild-triton-riscv.sh`](scripts/rebuild-triton-riscv.sh) does the following:

- reuses the environment from `triton-riscv-env.sh`
- removes a stale Triton build directory when it was created under an old path or with a different Python version
- runs `python -m pip install --no-build-isolation -vvv .` in `$TRITON_DIR`, which configures and builds Triton with the current `LLVM_SYSPATH`, `JSON_SYSPATH`, and plugin settings
- syncs `backend/compiler.py` and `backend/driver.py` into the installed `triton_shared` backend package
- verifies the rebuilt install with Python import and `triton-shared-opt --version`

Use the same command sequence after changing the backend, lowering pipeline, or C++ sources.
If you change the buddy-opt pass pipeline or cached compilation behavior, clear Triton's cache after sourcing the environment and before rerunning tests:

```sh
rm -rf "$TRITON_CACHE_DIR"
```

Build artifacts are placed under `triton/build/{current_cmake_version}/third_party/triton_shared`.

## Verify the build

```sh
conda activate ruyiai   # or: source /path/to/triton-riscv/.venv/bin/activate
cd /path/to/triton-riscv
source scripts/triton-riscv-env.sh
python -c "import triton; import triton.backends.triton_shared.compiler as c; print(triton.__version__); print(c.__file__)"
"$TRITON_SHARED_OPT_PATH" --version
```

## Run the example test suite

```sh
pytest python/examples/ \
    --ignore=python/examples/test_core.py \
    --ignore=python/examples/test_annotations.py \
    --ignore=python/examples/flaggems \
    -v
```

To run a single test:

```sh
pytest python/examples/test_vec_add.py -v
```

## Generate and run RVV ELF files with QEMU

The backend can compile a Python Triton kernel directly to a self-contained RISC-V Linux ELF, automatically generate its runner, execute it with QEMU, and verify outputs. No user-written C runner is required.

```sh
source scripts/triton-riscv-env.sh
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

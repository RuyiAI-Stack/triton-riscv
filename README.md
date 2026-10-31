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

Apply the rebased Triton integration patch using the helper script. It configures Triton as a frontend-only build, keeps GPU-specific dependencies conditional, and adds the small frontend APIs required by triton-riscv.

```sh
"${TRITON_RISCV_DIR}/scripts/apply_patches.sh" "${TRITON_RISCV_DIR}/../triton"
```

The script is idempotent: re-running it on an already-patched tree prints `SKIPPED (patch already applied)` per patch and exits cleanly.

The patch set also preserves Triton's existing `attr-dict` textual IR format
when building against LLVM revisions that enable strict properties by default.
It opts Triton's dialects into MLIR's legacy assembly-format behavior without
changing the LLVM checkout. LLVM 24 compatibility covers the instruction
`contract` flags for floating-point fusion, the renamed pass-plugin loader, the
`RegionSuccessor` control-flow API, the `OptionalPassInfoMixin` pass base
class, debug-info attribute signatures, and linking every target provided by
the toolchain LLVM so `init_targets` resolves on any host.
Re-run `scripts/apply_patches.sh` before rebuilding an existing Triton checkout
after updating this repository's patches.

## Supported dependencies

| Component | Version | Authoritative pin |
|---|---|---|
| Triton | official `v3.8.0` release tag | `triton-hash.txt` |
| Buddy MLIR | `nightly/v0.1.3.dev20261027` | `buddy-hash.txt` |
| LLVM toolchain | `24.0.0git` (Buddy prebuilt packages) | Buddy release tag |
| Python | 3.12, 3.13, or 3.14 | — |
| pybind11 | `>=2.13.1` | build environment |
| torch | 2.x CPU wheel (x86_64); third-party RISC-V builds | — |

Notes:

- Triton is integrated at the official `v3.8.0` release revision — not a
  mainline development commit — and the integration patches under `patches/`
  are written against that revision.
- Triton 3.8.0 predates the upstream nanobind migration, so both Triton and
  the triton-riscv plugin bind with **pybind11**; `nanobind` is not needed.
- Buddy ships `cp312-abi3` wheels and prebuilt tarballs, so Python 3.12 or
  newer is required. The third-party RISC-V torch builds support Python
  3.12/3.13.

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
   pip install cmake ninja pytest-xdist "pybind11>=2.13.1" setuptools
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

When using the prebuilt toolchain packages fetched by
[`scripts/build.sh`](scripts/build.sh) (`.cache/llvm`, `.cache/buddy`), point
the helper at them before sourcing — its defaults assume a source-built Buddy
checkout layout:

```sh
export BUDDY_MLIR_BINARY_DIR=$TRITON_RISCV_DIR/.cache/buddy/bin
export LLVM_SYSPATH=$TRITON_RISCV_DIR/.cache/llvm
source scripts/triton-riscv-env.sh
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

[`scripts/build.sh`](scripts/build.sh) is the automated one-click path used by
CI: it clones/updates the pinned Triton checkout, applies the patch stack,
downloads the prebuilt Buddy/LLVM packages into `.cache/`, and builds and
installs Triton into `.venv` (override the interpreter with `PYTHON=` and the
parallelism with `MAX_JOBS=`). On a fresh machine it is the fastest
reproducible path and needs no manual clone/patch steps.

The editable Triton install keeps a single `libtriton.so` in the source tree,
shared by every virtual environment pointing at that checkout. After building
with a different Python interpreter, rebuild before running tests from another
environment.

## Verify the build

```sh
conda activate ruyiai   # or: source /path/to/triton-riscv/.venv/bin/activate
cd /path/to/triton-riscv
source scripts/triton-riscv-env.sh
python -c "import triton; import triton.backends.triton_shared.compiler as c; print(triton.__version__); print(c.__file__)"
"$TRITON_SHARED_OPT_PATH" --version
```

## Run the test suites

All suites expect the environment from `scripts/triton-riscv-env.sh` to be
sourced (see the prebuilt-toolchain note above when `.cache` packages are
used). Clear the compilation cache before comparing runs:

```sh
rm -rf "$TRITON_CACHE_DIR"
```

### lit tests (MLIR middle layer)

```sh
cmake --build "$TRITON_DIR/build/cmake.linux-$(uname -m)-cpython-<python-version>" \
      --target check-triton-shared-lit-tests
```

For example, on an x86_64 host with Python 3.12:
`triton/build/cmake.linux-x86_64-cpython-3.12`.

Reference result on x86_64 at the current pins: 301 tests — 292 passed, 9
expected failures.

### Example suite (end-to-end compilation + CPU execution)

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

On hosts without native RISC-V execution, the runtime-execution variants
(`*_matches_torch`, tail-shape cases, and similar) are skipped; only the
cross-compilation checks run.

### FlagGems operator suite

Run the FlagGems directory separately from the common suite. A few files
crash the whole test process — these are known, pre-existing issues tracked
separately from the Triton integration:

- `test_affine_grid_generator.py` — segfault in the kernel launch path
- `test_avg_pool2d.py`, `test_avg_pool3d.py` — heap-corruption aborts

Because the crashes are process-level, run the suite file-by-file so one
crash does not kill the whole run:

```sh
cd "$TRITON_RISCV_DIR"
for f in python/examples/flaggems/test_*.py; do
    ( cd triton && ../.venv/bin/python -m pytest "../$f" -q ) \
        || echo "FAILED: $f"
done
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

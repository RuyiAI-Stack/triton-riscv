# Getting Started

> **Audience:** first-time Triton-RISCV users. No Triton, MLIR, or RISC-V
> compiler knowledge is required.
>
> **Prerequisites:** Ubuntu 24.04, Bash, an `x86_64` or `riscv64` host, and
> Python 3.12–3.14. These are the platforms currently covered by CI.
>
> **Outcome:** run vector addition through the Host CPU backend—the compiler
> path that emits code for the current machine—and prove that its output
> matches PyTorch. RISC-V/QEMU is a separate, optional next step.

This document has two independent parts:

- **15-Minute Quick Start:** the shortest CI-like path using downloaded
  Buddy/LLVM packages. Beginners should stop after this part.
- **Build from Source:** an advanced setup for compiler contributors who need
  editable Buddy/LLVM checkouts or the RISC-V GNU toolchain.

## 15-Minute Quick Start

The commands are intentionally linear: run them top to bottom in one shell.
Initial compiler-package downloads can make the wall-clock time longer than 15
minutes on a slow network, but no source build of LLVM or Buddy is required.

### Step 1: Confirm the Host

```sh
uname -m
python3 --version
```

Success means the architecture is `x86_64` or `riscv64` and Python is 3.12,
3.13, or 3.14. The commands below show the common `x86_64` PyTorch install;
native `riscv64` uses the alternative noted in Step 4.

### Step 2: Install System Packages

```sh
sudo apt-get update
sudo apt-get install -y git curl clang lld ninja-build ccache python3-venv
```

Success means `apt-get` exits with status 0. `clang`, `lld`, and Ninja are used
while building Triton and its launcher; the full LLVM/Buddy tools are downloaded
by the repository build script.

### Step 3: Clone and Build

```sh
git clone https://github.com/RuyiAI-Stack/triton-riscv.git
cd triton-riscv
./scripts/build.sh
```

The script:

1. creates `.venv/`;
2. clones upstream Triton into `triton/`;
3. checks out the exact commit in `triton-hash.txt`;
4. applies the patches under `patches/`;
5. downloads matching LLVM and Buddy packages into `.cache/`;
6. builds and installs Triton with this repository as a backend plugin.

If `triton/` already exists, the script resets that nested checkout to the
pinned commit. Do not keep unrelated uncommitted work there.

Because build output is verbose, use file checks as the success criterion:

```sh
(
  set -e
  test -x .venv/bin/python
  test -x .cache/llvm/bin/llc
  test -x .cache/buddy/bin/buddy-opt
  echo "PASS: prebuilt Triton/Buddy/LLVM layout is present"
)
```

Expected final line:

```text
PASS: prebuilt Triton/Buddy/LLVM layout is present
```

### Step 4: Install PyTorch

Activate the environment created by the build:

```sh
source .venv/bin/activate
```

On `x86_64`, install the upstream CPU wheel:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On a native `riscv64` host, CI currently uses:

```sh
python -m pip install torch==2.10 \
  --index-url https://ruyirepo.ruyicommunity.cn/pypi/simple
```

PyTorch is the tensor container and correctness reference in this tutorial.
Calling PyTorch for the expected result is **not** a Triton-RISCV fallback and
does not count as compiler support for an operation.

Verify the installation:

```sh
python -c "import torch; print(torch.__version__); print(torch.empty(1).device)"
```

Success prints a PyTorch version and `cpu`.

### Step 5: Export the Prebuilt-Layout Paths

Run these exports in every new shell after activating `.venv`:

```sh
cd /path/to/triton-riscv
source .venv/bin/activate

export TRITON_PLUGIN_DIRS="$PWD"
export LLVM_SYSPATH="$PWD/.cache/llvm"
export LLVM_BINARY_DIR="$LLVM_SYSPATH/bin"
export BUDDY_MLIR_BINARY_DIR="$PWD/.cache/buddy/bin"
```

Do not source `scripts/triton-riscv-env.sh` for this in-repository layout unless
you also override `TRITON_DIR` and `BUDDY_DIR`. That helper defaults to the
sibling source-build layout described later.

### Step 6: Verify Backend Discovery

```sh
python - <<'PY'
import triton
from triton.backends.triton_shared import compiler
from triton.backends.triton_shared.paths import _get_triton_shared_opt_path

print("Triton version:", triton.__version__)
print("Backend module:", compiler.__file__)
print("Optimizer:", _get_triton_shared_opt_path())
PY
```

Success criteria:

- the backend module path contains `triton_shared`;
- the optimizer path ends in `triton-shared-opt`;
- the command exits without an import or missing-tool exception.

### Step 7: Run and Strictly Check Host CPU Vector Add

This command reuses `add()` from `python/examples/test_vec_add.py`, explicitly
selects `CPUDriver`, and checks the result with `torch.testing.assert_close`.
`CPUDriver` is the runtime adapter that loads and launches the native Host CPU
object:

```sh
PYTHONPATH="$PWD/python/examples${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import torch
import benchmark
from test_vec_add import add

benchmark.select_cpu_backend()
torch.manual_seed(0)
x = torch.rand(257, device="cpu")
y = torch.rand(257, device="cpu")
expected = x + y
actual = add(x, y)
torch.testing.assert_close(actual, expected)
max_error = torch.max(torch.abs(actual - expected)).item()
print(f"PASS: Host CPU vector-add matches PyTorch; max error = {max_error}")
PY
```

Expected output ends with:

```text
PASS: Host CPU vector-add matches PyTorch; max error = 0.0
```

The exact maximum can be a tiny floating-point value on a different toolchain;
success is the `torch.testing.assert_close` check completing and the final
`PASS` line being printed.

> **This is Host CPU execution, not RISC-V/QEMU.** `CPUDriver` asks the backend
> to emit a native object for the current machine, loads it, and launches the
> kernel. No RISC-V ELF is created by this command.

The same Triton kernel has two later code-generation routes:

```text
                         same @triton.jit kernel
                                    |
                                   TTIR
                                    |
                         Structured / Vector MLIR
                                    |
                                 LLVM IR
                           +--------+---------+
                           |                  |
                    Host CPU route       RISC-V route
                           |                  |
                    native .o          RISC-V .o + runner
                           |                  |
                     CPUDriver          static RISC-V ELF
                           |                  |
                    current CPU          QEMU or hardware
```

To run the repository's original pytest wrapper as CI does:

```sh
python -m pytest python/examples/test_vec_add.py -v -s
```

Success prints the expected/actual tensors, a small maximum difference, and a
`PASSED` status. The historical pytest function prints the difference but does
not itself assert it; the standalone command above is the strict correctness
check for this Quick Start.

## What Happened During Vector Add

The main kernel in `python/examples/test_vec_add.py` is:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

Read it in this order:

1. **`@triton.jit`** marks a Python function as a Triton kernel. It is
   specialized and compiled when launched with concrete argument types and
   `tl.constexpr` values.
2. **`tl.program_id(0)`** identifies one program instance. A Triton launch grid
   creates many instances; each handles a different block of the vector.
3. **`tl.arange(0, BLOCK_SIZE)`** creates the element positions handled by one
   instance. Adding `block_start` turns them into global vector offsets.
4. **`mask = offsets < n_elements`** disables lanes beyond the real vector end.
   The Quick Start uses 257 elements with a block size of 1024 specifically to
   exercise this masked tail.
5. **`tl.load` and `tl.store`** read and write the valid lanes. The middle
   expression performs elementwise addition.

The Python wrapper calculates the grid:

```python
def grid(meta):
    return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
```

`triton.cdiv` is ceiling division. With 257 elements and `BLOCK_SIZE=1024`, the
grid contains one program instance. With 2048 elements it would contain two.

Outside pytest, this line selects the runtime route:

```python
triton.runtime.driver.set_active(CPUDriver())
```

A **backend** is the compiler plugin that produces an object file. A **driver**
is the runtime adapter that loads and launches that object. The repository's
`python/examples/conftest.py` selects `CPUDriver` automatically for pytest; a
standalone program must select it explicitly, as the Quick Start does through
`benchmark.select_cpu_backend()`.

Finally, the strict check compares the compiled kernel's output with the
independent PyTorch expression:

```python
expected = x + y
actual = add(x, y)
torch.testing.assert_close(actual, expected)
```

At a high level, TTIR is the compiler's first form of the Triton kernel;
Structured MLIR uses Linalg for structured computation and MemRef for shaped
memory views; the Vector Dialect makes vector work explicit; and LLVM IR is the
low-level input to machine-code generation. The compiler transforms the kernel
through these forms:

| Stage | Plain-language meaning | What vector add looks like there |
| --- | --- | --- |
| TTIR | Triton's first compiler form | Triton load, add, store, grid, and mask operations |
| Structured MLIR | computation and memory shapes made explicit | Linalg-style computation and MemRef-style memory views |
| Vector MLIR | vector work made explicit | vector/loop loads, addition, and stores |
| LLVM IR | low-level code before machine code | pointer arithmetic, loads, arithmetic, stores, and calls |
| object | machine code plus relocation data | native host object in this Quick Start |

More precisely, **Linalg** is MLIR's structured computation dialect.
**MemRef** describes a shaped memory view including element type, offsets, and
strides. The **Vector Dialect** represents explicit portable vector operations.
**LLVM IR** is the low-level representation consumed by LLVM's object-code
generator.

You do not need to understand those dialects to use the backend. Kernel
developers can inspect them in [Inspecting IR](03-IR.md); compiler contributors
can continue to [Implementation details](02-Implementation.md).

## Quick Start Complete

At this point you have:

- built the CI-like prebuilt configuration;
- selected the Host CPU backend;
- compiled and launched a Triton kernel;
- exercised a masked tail;
- checked the result against PyTorch;
- learned where the Host CPU and RISC-V routes diverge.

Beginners should read [Project overview](01-Overview.md) next. Try the
[minimal RISC-V/QEMU tutorial](06-RISCV-QEMU.md#minimal-risc-vqemu-tutorial)
only after its cross-toolchain prerequisites are available.

---

## Build from Source

> **Audience:** compiler contributors who need to edit Buddy/LLVM, control
> dependency revisions, or build the RISC-V GNU toolchain and QEMU.
>
> **Not required for the first Host CPU example.**

The source layout is different from the Quick Start layout:

```text
workspace/
  triton-riscv/
  triton/
  buddy-mlir/
```

### Install Source-Build Dependencies

```sh
sudo apt-get update
sudo apt-get install -y \
  build-essential git curl clang lld cmake ninja-build ccache python3-venv \
  flatbuffers-compiler libflatbuffers-dev libnuma-dev
```

LLVM's complete prerequisite list varies by distribution. Consult the current
[LLVM getting-started guide](https://llvm.org/docs/GettingStarted.html#software)
and [Buddy README](https://github.com/buddy-compiler/buddy-mlir#dependencies)
if CMake reports another missing library.

### Clone the Three Repositories

```sh
mkdir -p workspace
cd workspace

git clone https://github.com/RuyiAI-Stack/triton-riscv.git
git clone https://github.com/triton-lang/triton.git
git -C triton checkout "$(cat triton-riscv/triton-hash.txt)"

git clone https://github.com/buddy-compiler/buddy-mlir.git
git -C buddy-mlir submodule update --init llvm
```

Do not use an arbitrary Triton `main` commit. Plugin APIs and MLIR dialects
change frequently; `triton-hash.txt` is the compatibility contract.

### Create One Python Environment

```sh
cd triton-riscv
python3 -m venv .venv --prompt triton-riscv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cmake ninja pytest pytest-xdist pybind11 setuptools
cd ../buddy-mlir
python -m pip install -r requirements.txt
```

Install PyTorch in the same environment using the command appropriate for the
host from Quick Start Step 4.

### Build LLVM/MLIR and Buddy

```sh
cd /path/to/workspace/buddy-mlir

cmake -G Ninja -S llvm/llvm -B llvm/build \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"

ninja -C llvm/build

cmake -G Ninja -S . -B build \
  -DMLIR_DIR="$PWD/llvm/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm/build/lib/cmake/llvm" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"

ninja -C build
```

Success criteria:

```sh
(
  set -e
  test -x llvm/build/bin/llc
  test -x build/bin/buddy-opt
  echo "PASS: source-built LLVM and Buddy tools are present"
)
```

Reduce Ninja parallelism, for example `ninja -C llvm/build -j4`, if the machine
runs out of memory.

### Patch and Build Triton with the Plugin

```sh
cd /path/to/workspace/triton-riscv
source .venv/bin/activate

scripts/apply_patches.sh ../triton
source scripts/triton-riscv-env.sh
scripts/rebuild-triton-riscv.sh
```

The patch script is idempotent. The rebuild script installs Triton, synchronizes
`compiler.py`, `driver.py`, and `riscv.py`, checks imports, and runs
`triton-shared-opt --version`.

A successful rebuild ends by printing a Triton version, the installed backend
path, and the optimizer version. For future source changes, repeat:

```sh
cd /path/to/workspace/triton-riscv
source .venv/bin/activate
source scripts/triton-riscv-env.sh
scripts/rebuild-triton-riscv.sh
```

## Source-Build Environment Reference

For the sibling layout, `scripts/triton-riscv-env.sh` sets:

| Variable | Default purpose |
| --- | --- |
| `TRITON_RISCV_DIR` | this repository |
| `TRITON_DIR` | sibling `../triton` checkout |
| `BUDDY_DIR` | sibling `../buddy-mlir` checkout |
| `TRITON_VENV` | active conda environment or `.venv` |
| `LLVM_SYSPATH` | `$BUDDY_DIR/llvm/build` |
| `LLVM_BINARY_DIR` | `$LLVM_SYSPATH/bin` |
| `BUDDY_MLIR_BINARY_DIR` | `$BUDDY_DIR/build/bin` |
| `BUILD_DIR` | Triton CMake build matching the active Python |
| `TRITON_SHARED_OPT_PATH` | built `triton-shared-opt` executable |
| `TRITON_CACHE_DIR` | compiled-kernel cache under `~/.triton` |
| `TRITON_SHARED_DUMP_PATH` | middle-layer IR dumps under `~/.triton/dump` |

Override non-default paths before sourcing:

```sh
export TRITON_DIR=/custom/path/to/triton
export BUDDY_DIR=/custom/path/to/buddy-mlir
export TRITON_VENV=/custom/path/to/venv
source scripts/triton-riscv-env.sh
```

[Repository README](../README.md) · [Next for beginners: Project overview](01-Overview.md)

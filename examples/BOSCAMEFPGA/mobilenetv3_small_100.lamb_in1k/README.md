# MobileNetV3-Small on FPGA

This example contains 88 float32 Triton specializations and a complete
pretrained `mobilenetv3_small_100.lamb_in1k` runner with 121 operator calls.

The default images execute scalar float32 code on NH. They do not release E6
or use DMA, AME or RVV. Both kernels and their generated ABI adapters are built
without vector instructions. The optional `BOARD_PROFILE=nh-e6` builds legacy
dual-core standalone images; it does not convert scalar kernels into AME
kernels. The complete model uses the NH-only runtime.

Source the repository build environment and select the pinned Buddy tools:

```sh
export BUDDY_ROOT=/path/to/buddy-mlir
export LLVM_BIN="$BUDDY_ROOT/llvm/build/bin"
export BUDDY_MLIR_BINARY_DIR="$BUDDY_ROOT/build/bin"
# Run from this directory with the Python environment used for Triton export.
make -j8 asm PYTHON=python3
make -j8 bin PYTHON=python3
```

Each case produces `build/kernels/<case>/<case>.bin`, together with ELF,
assembly, frontend contracts, audit and build logs. Each image uses a separate
C numerical oracle and compares every output element. Output starts as NaN so
that missing stores cannot pass merely because an expected value is zero.

The standalone launcher prints a named `BEGIN/END` pair around the actual typed
adapter. Its cycle interval includes descriptor checks, the complete grid and
final synchronization; it excludes UART printing, input initialization and the
C oracle. The final report includes returned/validated flags, checked elements,
errors and cycles. A trap reports the in-flight case and machine context.
The NH runtime owns UART initialization and emits UART v1 START/RESULT records;
this directory does not provide a scheduler client or submission script.

Run local numerical checks after `asm`:

```sh
python3 tools/check_host.py --clang "$LLVM_BIN/clang"
python3 tools/export_model.py --output build/model
python3 tools/check_model_host.py --clang "$LLVM_BIN/clang"
make -f model/model.mk -j8 bin PYTHON=python3
```

The model exporter requires `torch`, `torchvision`, `timm`, `numpy`, `Pillow`
and the timm pretrained-download dependencies. A local pretrained state dict
can be selected with `--checkpoint`; `--image` selects a real RGB image using
timm's validation preprocessing. Full-model builds recheck the selected kernel
sources and effective compiler flags before linking. `LINKER=/path/to/ld.lld`
can select an external linker when it is absent from `LLVM_BIN`.

Host checks compile the actual emitted LLVM IR and original ABI adapters. With
a RISC-V Linux compiler and emulator, the same original `kernel.o` objects can
also be checked through QEMU:

```sh
python3 tools/check_riscv.py \
  --cc /path/to/riscv64-unknown-linux-gnu-gcc \
  --qemu /path/to/qemu-riscv64 --model build/model
```

Host/QEMU validation and ELF audits do not establish physical FPGA behavior or
cycle counts. The offline regressions cover launch reporting, missing stores,
ABI checks, export provenance and build invalidation:

```sh
python3 -m unittest discover -s tools -p 'test_*.py' -v
```

Generated model blobs, checkpoints, binaries and reports belong under `build/`
and are excluded from version control.

# EfficientNet-B3 FPGA example

This directory has the 99 standalone operators for
`timm/efficientnet_b3.ra2_in1k`, plus tools that compile, run and check the
complete model. The input is `1x3x288x288`. The complete graph has 308
logical nodes and 503 physical calls.

37 INT8 pointwise convolutions keep the existing packed GEMM. 4 ordinary
convolutions, 14 depthwise convolutions and the classifier keep the FP32
public ABI and use quantized **i8xi8 to i32 MMA** internally. Convolution
uses im2col, then dequantizes and adds bias. GAP uses the scalar Torch sum
order. SiLU uses RVV and writes the final output directly. The default build
avoids the vector max and add reductions that have numerical problems. Do
not disable the reduction checks.

Each operator directory's `kernel.py`, `launch.c` and `makefile` define the
fixed configuration, the bare-metal entry and the build. The shared compile
chain is `common.mk`, `board.mk` and `tools/`. Complete-model tools are in
`diagnostics/model_integration/`. Generated IR, objects, images and link
metadata stay in the gitignored `build/`.

## Environment

Run the commands below from the repository root. Use the environment that
has this repository's Triton backend installed, and a local Buddy/LLVM
toolchain:

```sh
source scripts/triton-riscv-env.sh
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$BUDDY_DIR"

EFFNET=examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k
MODEL="$EFFNET/build/model-integration"
TOOLS="$EFFNET/diagnostics/model_integration"
```

The complete-model CPU reference needs NumPy, Torch 2.10, torchvision,
safetensors and huggingface_hub. The GAP order matches Torch commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`. timm is the separately installed
`1.0.22` below. `BUDDY_OPT`, `LLVM_BIN`, `LLVM_OPT`, `LLC`, `CLANG` and
`LINKER` override tool locations.

## Compile and check locally

```sh
# All 99 operators: compile, link and audit the final ELF.
make -C "$EFFNET" -j4 bin

# One operator:
make -C "$EFFNET/conv2d_288x288_3x40_k3_s2" bin

# A subset:
make -C "$EFFNET" -j4 bin CASES='add_1x232x9x9 silu_1x6x1x1'

# Descriptor, grid and illegal-ABI checks:
make -C "$EFFNET" launchcheck

# Compare generated BOSCAME IR with an independent reference.
# A full-grid simulation of a large shape is slow:
make -C "$EFFNET" simcheck CASES='silu_1x6x1x1 gap_9x9x816 quantize_1x1536'
```

`asm` stops at `kernel.o`. `ttir`, `boscame` and `llvm` stop at earlier
stages. Each operator BIN is `build/kernels/<case>/<case>.bin`. IR and
assembly are in the same directory. 18 convolutions insert a per-patch DDR
read/write phase by default and run LLVM O2 with loop and SLP vectorization
off. The default GAP and SiLU objects come from the C implementation bound
by the case.

`launchcheck` checks the ABI with a mock raw entry. `simcheck` checks IR
arithmetic. Neither replaces a board numerical check. Some bare-metal
launchers are only a smoke test: exit code 0 means the call returned. The
complete model is accepted only after every logit is read back, as below.

## fpga112 connection and one operator

Use the `fpga-scheduler` client and confirm the host tunnel points at
fpga112. Reuse a correct HTTP forward. If there is none, open one:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 fpga112 true
ssh -fNT -o BatchMode=yes -o ConnectTimeout=8 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18762:127.0.0.1:18765 fpga112

export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"
FPGA_CLIENT=/home/chh/gitprojects/fpga-scheduler/.venv/bin/fpga-scheduler

"$FPGA_CLIENT" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" health
"$FPGA_CLIENT" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" devices
```

After a READY device is present:

```sh
"$FPGA_CLIENT" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" \
  --lease-seconds 300 --heartbeat-seconds 2 --request-timeout 180 \
  run "$EFFNET/build/kernels/conv2d_288x288_3x40_k3_s2/conv2d_288x288_3x40_k3_s2.bin" \
  --timeout 600 --queue-timeout 3600
```

The client must stay in the foreground until the task ends. The execution
limit, the client lease and the HTTP deadline are different settings. A
single operator over 60 seconds is marked HANG by convention. A longer
execution limit is only to see whether it eventually returns.
`launch_cycles` includes launcher work and is not pure kernel cycles. After
a disconnect, restore the tunnel and query the original task id. Do not
submit it again immediately.

## Prepare the complete model

Finish the operator build above first. Install the pinned timm and download
the weights and config the script requires:

```sh
python3 -m pip install --no-deps --target "$MODEL/python-deps" timm==1.0.22
python3 - <<'ASSETS'
from huggingface_hub import hf_hub_download
for filename in ("model.safetensors", "config.json"):
    hf_hub_download(
        repo_id="timm/efficientnet_b3.ra2_in1k",
        revision="0366a75518620e0f2077789202073759f2d61393",
        filename=filename,
        local_dir="examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/build/model-integration/assets",
    )
ASSETS

python3 "$TOOLS/model_graph.py"
python3 "$TOOLS/reference.py"
python3 "$TOOLS/check_connectors.py" --buddy-opt "$BUDDY_DIR/build/bin/buddy-opt"
python3 "$TOOLS/physical_plan.py"
python3 "$TOOLS/audit_model_plan.py"
python3 "$TOOLS/build_connectors.py" --jobs 2 --output "$MODEL/connectors-ddr-policy"
python3 "$TOOLS/generate_runner.py" --plan-dir "$MODEL/physical-plan" --output "$MODEL/runner"
```

`reference.py` checks the SHA of the pinned weights and config and writes
the original FP32 reference and the W8A8 CPU reference. The default input is
the deterministic tensor `((i*17)%257-128)/64`. It is not an image and does
not represent ImageNet accuracy. `reference.py --input-f32 PATH` selects a
preprocessed little-endian FP32 `[1,3,288,288]` input. Regenerate the
physical plan, runner and image after that.

Build the extra SiLU shapes and the RVV dequantization connectors:

```sh
python3 "$TOOLS/prepare_prefix.py"
python3 "$TOOLS/check_silu_direct_output.py" \
  --buddy-opt "$BUDDY_DIR/build/bin/buddy-opt" --stem "$MODEL/prefix8/stem.f32"
python3 "$TOOLS/build_silu_direct_output.py" \
  --output "$MODEL/silu-rvv-direct-output-overlay"
python3 "$TOOLS/check_dequant_candidate.py"
python3 "$TOOLS/build_dequant_overlay.py" \
  --plan "$MODEL/physical-plan/physical-plan.json" \
  --source-connectors "$MODEL/connectors-ddr-policy" --output "$MODEL/dequant-rvv-overlay"
```

GAP uses the default Torch-order implementation that the standard Make
target writes under `build/kernels/`. The checks above regenerate the host
validation and object manifests they need. They do not require an older
board record.

## Build and run the complete model

```sh
IMAGE="$MODEL/model-image"
python3 "$TOOLS/build_model_image.py" --jobs 4 \
  --host-numerical-validation --strict-probe \
  --plan-dir "$MODEL/physical-plan" --runner "$MODEL/runner" \
  --kernels "$EFFNET/build/kernels" --conv-overlay "$EFFNET/build/kernels" \
  --silu-overlay "$MODEL/silu-rvv-direct-output-overlay" \
  --silu-source "$TOOLS/silu_rvv_direct_output_candidate.c" \
  --dequant-overlay "$MODEL/dequant-rvv-overlay" \
  --connectors "$MODEL/connectors-ddr-policy" --output "$IMAGE"
python3 "$TOOLS/audit_image.py" --image-dir "$IMAGE" --plan-dir "$MODEL/physical-plan"
```

After the standalone checks, submit the complete model through the
scheduler:

```sh
python3 "$TOOLS/run_collect.py" --phase-gates-complete \
  --image-dir "$IMAGE" --reference-dir "$MODEL/physical-plan" \
  --output "$MODEL/board" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" \
  --timeout 7200 --queue-timeout 3600 \
  --lease-seconds 300 --heartbeat-seconds 2 --request-timeout 180
```

The command stays in the foreground and, when the task finishes, collects
UART, all 1,000 `logits_f32` values and 503 `call_cycles_u64` records.
Replace `TASK_ID` with the id this run printed. The following checks the
complete output, the image and reference hashes, and whether the final UART
and DDR cycles agree:

```sh
TASK_ID=replace-with-this-run-id
RUN="$MODEL/board/$TASK_ID"
python3 "$TOOLS/audit_board_run.py" --run-dir "$RUN" \
  --reference-dir "$MODEL/physical-plan" --primary-reference-dir "$MODEL/physical-plan"
python3 "$TOOLS/summarize_model_runtime.py" --run-dir "$RUN" --plan-dir "$MODEL/physical-plan"
```

`--strict-probe` uses `atol=rtol=2e-6`. `--host-numerical-validation` allows
the result to be exported after a complete run; the host numerical audit
decides whether it passes. A board `app0` exit, or a passing compile/ISA
check, is not a numerical pass. After a rebuild, check the new image. Do
not inherit a conclusion from an older object.

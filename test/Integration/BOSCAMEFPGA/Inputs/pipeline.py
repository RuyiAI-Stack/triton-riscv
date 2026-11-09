"""Shared opt pipeline driver for BOSCAME FPGA integration tests (pipeline.py).

Inputs: TTIR snippets and triton-shared-opt / buddy-opt pass lists.

Outputs: lowered IR for FileCheck or numeric Python assertions.

When used: Integration/BOSCAMEFPGA tests that reuse the same AOT pass recipe.
"""

import os
import subprocess


def buddy_opt_path():
    """The explicitly configured frontend/Contract/backend test bridge."""
    return os.environ["BOSCAME_TEST_PIPELINE_OPT"]


def run_pass(buddy_opt, fixture, output, dma_abi=True, extra=None):
    """Run the two-stage test bridge, returning (rc, stderr)."""
    command = [buddy_opt, fixture, *experimental_pipeline()]
    if not dma_abi:
        command[2] = "--triton-to-linalg-experimental=contract=true"
    if extra:
        command.extend(extra)
    command.extend(["-o", output])
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stderr


def experimental_pipeline(launch_grid=None):
    """Neutral frontend followed by explicit plan verification and emission."""
    frontend = "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias"
    if launch_grid:
        frontend += f" launch-grid={launch_grid}"
    return [
        frontend,
        "--kernel-contract-verify",
        "--bosc-ame-plan",
        "--bosc-ame-plan-verify",
        "--bosc-ame-emit",
    ]

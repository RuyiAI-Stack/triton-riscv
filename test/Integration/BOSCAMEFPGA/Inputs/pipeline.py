"""Two-process TTIR/Contract bridge helpers for optional integration tests."""

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

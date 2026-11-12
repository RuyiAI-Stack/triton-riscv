"""run_bin.py for BOSCAME FPGA model 'run_bin.py'.

Inputs: local paths, KERNEL_CONFIG, or board artifacts for this model.

Outputs: generated sources, checks, or reports used by the FPGA flow.

When used: from examples/BOSCAMEFPGA/run_bin.py tooling and Make targets.
"""

import argparse
import os
from fpga_scheduler import Client


def execution_timeout(value):
    seconds = int(value)
    if not 1 <= seconds <= 600:
        raise argparse.ArgumentTypeError("timeout must be an integer between 1 and 600")
    return seconds


parser = argparse.ArgumentParser(description="Run a binary through the FPGA scheduler.")
parser.add_argument("image")
parser.add_argument(
    "--timeout",
    type=execution_timeout,
    default=120,
    help="scheduler execution deadline in seconds (1..600; queue wait is separate)",
)
parser.add_argument(
    "--lease-seconds",
    type=execution_timeout,
    default=120,
    help="client liveness lease during upload/queue/run; separate from execution timeout",
)
args = parser.parse_args()

with Client(
    "http://127.0.0.1:18765",
    token_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "client.token"),
    request_timeout=360,
    lease_seconds=args.lease_seconds,
) as client:
    with client.run_test(args.image, timeout=args.timeout) as stream:
        for chunk in stream:
            print(chunk, end="", flush=True)
        result = stream.result
print(result.execution_status, result.app_exit_code, result.ok)
raise SystemExit(0 if result.ok else 1)

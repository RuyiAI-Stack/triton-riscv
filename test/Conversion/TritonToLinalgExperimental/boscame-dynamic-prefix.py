# RUN: %PYTHON %s %t triton-shared-opt
"""A dynamically aligned prefix must not become a falsely static DMA plan."""

import subprocess
import sys
from pathlib import Path


def main():
    prefix, opt = sys.argv[1:]
    fixture = Path(__file__).with_name("dynamic-prefix-dma.mlir")
    source = fixture.read_text()
    # Alignment alone does not prove the extent. Check both aligned and raw
    # bounds so the v1 rejection cannot accidentally depend on the bit mask.
    for name, text in (
        ("aligned", source),
        (
            "raw",
            source.replace(
                "%aligned_bound = arith.andi %limit, %packet_mask : i32",
                "%aligned_bound = arith.addi %limit, %packet_mask : i32",
            ),
        ),
    ):
        path = Path(prefix + "." + name + ".mlir")
        path.write_text(text)
        result = subprocess.run(
            [
                opt,
                str(path),
                "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias",
            ],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode > 0 and "kernel contract V4:" in result.stderr
        ), result.stderr
        assert (
            "LLVM ERROR" not in result.stderr
            and "Stack dump" not in result.stderr
        )


if __name__ == "__main__":
    main()

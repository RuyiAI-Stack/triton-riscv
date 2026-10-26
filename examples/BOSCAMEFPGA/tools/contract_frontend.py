#!/usr/bin/env python3
"""External triton-riscv AOT handoff shared by the real Triton examples."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def export_kernel(
    fn, signature, constexprs, grid, symbol, output, math_accuracy=None
):
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.contract import export_contract
    from triton.compiler import ASTSource

    artifact = export_contract(
        ASTSource(fn=fn, signature=signature, constexprs=constexprs),
        target=GPUTarget("boscame", "generic", 1),
        launch_grid=grid,
        entry_name=symbol,
        caller_contract="args_aligned64_nonalias",
        math_accuracy=math_accuracy,
    )
    artifact.write(output)
    return artifact


def frontend_identity():
    from triton._C import libtriton
    from triton.backends.triton_shared import contract
    from triton.backends.triton_shared.paths import _get_triton_shared_opt_path

    def sha(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    tool = _get_triton_shared_opt_path()
    return {
        "tool_sha256": sha(tool),
        "exporter_sha256": sha(contract.__file__),
        "triton_binary_sha256": sha(libtriton.__file__),
        "integration_sha256": sha(__file__),
        "build": json.loads(
            subprocess.check_output([tool, "--contract-build-info"], text=True)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    data = json.dumps(frontend_identity(), indent=2, sort_keys=True) + "\n"
    if not args.identity.exists() or args.identity.read_text() != data:
        args.identity.parent.mkdir(parents=True, exist_ok=True)
        args.identity.write_text(data)


if __name__ == "__main__":
    main()

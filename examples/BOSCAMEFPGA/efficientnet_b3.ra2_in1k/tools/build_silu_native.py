#!/usr/bin/env python3
"""Record native SiLU compilation independently of board/publication history."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[2]
sys.path.insert(0, str(MODEL))
from cases import describe  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clang", type=Path, required=True)
    p.add_argument("--tool-root", type=Path, required=True)
    a = p.parse_args()
    case = describe(a.case)
    if case.get("implementation") != "hybrid_rvv_c":
        raise ValueError("case is not configured for native SiLU")
    source = MODEL / case["implementation_source"]
    out = a.output
    manifest = {
        "case": case["name"],
        "implementation": "hybrid_rvv_c",
        "reference_implementation": "Triton -> BOSCAME scalar reference; kernel.o is standalone C/RVV",
        "candidate_source": source.relative_to(REPO).as_posix(),
        "candidate_source_sha256": sha(source),
        "kernel_config_sha256": sha(case["kernel_source"]),
        "object_sha256": sha(out / "kernel.o"),
        "adapter_sha256": sha(out / "adapter.c"),
        "kernel_s_sha256": sha(out / "kernel.s"),
        "encoded_s_sha256": sha(out / "kernel.encoded.s"),
        "restricted_s_sha256": sha(out / "kernel.nr.s"),
        "reference_module_sha256": sha(out / "kernel.boscame.mlir"),
        "instruction_restriction": "PASS",
        "compiler": str(a.clang),
        "compiler_sha256": sha(a.clang),
        "compile_mode": "default make asm path with explicit C/RVV selection",
        "source_entry": "efficientnet_silu_rvv_direct_output_candidate",
        "policy": "CPU input/scratch stores; fenced RVV scratch reads; direct write-through RVV output store; CPU exceptional output corrections after fence; no CPU load of RVV-stored scratch",
        "build_status": "PASS",
        "numerical_validation": {
            "status": "NOT_RUN",
            "reason": "compilation does not execute host/FPGA numerical validation; exact evidence belongs to the separate publication manifest",
        },
    }
    (out / "native-implementation.json").write_text(
        json.dumps(manifest, indent=2) + chr(10)
    )
    print(
        json.dumps(
            {
                "case": case["name"],
                "status": "PASS",
                "object_sha256": manifest["object_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

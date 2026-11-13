#!/usr/bin/env python3
"""Verify current standalone ELF/BIN/audit provenance without executing a board."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402 - local modules require the example path


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument("--report", type=Path, default=ROOT / "build/bin-report.json")
    p.add_argument(
        "--objcopy",
        default=os.environ.get(
            "OBJCOPY", str(ROOT.parents[2] / "buddy-mlir/llvm/build/bin/llvm-objcopy")
        ),
    )
    args = p.parse_args()
    records = []
    for case in inventory():
        name = case["name"]
        out = args.kernels / name
        board = out / "board"
        record = {"case": name, "status": "FAIL"}
        try:
            elf = board / (name + ".elf")
            binary = board / (name + ".bin")
            audit = board / "elf-audit.json"
            checked = json.loads(audit.read_text())
            assert checked["status"] == "PASS" and not checked["errors"], (
                "ELF audit failed"
            )
            assert checked["elf_sha256"] == digest(elf), "ELF audit hash is stale"
            frontend = json.loads((out / "frontend.json").read_text())
            assert frontend["kernel_module_sha256"] == digest(
                Path(case["kernel_source"])
            ), "kernel source differs from compiled frontend"
            assert frontend["ttir_sha256"] == digest(out / "kernel.ttir"), (
                "frontend TTIR hash is stale"
            )
            assert frontend["adapter_sha256"] == digest(out / "adapter.c"), (
                "typed ABI adapter hash is stale"
            )
            build = json.loads((out / "build-result.json").read_text())
            assert build["status"] == "PASS" and build["stage"] == "asm", (
                "current ASM build did not pass"
            )
            for filename, sha in build["artifacts"].items():
                assert digest(out / filename) == sha, (
                    "current ASM artifact hash is stale: " + filename
                )
            # Re-extract bytes from the audited ELF: a stale/foreign BIN cannot pass by existence alone.
            with tempfile.TemporaryDirectory(prefix="xlnet-bincheck-") as temp:
                copy = Path(temp) / "extracted.bin"
                subprocess.run(
                    [args.objcopy, "-O", "binary", str(elf), str(copy)],
                    check=True,
                    capture_output=True,
                )
                assert copy.read_bytes() == binary.read_bytes(), (
                    "BIN differs from current audited ELF"
                )
            files = {
                "elf": elf,
                "bin": binary,
                "elf_audit": audit,
                "kernel_object": out / "kernel.o",
                "kernel_source": Path(case["kernel_source"]),
                "launch_source": Path(case["launch_source"]),
                "link_script": ROOT / "board.ld",
                "board_inputs": board / ".board-inputs.json",
            }
            record.update(
                status="PASS",
                files={
                    key: {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": digest(path),
                        "bytes": path.stat().st_size,
                    }
                    for key, path in files.items()
                },
            )
        except (
            OSError,
            AssertionError,
            ValueError,
            subprocess.SubprocessError,
        ) as error:
            record["reason"] = str(error)
        records.append(record)
        print(record["status"] + ": " + name, flush=True)
    report = {
        "status": "PASS" if all(x["status"] == "PASS" for x in records) else "FAIL",
        "scope": "current standalone offline ELF/BIN/NR-audit hashes; no board execution",
        "case_count": len(records),
        "cases": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

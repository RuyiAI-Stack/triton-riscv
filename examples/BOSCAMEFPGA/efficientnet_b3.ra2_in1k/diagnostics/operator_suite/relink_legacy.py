#!/usr/bin/env python3
"""Relink unchanged kernels with current scheduler protocol; never re-export."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
sys.path.insert(0, str(MODEL))
from cases import inventory  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(case, args):
    source = args.source / case["name"]
    target = args.output / case["name"]
    if target.exists():
        raise RuntimeError(
            "refuse to overwrite existing relink directory: " + str(target)
        )
    before = {
        name: sha256(source / name)
        for name in ("kernel.o", "adapter.c", "kernel.boscame.mlir")
    }
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix not in (".bin", ".elf", ".map"):
            shutil.copy2(path, target / path.name)
    command = [
        "make",
        "--no-print-directory",
        "-f",
        str(MODEL / "board.mk"),
        "bin",
        "KERNELS=" + str(args.output),
        "BUDDY_ROOT=" + str(REPO / "buddy-mlir"),
        "LLVM_BIN=" + str(REPO / "buddy-mlir/llvm/build/bin"),
        "PYTHON=" + sys.executable,
    ]
    with (
        (target / "relink.stdout").open("w") as stdout,
        (target / "relink.stderr").open("w") as stderr,
    ):
        process = subprocess.run(
            command, cwd=MODEL / case["name"], stdout=stdout, stderr=stderr
        )
    if process.returncode:
        raise RuntimeError(
            "board.mk relink failed: "
            + case["name"]
            + "; see "
            + str(target / "relink.stderr")
        )
    after = {name: sha256(target / name) for name in before}
    original_after = {name: sha256(source / name) for name in before}
    if before != after or before != original_after:
        raise RuntimeError("kernel/adapter/IR changed while linking: " + case["name"])
    image = target / (case["name"] + ".bin")
    data = image.read_bytes()
    markers = {
        marker: marker.encode() in data
        for marker in (
            "@@FPGA_START",
            "@@FPGA_RESULT",
            "launch_returned=",
            "launch_cycles=0x",
        )
    }
    if not all(markers.values()):
        raise RuntimeError("new image lacks protocol/timer marker: " + case["name"])
    audit = json.loads((target / "elf-audit.json").read_text())
    result = {
        "case": case["name"],
        "family": case["family"],
        "source_directory": str(source),
        "artifact_directory": str(target),
        "unchanged_payload_sha256": before,
        "original_payload_still_matches": before == original_after,
        "command": command,
        "board_link_only": True,
        "reexported": False,
        "old_image_sha256": sha256(source / image.name),
        "image_sha256": sha256(image),
        "image_bytes": image.stat().st_size,
        "protocol_markers_in_image": markers,
        "elf_audit": audit,
    }
    (target / "protocol-relink.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LINKED " + case["name"] + " sha256=" + result["image_sha256"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=MODEL / "build/kernels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if args.workers < 1 or args.source == args.output:
        parser.error("positive workers and a distinct output directory are required")
    excluded = {"conv2d_f32", "dwconv2d_f32", "linear_f32"}
    cases = [
        case
        for case in inventory()
        if case["family"] not in excluded
        and (not args.case or case["name"] in args.case)
    ]
    if args.case and set(args.case) != {case["name"] for case in cases}:
        parser.error("selected case is not one of the unchanged80")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(worker, case, args): case for case in cases}
        for future in as_completed(pending):
            case = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"case": case["name"], "error": str(error)})
                print("ERROR " + case["name"] + ": " + str(error), flush=True)
            (args.output / "relink-summary.json").write_text(
                json.dumps(
                    {
                        "selected_cases": len(cases),
                        "finished_cases": len(results) + len(errors),
                        "status": "PASS"
                        if len(results) == len(cases) and not errors
                        else "PARTIAL",
                        "results": sorted(results, key=lambda item: item["case"]),
                        "errors": errors,
                    },
                    indent=2,
                )
                + "\n"
            )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

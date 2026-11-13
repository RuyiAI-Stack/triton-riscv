#!/usr/bin/env python3
"""Rerun strict pointwise kernels and compare saved pre-rename output hashes.

Each case is interpreted in a separate Python process. Historical compilation,
simulation, and FPGA records are read only; all new records use --report-dir.
This command neither submits FPGA tasks nor executes a complete model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402

EXCLUDED = {
    "gemm_i8",
    "matmul_i8",
    "gelu",
    "softmax",
    "patch_embed",
    "attention_pv",
    "attention_qk",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels", type=Path, default=ROOT / "build/operator-naming-20261029/kernels"
    )
    parser.add_argument(
        "--historical-kernels",
        type=Path,
        default=ROOT / "build/quantized-attention-20261029/kernels",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "build/operator-naming-20261029/pointwise-regression",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.kernels = args.kernels.resolve()
    args.historical_kernels = args.historical_kernels.resolve()
    args.report_dir = args.report_dir.resolve()
    if args.kernels == args.historical_kernels:
        parser.error("new and historical kernels must be different directories")
    cases = [case for case in inventory() if case["family"] not in EXCLUDED]
    if len(cases) != 28:
        parser.error(f"expected 28 pointwise/layout cases, found {len(cases)}")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    def run(case):
        name = case["name"]
        old_path = args.historical_kernels / name / "simcheck-result.json"
        old_sha_before = sha(old_path)
        old = read(old_path)
        path = args.report_dir / (name + ".json")
        command = [
            sys.executable,
            str(ROOT / "tools/simcheck.py"),
            "--case",
            name,
            "--output",
            str(args.kernels),
            "--report",
            str(path),
        ]
        start = time.monotonic()
        with (args.report_dir / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        new = read(args.kernels / name / "simcheck-result.json")
        before = [
            {
                key: r.get(key)
                for key in ("name", "order", "programs", "outputs", "output_sha256")
            }
            for r in old.get("runs", [])
        ]
        after = [
            {
                key: r.get(key)
                for key in ("name", "order", "programs", "outputs", "output_sha256")
            }
            for r in new.get("runs", [])
        ]
        hashes_equal = bool(before) and before == after
        preserved = old_sha_before == sha(old_path)
        passed = (
            result.returncode == 0
            and new.get("status") == "PASS"
            and old.get("status") == "PASS"
            and hashes_equal
            and preserved
        )
        return {
            "name": name,
            "family": case["family"],
            "status": "PASS" if passed else "FAIL",
            "returncode": result.returncode,
            "seconds": round(time.monotonic() - start, 3),
            "command": command,
            "strict_status": new.get("status"),
            "programs": new.get("programs", 0),
            "outputs": new.get("outputs", 0),
            "rounds": len(after),
            "output_hashes_equal": hashes_equal,
            "historical_record_preserved": preserved,
            "historical_record": str(old_path),
            "historical_record_sha256": old_sha_before,
            "current_record": str(args.kernels / name / "simcheck-result.json"),
            "current_record_sha256": sha(args.kernels / name / "simcheck-result.json"),
            "historical_runs": before,
            "current_runs": after,
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, case): case["name"] for case in cases}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"{record['status']}: {record['name']} programs={record['programs']} "
                f"outputs={record['outputs']} old/new hashes={record['output_hashes_equal']}",
                flush=True,
            )
    records.sort(key=lambda r: r["name"])
    report = {
        "schema": 1,
        "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL",
        "scope": "28 pointwise/layout independent operator strict emitted-IR regression",
        "full_model_executed": False,
        "fpga_submitted": False,
        "workers": args.workers,
        "required": len(cases),
        "passed": sum(r["status"] == "PASS" for r in records),
        "total_programs": sum(r["programs"] for r in records),
        "total_outputs": sum(r["outputs"] for r in records),
        "cases": records,
    }
    (args.report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "cases"}),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

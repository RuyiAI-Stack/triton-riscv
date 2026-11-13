#!/usr/bin/env python3
"""Publish a validated backend-only pointwise overlay with complete backups."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    return {
        str(item.relative_to(path)): sha(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay", type=Path, default=MODEL / "build/memory-task-lowering/pointwise37"
    )
    parser.add_argument(
        "--base", type=Path, default=MODEL / "build/operator-suite/suite-linked80"
    )
    parser.add_argument("--destination", type=Path, default=MODEL / "build/kernels")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=MODEL / "build/memory-task-lowering/publication37",
    )
    args = parser.parse_args()
    for key in ("overlay", "base", "destination", "evidence"):
        setattr(args, key, getattr(args, key).resolve())
    if args.evidence.exists():
        raise RuntimeError(
            "refusing to overwrite publication evidence: " + str(args.evidence)
        )
    source_hashes = json.loads(
        (MODEL / "build/pointwise-source-sha256.json").read_text()
    )
    names = sorted(Path(name).parent.name for name in source_hashes)
    assert len(names) == 37
    for name, digest in source_hashes.items():
        assert sha(MODEL / name) == digest, "pointwise kernel source changed: " + name
    full = MODEL / "build/board-runs/pointwise-memory-full-candidate/fused/all"
    numerical = [
        json.loads(path.read_text()) for path in full.glob("*/numerical-result.json")
    ]
    bounded = json.loads(
        (MODEL / "build/board-runs/pointwise-memory-abab32/results.json").read_text()
    )["runs"]
    assert len(numerical) == 2 and all(
        row["status"] == "PASS" and row["bit_exact"] and all(row["guards"].values())
        for row in numerical
    )
    assert len(bounded) == 8 and all(
        row["profile_counter_check_passed"]
        and row["execution_status"] == "exited"
        and row["app_exit_code"] == 0
        for row in bounded
    )
    manifests = {}
    for name in names:
        overlay, base, destination = (
            root / name for root in (args.overlay, args.base, args.destination)
        )
        record = json.loads((overlay / "overlay-manifest.json").read_text())
        assert (
            record["case"] == name
            and record["all_inputs_preserved"]
            and record["mma_static_sites_preserved"]
        )
        assert record["elf_audit_after"]["status"] == "PASS"
        assert record["kernel_object_sha256"] == sha(overlay / "kernel.o")
        assert record["bin_sha256"] == sha(overlay / (name + ".bin"))
        for filename in (
            "adapter.c",
            "kernel.boscame.mlir",
            "frontend.json",
            "contract-manifest.json",
        ):
            assert (
                sha(overlay / filename)
                == sha(base / filename)
                == sha(destination / filename)
            ), (name, filename)
        for filename, expected in record["source_hashes"].items():
            if filename == "kernel.py":
                assert sha(MODEL / name / filename) == expected
            elif filename not in ("source-kernel.o", "source-bin"):
                assert sha(overlay / filename) == expected
        manifests[name] = record
    args.evidence.mkdir(parents=True)
    backup = args.evidence / "backup-kernels"
    backup.mkdir()
    records = []
    # Back up every full directory before any production directory is mutated.
    for name in names:
        original = args.destination / name
        shutil.copytree(original, backup / name)
        original_hashes = tree(original)
        assert tree(backup / name) == original_hashes
        records.append({"case": name, "before_files": original_hashes})
    write(args.evidence / "backup-manifest.json", records)
    for row in records:
        name = row["case"]
        destination = args.destination / name
        shutil.copytree(args.base / name, destination, dirs_exist_ok=True)
        shutil.copytree(args.overlay / name, destination, dirs_exist_ok=True)
        result = json.loads((destination / "build-result.json").read_text())
        for filename in result.get("artifacts", {}):
            result["artifacts"][filename] = sha(destination / filename)
        result["artifact_dir"] = str(destination)
        result["backend_overlay_provenance"] = str(
            args.overlay / name / "overlay-manifest.json"
        )
        result["backend_validation_scope"] = (
            "2 complete representative nonzero FPGA probes, 8 bounded ABAB probes; all37 formal board timing pending"
        )
        write(destination / "build-result.json", result)
        build_report = json.loads((destination / "build-report.json").read_text())
        build_report["cases"] = [
            result if case.get("name") == name else case
            for case in build_report["cases"]
        ]
        write(destination / "build-report.json", build_report)
        overlay_files = tree(args.overlay / name)
        assert all(
            sha(destination / filename) == digest
            for filename, digest in overlay_files.items()
        )
        assert (
            sha(destination / "kernel.boscame.mlir")
            == row["before_files"]["kernel.boscame.mlir"]
        )
        assert sha(destination / "adapter.c") == row["before_files"]["adapter.c"]
        row.update(
            {
                "backup": str(backup / name),
                "after_files": tree(destination),
                "overlay": str(args.overlay / name),
                "object_changed": manifests[name]["object_changed"],
                "mma_static_sites_preserved": manifests[name][
                    "mma_static_sites_preserved"
                ],
                "elf_audit": "PASS",
            }
        )
    assert all(sha(MODEL / name) == digest for name, digest in source_hashes.items())
    report = {
        "status": "PUBLISHED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": len(records),
        "changed_objects": sum(row["object_changed"] for row in records),
        "source_kernels_unchanged": True,
        "source_sha256": source_hashes,
        "backup_root": str(backup),
        "destination": str(args.destination),
        "numerical": numerical,
        "bounded_ABAB_tasks": [row["task_id"] for row in bounded],
        "formal_board_validation": "pending for37 exact published images",
        "records": records,
    }
    write(args.evidence / "publication-report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "cases",
                    "changed_objects",
                    "source_kernels_unchanged",
                    "backup_root",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

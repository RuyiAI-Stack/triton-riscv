#!/usr/bin/env python3
"""Execute strict emitted-IR checks in isolated per-suite artifact copies.

The production build tree is read-only. Existing validated records can first be
collected by tools.simcheck.collect_case; otherwise its actual check_case runs
the full grid. A requested timeout is reported as incomplete, never as PASS.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
spec = importlib.util.spec_from_file_location("suite_runner", HERE / "run_suite.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def snapshot_case(record, output, reuse_evidence=None):
    runner.check_record(record)
    source = Path(record["artifact_directory"])
    destination = output / "kernels" / record["case"]
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or path.suffix not in (".json", ".mlir", ".ttir"):
            continue
        if path.name.startswith("simcheck-"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied[str(relative)] = runner.helpers.sha256(target)
    # Existing output evidence can be collected only after all sources/module
    # hashes and complete grid checks pass in the authoritative collector.
    saved = source / "simcheck-result.json"
    if reuse_evidence is not None:
        isolated = reuse_evidence / record["case"] / "simcheck-result.json"
        if isolated.is_file():
            saved = isolated
    if saved.is_file():
        shutil.copy2(saved, destination / saved.name)
    runner.helpers.write_json(
        destination / "snapshot.json",
        {
            "case": record["case"],
            "files": copied,
            "production_image_sha256": record["image_sha256"],
            "copied_simcheck_evidence": str(saved) if saved.is_file() else None,
            "copied_simcheck_evidence_sha256": runner.helpers.sha256(saved)
            if saved.is_file()
            else None,
        },
    )
    return destination


def validation_sources():
    buddy = Path(os.environ["BUDDY_ROOT"]).resolve()
    paths = [
        *sorted((MODEL / "tools").glob("*.py")),
        *sorted((buddy / "examples/FPGA-BOSCAME/tools").glob("*.py")),
    ]
    return {str(path): runner.helpers.sha256(path) for path in paths}


def worker(record, args):
    directory = args.output / "kernels" / record["case"]
    directory.mkdir(parents=True, exist_ok=True)
    before = validation_sources()
    runner.helpers.write_json(directory / "helper-provenance-before.json", before)
    result = worker_impl(record, args)
    after = validation_sources()
    runner.helpers.write_json(directory / "helper-provenance-after.json", after)
    changed = sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
    result["suite_helper_sources_stable_during_worker"] = not changed
    result["suite_changed_helper_sources"] = changed
    result["suite_helper_provenance"] = str(directory / "helper-provenance-before.json")
    if changed:
        result["suite_observed_numerical_status"] = result["status"]
        result["status"] = "INCOMPLETE_SOURCE_CHANGED"
        result["reason"] = (
            "validation helper sources changed while worker ran; rerun with a frozen checkout"
        )
    return result


def worker_impl(record, args):
    directory = snapshot_case(record, args.output, args.reuse_evidence)
    report_path = directory / "numerical-report.json"
    command = [
        sys.executable,
        "-B",
        str(MODEL / "tools/simcheck.py"),
        "--case",
        record["case"],
        "--output",
        str(args.output / "kernels"),
        "--report",
        str(report_path),
    ]
    started = time.monotonic()
    with (
        (directory / "simcheck.stdout").open("w") as stdout,
        (directory / "simcheck.stderr").open("w") as stderr,
    ):
        if not args.force and (directory / "simcheck-result.json").is_file():
            collect = subprocess.run(
                [*command, "--collect"], stdout=stdout, stderr=stderr, check=False
            )
            if collect.returncode == 0:
                result = json.loads(report_path.read_text())["cases"][0]
                result["suite_reused_authoritative_collect"] = True
                return result
        if args.collect_only:
            return {
                "name": record["case"],
                "status": "BLOCKED",
                "reason": "no current complete numerical record; actual simulation was not requested",
            }
        runner.log("SIMULATE " + record["case"])
        try:
            process = subprocess.run(
                command, stdout=stdout, stderr=stderr, check=False, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            return {
                "name": record["case"],
                "status": "INCOMPLETE_TIMEOUT",
                "reason": f"actual IR simulation exceeded{args.timeout}s",
                "wall_seconds": time.monotonic() - started,
            }
    if not report_path.is_file():
        return {
            "name": record["case"],
            "status": "FAIL",
            "reason": "simcheck returned no structured result",
            "exit_code": process.returncode,
        }
    result = json.loads(report_path.read_text())["cases"][0]
    result["suite_reused_authoritative_collect"] = False
    result["suite_wall_seconds"] = time.monotonic() - started
    result["suite_image_sha256"] = record["image_sha256"]
    runner.log(
        f"SIMRESULT {record['case']} {result['status']} programs={result.get('programs')}/{result.get('total_programs')}"
    )
    runner.check_record(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="optional per-case wall time ceiling; timeout remains incomplete",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even if the authoritative current collector accepts saved evidence",
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--reuse-evidence",
        type=Path,
        help="preserved isolated kernels root; collector still verifies exact source/module/full grid",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.timeout is not None and args.timeout < 1:
        parser.error("workers/timeout must be positive")
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    if args.reuse_evidence is not None:
        args.reuse_evidence = args.reuse_evidence.resolve()
    manifest, records = runner.load_selection(args)
    args.output.mkdir(parents=True, exist_ok=True)
    buddy_value = os.environ.get("BUDDY_ROOT")
    if not buddy_value:
        parser.error("BUDDY_ROOT must name the current local Buddy checkout")
    buddy = Path(buddy_value).resolve()
    simulator = buddy / "examples/FPGA-BOSCAME/tools/boscame_buffer_sim.py"
    if not simulator.is_file():
        parser.error("BUDDY_ROOT does not provide the strict BOSCAME simulator")
    runner.helpers.write_json(
        args.output / "numerical-environment.json",
        {
            "python": sys.executable,
            "python_version": sys.version,
            "buddy_root": str(buddy),
            "simulator": str(simulator),
            "simulator_sha256": runner.helpers.sha256(simulator),
            "source_sha256": validation_sources(),
            "thread_limits": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
    )
    shutil.copy2(args.manifest, args.output / "suite-manifest.json")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, record, args): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "name": record["case"],
                    "status": "FAIL",
                    "reason": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            report = {
                "schema": 1,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "suite_manifest_sha256": runner.helpers.sha256(args.manifest),
                "selected_cases": len(records),
                "finished_cases": len(results),
                "status": "PASS"
                if len(results) == len(records)
                and all(x["status"] == "PASS" for x in results)
                else "PARTIAL",
                "cases": sorted(results, key=lambda value: value["name"]),
            }
            runner.helpers.write_json(args.output / "validation-summary.json", report)
    return int(any(result["status"] != "PASS" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())

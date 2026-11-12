#!/usr/bin/env python3
"""Execute every selected frozen production BIN and preserve scheduler evidence.

60 seconds is the acceptable performance threshold, not proof of permanent
hang. A timeout escalates to600 then1200 seconds to obtain a returned cycle
count where possible. Known historical runtimes may start at a longer deadline.
Board completion and launcher smoke remain separate from numerical validation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
PROFILE_RUNNER = HERE.parent / "conv_profile/run_profile.py"
spec = importlib.util.spec_from_file_location("profile_runner_helpers", PROFILE_RUNNER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
CLIENT = helpers.CLIENT
STOP = threading.Event()
PRINT_LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def check_record(record):
    directory = Path(record["artifact_directory"])
    for name, expected in record["artifact_sha256"].items():
        path = directory / name
        if not path.is_file() or helpers.sha256(path) != expected:
            raise RuntimeError("frozen build artifact changed: " + str(path))
    for name, expected in record["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or helpers.sha256(path) != expected:
            raise RuntimeError("source changed after suite snapshot: " + str(path))


def load_selection(args):
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("formal_run_allowed") or manifest.get("status") != "READY":
        raise ValueError(
            "suite is preparation only; regenerate prepare_suite.py --ready after optimized artifacts freeze"
        )
    records = [
        row for row in manifest["cases"] if not args.case or row["case"] in args.case
    ]
    if not args.case and len(records) != 99:
        raise ValueError(
            "a formal full suite must contain exactly99 actual case images"
        )
    if args.case and set(args.case) != {row["case"] for row in records}:
        raise ValueError("unknown selected case")
    for record in records:
        check_record(record)
    return manifest, records


def query(base, arguments, directory, label):
    return helpers.invoke(base, arguments, directory, label)


def ready_after_platform_failure(base, directory):
    for number in range(12):
        devices = query(base, ["devices"], directory, f"ready-{number}")
        if any(
            item.get("state") == "READY" and item.get("available")
            for item in devices.get("devices", [])
        ):
            return True
        time.sleep(5)
    return False


def run_attempt(record, args, base, directory, deadline, attempt):
    check_record(record)
    attempt_dir = directory / f"attempt-{attempt}"
    attempt_dir.mkdir(parents=True)
    image = attempt_dir / Path(record["image"]).name
    shutil.copy2(record["image"], image)
    if helpers.sha256(image) != record["image_sha256"]:
        raise RuntimeError("BIN changed during snapshot: " + record["case"])
    task_id = uuid.uuid4().hex
    summary = {
        "case": record["case"],
        "family": record["family"],
        "task_id": task_id,
        "server": "fpga112",
        "attempt": attempt,
        "timeout_seconds": deadline,
        "queue_timeout_seconds": args.queue_timeout,
        "image_sha256": record["image_sha256"],
        "image_bytes": image.stat().st_size,
        "image_snapshot": str(image),
        "submitted_at_utc": utc(),
        "evidence_directory": str(attempt_dir),
        "acceptable_seconds": args.acceptable_seconds,
    }
    helpers.write_json(attempt_dir / "submission.json", summary)
    command = [
        *base,
        "run",
        str(image),
        "--id",
        task_id,
        "--name",
        "effnet-suite-" + record["case"],
        "--timeout",
        str(deadline),
        "--queue-timeout",
        str(args.queue_timeout),
    ]
    log(
        f"SUBMIT {record['case']} task={task_id} deadline={deadline}s sha256={record['image_sha256']}"
    )
    with (
        (attempt_dir / "run.stdout").open("w") as stdout,
        (attempt_dir / "run.stderr").open("w") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        summary["client_pid"] = process.pid
        helpers.write_json(attempt_dir / "submission.json", summary)
        # The foreground CLI stays alive until terminal state to own its lease.
        summary["client_exit_code"] = process.wait()
    summary["client_finished_at_utc"] = utc()
    helpers.write_json(attempt_dir / "submission.json", summary)
    try:
        task = query(base, ["job", task_id], attempt_dir, "job")
    except Exception:
        STOP.set()
        log(
            f"ERROR fpga112 API/connection lost task={task_id}; final task state unconfirmed; stop new submissions"
        )
        raise
    helpers.write_json(attempt_dir / "task.json", task)
    result = task.get("result") or {}
    details = result.get("details") or {}
    uart = result.get("uart_output", result.get("output", "")) or ""
    (attempt_dir / "uart.log").write_text(uart)
    match = re.search(r"launch_cycles=(0x[0-9a-fA-F]+|[0-9]+)", uart)
    cycles = helpers.integer(match.group(1)) if match else None
    returned = bool(re.search(r"launch_returned=1(?:\s|$)", uart))
    image_matches = details.get("image_sha256") == record["image_sha256"]
    summary.update(
        {
            "fpga": task.get("fpga"),
            "state": task.get("state"),
            "execution_status": result.get("execution_status"),
            "app_exit_code": result.get("app_exit_code"),
            "exit_code": result.get("exit_code"),
            "scheduler_seconds": result.get("seconds"),
            "details": details,
            "application_started": details.get("application_started"),
            "launch_returned": returned,
            "launch_cycles": cycles,
            "launch_cycles_hex": match.group(1) if match else None,
            "launch_seconds_at_5529600hz": cycles / 5529600.0
            if cycles is not None
            else None,
            "image_hash_match": image_matches,
            "exceeds_acceptable_60s_by_cycles": cycles
            > args.acceptable_seconds * 5529600
            if cycles is not None
            else None,
            "scheduler_exceeds_acceptable_seconds": result.get("seconds", 0)
            > args.acceptable_seconds,
            "timeout_interpretation": (
                "did not return by this execution deadline; does not prove permanent hang"
                if result.get("execution_status") == "timeout"
                else None
            ),
            "cycle_scope": "entire C launch, including initialization, adapter work and launcher postprocessing",
            "board_numerical_validation": (
                "launcher C oracle"
                if record["family"] in ("add", "silu")
                else "completion smoke only; numerical acceptance requires current IR simcheck"
            ),
        }
    )
    if details.get("image_sha256") not in (None, record["image_sha256"]):
        raise RuntimeError(task_id + ": scheduler image hash mismatch")
    summary["protocol_failure"] = (
        result.get("execution_status") == "timeout"
        and "@@FPGA_START" not in uart
        and details.get("application_started") is False
    )
    if summary["protocol_failure"]:
        summary["timeout_interpretation"] = (
            "PROTOCOL_FAILURE: no scheduler start marker; no operator hang established"
        )
    artifacts = query(base, ["artifacts", task_id], attempt_dir, "artifacts")
    helpers.write_json(attempt_dir / "artifacts.json", artifacts)
    downloads = []
    for artifact in artifacts.get("artifacts", []):
        name = artifact["name"]
        destination = attempt_dir / "downloads" / Path(name).name
        destination.parent.mkdir(exist_ok=True)
        query(
            base,
            ["download", task_id, name, destination],
            destination.parent,
            destination.name + ".download",
        )
        downloads.append(
            {
                **artifact,
                "path": str(destination),
                "downloaded_sha256": helpers.sha256(destination),
                "downloaded_bytes": destination.stat().st_size,
            }
        )
    summary["downloads"] = downloads
    summary["timing_passed"] = (
        task.get("state") == "completed"
        and result.get("execution_status") == "exited"
        and result.get("app_exit_code") == 0
        and result.get("exit_code") == 0
        and returned
        and cycles is not None
        and image_matches
    )
    helpers.write_json(attempt_dir / "summary.json", summary)
    log(
        f"RESULT {record['case']} task={task_id} fpga={summary['fpga']} execution={summary['execution_status']} cycles={cycles} seconds={summary['scheduler_seconds']}"
    )
    return summary


def run_case(record, args, base):
    directory = args.output / record["case"]
    directory.mkdir(parents=True, exist_ok=True)
    prior = directory / "summary.json"
    if args.resume and prior.is_file():
        saved = json.loads(prior.read_text())
        if saved.get("image_sha256") == record["image_sha256"] and saved.get(
            "final", {}
        ).get("timing_passed"):
            log(
                f"RESUME {record['case']} existing actual run={saved['final']['task_id']}"
            )
            return saved
    if STOP.is_set():
        return {
            "case": record["case"],
            "error": "new submission stopped after another scheduler/API error",
        }
    deadline = args.timeout or record["planned_initial_timeout_seconds"]
    deadlines = [
        value
        for value in (60, 600, 1200)
        if value >= deadline and value <= args.max_timeout
    ]
    if deadline not in deadlines:
        deadlines.insert(0, min(deadline, args.max_timeout))
    attempts = []
    for deadline in deadlines:
        for platform_retry in range(2):
            index = 1
            while (directory / f"attempt-{index}").exists():
                index += 1
            attempt = run_attempt(record, args, base, directory, deadline, index)
            attempts.append(attempt)
            summary = {
                "case": record["case"],
                "family": record["family"],
                "image_sha256": record["image_sha256"],
                "attempts": attempts,
                "final": attempt,
                "numerical_current_full_pass": record["numerical_current_full_pass"],
                "numerical_evidence": record["numerical_evidence"],
            }
            helpers.write_json(prior, summary)
            if attempt.get("protocol_failure"):
                STOP.set()
                log(
                    "PROTOCOL_FAILURE "
                    + record["case"]
                    + "; stop new submissions and do not escalate as operator hang"
                )
                break
            if (
                attempt["execution_status"] == "platform_error"
                and attempt["application_started"] is False
                and platform_retry == 0
                and ready_after_platform_failure(base, directory)
            ):
                log(
                    "RETRY "
                    + record["case"]
                    + " pre-application platform error after READY"
                )
                continue
            break
        if attempt["execution_status"] != "timeout" or attempt.get("protocol_failure"):
            break
        log(
            "ESCALATE "
            + record["case"]
            + f" did not return by{deadline}s; preserve attempt and allow next deadline"
        )
    return summary


def save_report(args, manifest, results):
    records = sorted(results, key=lambda item: item["case"])
    report = {
        "schema": 1,
        "server": "fpga112",
        "updated_at_utc": utc(),
        "suite_manifest": str(args.manifest),
        "suite_manifest_sha256": helpers.sha256(args.manifest),
        "selected_cases": len(manifest["cases"]) if not args.case else len(args.case),
        "finished_cases": len(records),
        "results": records,
    }
    helpers.write_json(args.output / "result-summary.json", report)
    fields = [
        "case",
        "family",
        "task_id",
        "fpga",
        "image_sha256",
        "execution_status",
        "app_exit_code",
        "exit_code",
        "launch_cycles",
        "scheduler_seconds",
        "launch_seconds_at_5529600hz",
        "exceeds_acceptable_60s_by_cycles",
        "timing_passed",
        "numerical_current_full_pass",
    ]
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(
                record.get("final", {}),
                case=record["case"],
                family=record.get("family"),
                numerical_current_full_pass=record.get("numerical_current_full_pass"),
            )
            writer.writerow({name: row.get(name) for name in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout", type=int, choices=(60, 600, 1200))
    parser.add_argument(
        "--max-timeout", type=int, choices=(60, 600, 1200), default=1200
    )
    parser.add_argument("--acceptable-seconds", type=int, default=60)
    parser.add_argument("--queue-timeout", type=int, default=3600)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--ready-wait", type=int, default=600)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only actual completed same-BIN runs from this output",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    manifest, selected = load_selection(args)
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "case": row["case"],
                        "sha256": row["image_sha256"],
                        "deadline": args.timeout
                        or row["planned_initial_timeout_seconds"],
                        "numerical_current_full_pass": row[
                            "numerical_current_full_pass"
                        ],
                    }
                    for row in selected
                ],
                indent=2,
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    token = REPO / ".agent/secrets/fpga112.token"
    if not token.is_file() or not token.stat().st_size:
        raise RuntimeError("fpga112 token missing or empty: " + str(token))
    base = [
        str(CLIENT),
        "--endpoint",
        "http://127.0.0.1:18762",
        "--token-file",
        str(token),
        "--json",
    ]
    helpers.preflight(base, args.output, args.ready_wait)
    shutil.copy2(args.manifest, args.output / "suite-manifest.json")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, row, args, base): row for row in selected}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as error:
                STOP.set()
                result = {
                    "case": row["case"],
                    "family": row["family"],
                    "error": f"{type(error).__name__}: {error}",
                }
                log("ERROR " + result["case"] + ": " + result["error"])
            results.append(result)
            save_report(args, manifest, results)
    return int(
        any(
            record.get("error") or not record.get("final", {}).get("timing_passed")
            for record in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

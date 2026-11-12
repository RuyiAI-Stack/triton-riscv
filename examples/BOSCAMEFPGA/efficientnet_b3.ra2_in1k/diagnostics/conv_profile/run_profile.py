#!/usr/bin/env python3
"""Run compiled profile manifests through fpga112's foreground scheduler client.

The runner snapshots each checked BIN, keeps one foreground CLI per lease, and
preserves final tasks, UART and every declared DDR artifact. A run that exceeds
its deadline is reported as a deadline timeout, never as a proven permanent hang.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
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
CLIENT = Path("/home/chh/gitprojects/fpga-scheduler/.venv/bin/fpga-scheduler")
TERMINAL = {"completed", "failed", "cancelled"}
PRINT_LOCK = threading.Lock()
STOP_SUBMISSIONS = threading.Event()


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value):
    """Hex counters are hex even if their spelling contains digits only."""
    if isinstance(value, int):
        return value
    value = str(value)
    if value.lower().startswith("0x"):
        return int(value[2:], 16)
    return int(value, 10)


def parse_profile(uart):
    report = {"lines": [], "stages": {}, "warnings": []}
    numeric = {
        "calls",
        "cycles",
        "adapter_total_cycles",
        "timer_pairs",
        "total_empty_pair_cycles",
        "launch_cycles",
        "launch_returned",
    }
    for line in uart.splitlines():
        if "PROFILE " not in line:
            continue
        payload = line.split("PROFILE ", 1)[1].strip()
        fields = dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", payload))
        parsed = dict(fields)
        for name in numeric & fields.keys():
            try:
                parsed[name] = integer(fields[name])
            except ValueError:
                report["warnings"].append(f"invalid {name}: {fields[name]}")
        report["lines"].append({"text": payload, "fields": dict(parsed)})
        if "stage" in parsed:
            stage = parsed.pop("stage")
            if stage in report["stages"]:
                report["warnings"].append("duplicate PROFILE stage: " + stage)
            report["stages"][stage] = parsed
        else:
            report.update(parsed)
    report["total_calls"] = sum(
        stage.get("calls", 0) for stage in report["stages"].values()
    )
    report["summed_stage_cycles"] = sum(
        stage.get("cycles", 0) for stage in report["stages"].values()
    )
    pairs = report.get("timer_pairs", 0)
    empty = report.get("total_empty_pair_cycles")
    overhead = empty / pairs if pairs and empty is not None else None
    report["timer_pair_mean_cycles"] = overhead
    report["estimated_total_timer_pair_cycles"] = (
        overhead * report["total_calls"] if overhead is not None else None
    )
    for stage in report["stages"].values():
        calls, cycles = stage.get("calls", 0), stage.get("cycles")
        stage["mean_cycles_per_call"] = (
            cycles / calls if calls and cycles is not None else None
        )
        stage["estimated_timer_pair_cycles"] = (
            overhead * calls if overhead is not None else None
        )
        # Preserve the primary raw measurement. This estimate removes only the
        # measured empty rdcycle pair, not wrapper/counter/cache overhead.
        stage["estimated_cycles_without_timer_pairs"] = (
            (cycles - overhead * calls)
            if cycles is not None and overhead is not None
            else None
        )
    if "adapter_total_cycles" in report:
        report["adapter_minus_stage_cycles"] = (
            report["adapter_total_cycles"] - report["summed_stage_cycles"]
        )
    launch = re.search(r"launch_cycles=(0x[0-9a-fA-F]+|[0-9]+)", uart)
    report["launch_cycles"] = integer(launch.group(1)) if launch else None
    return report


def expected_calls(manifest):
    """Count adapter raw calls; Triton tile lanes do not multiply C calls."""
    stages = {item["stage"]: item for item in manifest["stage_metadata"]}
    raw = stages.get("im2col_quantize", stages.get("pure_im2col"))
    constants = raw["constexprs"]
    depthwise = constants["DEPTHWISE"]
    rows = stages["weight_quantize"]["constexprs"]["ROWS"]
    positions = constants["OH"] * constants["OW"]
    tile = constants["TILE"]
    blocks = (positions + tile - 1) // tile
    groups = constants["C"] if depthwise else 1
    selected_tiles = integer(manifest.get("selected_tiles", groups * blocks))
    m_tiles = stages["gemm_i8"]["grid"][0]
    matrix_rows = stages["gemm_i8"]["constexprs"]["M"]
    values = {
        "weight_quantize": rows,
        "im2col_quantize": selected_tiles,
        "pure_im2col": selected_tiles,
        "activation_quantize": selected_tiles,
        "gemm_i8": selected_tiles * m_tiles,
        "dequantize_bias": selected_tiles * matrix_rows,
    }
    selected = manifest.get("selected_stage", "all")
    return {name: values[name] if selected in ("all", name) else 0 for name in stages}


def collect_manifests(arguments):
    paths = set()
    for path in arguments.manifest:
        path = path.resolve()
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "stage_metadata" in data:
            paths.add(path)
        elif isinstance(data, list):
            for item in data:
                if item.get("status") == "PASS" and item.get("bin_sha256"):
                    paths.add(Path(item["output"]).resolve() / "profile-manifest.json")
        else:
            raise ValueError(f"unsupported profile manifest/build report: {path}")
    for directory in arguments.manifest_root:
        paths.update(directory.resolve().rglob("profile-manifest.json"))
    result = []
    for path in sorted(paths):
        manifest = json.loads(path.read_text())
        if arguments.case and manifest["case"] not in arguments.case:
            continue
        if arguments.stage and manifest["selected_stage"] not in arguments.stage:
            continue
        if arguments.mode and manifest["mode"] not in arguments.mode:
            continue
        if manifest.get("split_variant") == "scalar_logical_width_v1":
            raise ValueError(
                f"invalid split coverage: scalar_logical_width_v1 processed one patch per tile call; rebuild corrected split profiles: {path}"
            )
        if (
            manifest.get("fixture", {}).get("kind") != "deterministic_nonzero_v1"
            or manifest.get("zero_initialized_launcher") is not False
        ):
            raise ValueError(
                f"profile lacks the required deterministic nonzero fixture: {path}"
            )
        image = path.parent / (manifest["case"] + ".bin")
        if not manifest.get("bin_sha256") or not image.is_file():
            raise ValueError(f"profile has not finished building: {path}")
        digest = sha256(image)
        if digest != manifest["bin_sha256"]:
            raise ValueError(f"profile BIN does not match completed manifest: {path}")
        result.append(
            {
                "path": path,
                "manifest": manifest,
                "image": image,
                "sha256": digest,
                "timeout": arguments.timeout or manifest.get("timeout_seconds", 600),
            }
        )
    if not result:
        raise ValueError("no compiled profile images match the requested selection")
    return result


def invoke(base, arguments, directory=None, name="query"):
    process = subprocess.run(
        [*base, *map(str, arguments)], capture_output=True, text=True, check=False
    )
    if directory is not None:
        (directory / (name + ".stdout")).write_text(process.stdout)
        (directory / (name + ".stderr")).write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(
            f"scheduler {arguments[0]} failed ({process.returncode}): {process.stderr[-1200:]}"
        )
    return json.loads(process.stdout)


def preflight(base, output, ready_wait):
    ssh = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ConnectionAttempts=1",
            "fpga112",
            "true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "ssh-preflight.stdout").write_text(ssh.stdout)
    (output / "ssh-preflight.stderr").write_text(ssh.stderr)
    if ssh.returncode:
        raise RuntimeError(
            "ascend 经 host 到 fpga112 (入口12222) 的 SSH 链路不可达；请检查并恢复 host 隧道。"
            + ssh.stderr[-500:]
        )
    write_json(output / "health.json", invoke(base, ["health"], output, "health"))
    deadline = time.monotonic() + ready_wait
    while True:
        devices = invoke(base, ["devices"], output, "devices")
        write_json(output / "devices.json", devices)
        if any(
            item.get("state") == "READY" and item.get("available")
            for item in devices.get("devices", [])
        ):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "fpga112 has no READY device; no new tasks submitted. See devices.json."
            )
        log("WAIT fpga112 has no READY device; existing tasks retain their leases")
        time.sleep(min(10, max(0.1, deadline - time.monotonic())))


def run_one(item, arguments, base):
    manifest = item["manifest"]
    key = Path(manifest["mode"]) / manifest["selected_stage"] / manifest["case"]
    directory = arguments.output / key
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        if (
            prior.get("image_sha256") == item["sha256"]
            and prior.get("state") in TERMINAL
            and not arguments.rerun
        ):
            log(f"SKIP {key} existing terminal task={prior.get('task_id')}")
            return prior
    if STOP_SUBMISSIONS.is_set():
        return {
            "key": str(key),
            "not_submitted": True,
            "error": "A scheduler call failed; stopped new submissions.",
        }
    attempt = 1
    while (directory / f"attempt-{attempt}").exists():
        attempt += 1
    attempt_dir = directory / f"attempt-{attempt}"
    attempt_dir.mkdir()
    image = attempt_dir / item["image"].name
    shutil.copy2(item["image"], image)
    shutil.copy2(item["path"], attempt_dir / "profile-manifest.json")
    if sha256(image) != item["sha256"]:
        raise RuntimeError("image changed while snapshotting: " + str(item["image"]))
    ident = uuid.uuid4().hex
    record = {
        "key": str(key),
        "case": manifest["case"],
        "mode": manifest["mode"],
        "selected_stage": manifest["selected_stage"],
        "task_id": ident,
        "server": "fpga112",
        "attempt": attempt,
        "submitted_at_utc": utc(),
        "timeout_seconds": item["timeout"],
        "queue_timeout_seconds": arguments.queue_timeout,
        "image_sha256": item["sha256"],
        "image_bytes": image.stat().st_size,
        "image_snapshot": str(image),
        "manifest_path": str(item["path"]),
        "manifest_sha256": sha256(item["path"]),
        "fixture": manifest.get("fixture"),
        "timing_semantics": manifest.get("timing"),
        "coverage": {
            name: manifest[name]
            for name in (
                "coverage",
                "selected_tile_pairs",
                "selected_tiles",
                "full_grid_tiles",
                "original_full_grid_stage_calls",
            )
            if name in manifest
        },
        "evidence_directory": str(attempt_dir),
    }
    command = [
        *base,
        "run",
        str(image),
        "--id",
        ident,
        "--name",
        "effnet-profile-" + str(key).replace("/", "-"),
        "--timeout",
        str(item["timeout"]),
        "--queue-timeout",
        str(arguments.queue_timeout),
    ]
    write_json(attempt_dir / "submission.json", record)
    log(f"SUBMIT {key} task={ident} timeout={item['timeout']} sha256={item['sha256']}")
    # Never detach this CLI: its stream owns and refreshes the task lease.
    with (
        (attempt_dir / "run.stdout").open("w") as stdout,
        (attempt_dir / "run.stderr").open("w") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        record["client_pid"] = process.pid
        write_json(attempt_dir / "submission.json", record)
        record["client_exit_code"] = process.wait()
    record["client_finished_at_utc"] = utc()
    write_json(attempt_dir / "submission.json", record)
    try:
        task = invoke(base, ["job", ident], attempt_dir, "job")
        write_json(attempt_dir / "task.json", task)
    except Exception:
        STOP_SUBMISSIONS.set()
        log(
            f"ERROR fpga112 API/连接异常 task={ident}；任务最终状态尚未确认，停止新提交；保留已收到日志。"
        )
        raise
    result = task.get("result") or {}
    details = result.get("details") or {}
    uart = result.get("uart_output", result.get("output", "")) or ""
    (attempt_dir / "uart.log").write_text(uart)
    record.update(
        {
            "fpga": task.get("fpga"),
            "state": task.get("state"),
            "execution_status": result.get("execution_status"),
            "app_exit_code": result.get("app_exit_code"),
            "exit_code": result.get("exit_code"),
            "scheduler_seconds": result.get("seconds"),
            "details": details,
            "application_started": details.get("application_started"),
            "timeout_interpretation": (
                "did not return before the configured execution deadline; does not prove a permanent hang"
                if result.get("execution_status") == "timeout"
                else None
            ),
        }
    )
    if details.get("image_sha256") not in (None, item["sha256"]):
        raise RuntimeError(f"{ident}: scheduler reported a different image hash")
    profile = parse_profile(uart)
    profile["expected_calls"] = expected_calls(manifest)
    for name, expected in (
        ("case", manifest["case"]),
        ("mode", manifest["mode"]),
        ("selected", manifest["selected_stage"]),
    ):
        if result.get("execution_status") == "exited" and profile.get(name) != expected:
            profile["warnings"].append(
                f"PROFILE {name}: expected {expected}, observed {profile.get(name)}"
            )
    if result.get("execution_status") == "exited":
        for name, expected in profile["expected_calls"].items():
            actual = profile["stages"].get(name, {}).get("calls")
            if actual != expected:
                profile["warnings"].append(
                    f"{name} call count: expected {expected}, observed {actual}"
                )
        if not profile["stages"]:
            profile["warnings"].append(
                "application returned without PROFILE stage counters"
            )
    write_json(attempt_dir / "profile.json", profile)
    record["profile"] = profile
    record["profile_counter_check_passed"] = (
        not profile["warnings"] if result.get("execution_status") == "exited" else None
    )
    artifacts = invoke(base, ["artifacts", ident], attempt_dir, "artifacts")
    write_json(attempt_dir / "artifacts.json", artifacts)
    record["downloads"] = []
    for artifact in artifacts.get("artifacts", []):
        name = artifact["name"]
        destination = attempt_dir / "downloads" / Path(name).name
        destination.parent.mkdir(exist_ok=True)
        invoke(
            base,
            ["download", ident, name, destination],
            destination.parent,
            destination.name + ".download",
        )
        record["downloads"].append(
            {
                **artifact,
                "path": str(destination),
                "downloaded_sha256": sha256(destination),
                "downloaded_bytes": destination.stat().st_size,
            }
        )
    write_json(attempt_dir / "summary.json", record)
    write_json(summary_path, record)
    log(
        f"RESULT {key} task={ident} fpga={record['fpga']} state={record['state']} execution={record['execution_status']} calls={profile['total_calls']} stage_cycles={profile['summed_stage_cycles']} counter_check={record['profile_counter_check_passed']}"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="completed profile-manifest.json or generator build-report JSON",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        action="append",
        default=[],
        help="recursively select completed profile manifests",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument(
        "--mode", action="append", choices=("fused", "split"), default=[]
    )
    parser.add_argument(
        "--output", type=Path, default=MODEL / "build/board-runs/w8a8-profile"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:18762")
    parser.add_argument(
        "--token-file", type=Path, default=REPO / ".agent/secrets/fpga112.token"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="execution deadline seconds (default manifest value or 600)",
    )
    parser.add_argument("--queue-timeout", type=int, default=3600)
    parser.add_argument("--ready-wait", type=int, default=600)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="make a new preserved attempt even if a terminal result already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate manifests/images and print selection without connecting or submitting",
    )
    args = parser.parse_args()
    if args.timeout is not None and not 1 <= args.timeout <= 86400:
        parser.error("--timeout must be between 1 and 86400")
    if not 1 <= args.queue_timeout <= 86400 or args.ready_wait < 0:
        parser.error("invalid queue timeout or ready wait")
    items = collect_manifests(args)
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "manifest": str(item["path"]),
                        "case": item["manifest"]["case"],
                        "mode": item["manifest"]["mode"],
                        "stage": item["manifest"]["selected_stage"],
                        "sha256": item["sha256"],
                        "timeout": item["timeout"],
                        "expected_calls": expected_calls(item["manifest"]),
                    }
                    for item in items
                ],
                indent=2,
            )
        )
        return 0
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.token_file.is_file() or not args.token_file.stat().st_size:
        raise RuntimeError(
            "fpga112 token file is missing or empty: " + str(args.token_file)
        )
    base = [
        str(CLIENT),
        "--endpoint",
        args.endpoint,
        "--token-file",
        str(args.token_file),
        "--json",
    ]
    preflight(base, args.output, args.ready_wait)
    results = []
    write_json(
        args.output / "selection.json",
        [
            {
                "manifest": str(item["path"]),
                "sha256": item["sha256"],
                "timeout_seconds": item["timeout"],
            }
            for item in items
        ],
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, item, args, base): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as error:
                STOP_SUBMISSIONS.set()
                record = {
                    "manifest_path": str(item["path"]),
                    "error": f"{type(error).__name__}: {error}",
                }
                log("ERROR " + record["error"])
            results.append(record)
            write_json(
                args.output / "result-summary.json",
                {
                    "server": "fpga112",
                    "updated_at_utc": utc(),
                    "results": sorted(
                        results,
                        key=lambda value: value.get(
                            "key", value.get("manifest_path", "")
                        ),
                    ),
                },
            )
    return int(
        any(
            record.get("error")
            or record.get("state") != "completed"
            or record.get("app_exit_code") != 0
            or not record.get("profile_counter_check_passed")
            for record in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

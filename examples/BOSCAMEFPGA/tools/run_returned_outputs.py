#!/usr/bin/env python3
"""Run FPGA operator images and validate immutable downloaded outputs locally."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[3]
MODELS = ("electra-base-discriminator", "xlnet-base-cased", "gte-small", "e5-small-v2")
SIBLING_CLIENT = REPO.parent / "fpga-scheduler/.venv/bin/fpga-scheduler"
CLIENT = os.environ.get(
    "FPGA_SCHEDULER_CLIENT",
    str(SIBLING_CLIENT) if SIBLING_CLIENT.is_file() else "fpga-scheduler",
)
ENDPOINT = os.environ.get("FPGA_SCHEDULER_URL", "http://127.0.0.1:18762")
TOKEN = Path(
    os.environ.get(
        "FPGA_SCHEDULER_TOKEN_FILE", str(REPO / ".agent/secrets/fpga112.token")
    )
)
BASE = [str(CLIENT), "--endpoint", ENDPOINT, "--token-file", str(TOKEN), "--json"]
TERMINAL = {"completed", "failed", "cancelled"}
DTYPES = {
    "|i1": 1,
    "|u1": 1,
    "<i2": 2,
    "<u2": 2,
    "<i4": 4,
    "<u4": 4,
    "<i8": 8,
    "<u8": 8,
    "<f2": 2,
    "<f4": 4,
    "<f8": 8,
}
FRONTEND_DTYPES = {
    "i8": "|i1",
    "u8": "|u1",
    "i16": "<i2",
    "i32": "<i4",
    "i64": "<i8",
    "f16": "<f2",
    "f32": "<f4",
    "f64": "<f8",
}


class StopSubmissions(RuntimeError):
    """A connection or unresolved existing job prevents subsequent submissions."""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def cli(arguments, directory: Path, label: str):
    process = subprocess.run(BASE + arguments, capture_output=True, text=True)
    (directory / (label + ".stdout")).write_text(process.stdout)
    (directory / (label + ".stderr")).write_text(process.stderr)
    if process.returncode:
        if label == "artifacts-list" or label.startswith("download-"):
            errors = (
                "Invalid artifact",
                "Artifact not found",
                "Artifact length or SHA256 mismatch",
                "Artifact response metadata mismatch",
                "Artifact response exceeds declared length",
                "HTTP 404",
            )
            if any(value in process.stderr for value in errors):
                raise ValueError(
                    "Scheduler artifact validation failed: " + process.stderr[-1200:]
                )
        raise StopSubmissions(
            f"Scheduler {label} failed; retain the existing task ID: "
            + process.stderr[-1200:]
        )
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise StopSubmissions(f"Scheduler {label} returned invalid JSON") from error


def stop_marker(directory):
    return next(
        (
            path / "STOP_SUBMISSIONS"
            for path in (directory, *directory.parents)
            if (path / "STOP_SUBMISSIONS").exists()
        ),
        None,
    )


def wait_ready(directory):
    """Wait before creating a task; keep every read-only preflight response."""
    counter, polls, recovering_polls = 0, 0, 0
    deadline = time.monotonic() + 3600
    while True:
        marker = stop_marker(directory)
        if marker is not None:
            raise StopSubmissions(
                "New submissions stopped by local marker: " + str(marker)
            )
        prefix = f"preflight-{counter:04d}"
        while (directory / (prefix + "-health.stdout")).exists():
            counter += 1
            prefix = f"preflight-{counter:04d}"
        health = cli(["health"], directory, prefix + "-health")
        devices = cli(["devices"], directory, prefix + "-devices")
        if (
            health.get("stopping")
            or health.get("scheduler_error")
            or health.get("dispatcher_alive") is not True
        ):
            raise StopSubmissions("Scheduler health is not ready for submission")
        if any(
            item.get("available") is True and item.get("state") == "READY"
            for item in devices.get("devices", [])
        ):
            return
        items = devices.get("devices", [])
        if not items or devices.get("errors"):
            raise StopSubmissions(
                "Device inventory is unavailable: "
                + json.dumps(devices, ensure_ascii=False)
            )
        recovering_polls = (
            0
            if any(item.get("state") == "BUSY" for item in items)
            else recovering_polls + 1
        )
        if recovering_polls >= 3:
            raise StopSubmissions(
                "No READY device; persistent recovery/unavailability: "
                + json.dumps(items, ensure_ascii=False)
            )
        if time.monotonic() >= deadline:
            raise StopSubmissions(
                "No READY device within the 3600-second preflight wait"
            )
        if polls % 6 == 0:
            print(
                "WAIT_READY",
                directory.name,
                {item.get("number"): item.get("state") for item in items},
                flush=True,
            )
        counter += 1
        polls += 1
        time.sleep(10)


def output_contract(config, frontend, allow_selected=False):
    outputs = config.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("board_config.json must contain a nonempty outputs list")
    expected, invocations = {}, config["expected_invocations"]
    for item in outputs:
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]{0,63}", name
        ):
            raise ValueError("Invalid output name")
        if name in expected:
            raise ValueError("Duplicate expected output " + name)
        shape, dtype = item["shape"], item["dtype"]
        if (
            dtype not in DTYPES
            or not isinstance(shape, list)
            or not shape
            or any(type(d) is not int or d < 1 for d in shape)
            or item["nbytes"] != math.prod(shape) * DTYPES[dtype]
        ):
            raise ValueError("Invalid dtype/shape/nbytes for " + name)
        argument, invocation = item["argument"], item["invocation"]
        if (
            type(argument) is not int
            or not 0 <= argument < len(frontend["arguments"])
            or type(invocation) is not int
            or not 0 <= invocation < invocations
        ):
            raise ValueError("Invalid argument/invocation for " + name)
        if shape != frontend["argument_shapes"][argument]:
            raise ValueError("Output shape differs from compiled frontend for " + name)
        if FRONTEND_DTYPES.get(frontend["arguments"][argument]["dtype"]) != dtype:
            raise ValueError("Output dtype differs from compiled frontend for " + name)
        if "outputs" in frontend and argument not in frontend["outputs"]:
            raise ValueError("Output argument differs from frontend output contract")
        if name != f"r{invocation}_{frontend['arguments'][argument]['name']}":
            raise ValueError("Output name differs from invocation/argument contract")
        expected[name] = item
    covered = {(item["invocation"], item["argument"]) for item in outputs}
    selected = config.get("export_invocation")
    if allow_selected and selected is None:
        selected = config.get("selected_export_invocation")
    required_invocations = (
        [selected] if allow_selected and selected is not None else range(invocations)
    )
    required = {
        (invocation, argument)
        for invocation in required_invocations
        for argument in frontend["outputs"]
    }
    if covered != required or len(covered) != len(outputs):
        raise ValueError(
            "Output snapshots do not cover every invocation/output argument exactly once"
        )
    return expected


def normalized_frontend(root, case, frontend):
    """Resolve physical shapes for older manifests without importing Triton."""
    value = dict(frontend)
    if "argument_shapes" not in value:
        path = root / "tools/export.py"
        spec = importlib.util.spec_from_file_location(
            "returned_outputs_export_" + root.name.replace("-", "_"), path
        )
        module = importlib.util.module_from_spec(spec)
        previous_cases = sys.modules.pop("cases", None)
        previous_path = sys.path[:]
        try:
            spec.loader.exec_module(module)
            value["argument_shapes"] = [
                list(shape) for shape in module.argument_shapes(module.describe(case))
            ]
        finally:
            sys.path[:] = previous_path
            sys.modules.pop("cases", None)
            if previous_cases is not None:
                sys.modules["cases"] = previous_cases
    path = root / "tools/output_spec.py"
    spec = importlib.util.spec_from_file_location(
        "returned_outputs_spec_" + root.name.replace("-", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    indices = list(module.output_indices(case))
    if "outputs" in value and value["outputs"] != indices:
        raise ValueError("Frontend outputs differ from reviewed output_spec.py")
    value["outputs"] = indices
    return value


def source_files(root: Path, source: Path):
    files = {
        root / source.name / "kernel.py",
        root / source.name / "launch.c",
        source / "adapter.c",
    }
    case_makefile = root / source.name / "makefile"
    if case_makefile.exists():
        files.add(case_makefile)
    files.update((source / "board").glob("*.c"))
    files.update((source / "board").glob("*.h"))
    files.update(root.glob("*.h"))
    for name in (
        "support.c",
        "cases.py",
        "board.mk",
        "board.ld",
        "common.mk",
        "makefile",
    ):
        path = root / name
        if path.exists():
            files.add(path)
    for name in (
        "board_config.py",
        "board_support.c",
        "board_support.h",
        "output_spec.py",
        "export.py",
        "attention_reference.h",
        "launch_checks.h",
    ):
        path = root / "tools" / name
        if path.exists():
            files.add(path)
    for name in (
        "scheduler_outputs.py",
        "scheduler_output_runtime.h",
        "scheduler_output_runtime.inc",
    ):
        shared = REPO / "examples/BOSCAMEFPGA/tools" / name
        if shared.exists():
            files.add(shared)
    for directory in ("nh_e6", "runtime", "uart", "nr"):
        runtime = REPO / "buddy-mlir/runtime/boscame" / directory
        for suffix in ("*.c", "*.h", "*.S"):
            files.update(runtime.glob(suffix))
    for path in (
        source / "board/.board-inputs.json",
        source / ".ordered-reduction-policy",
    ):
        if path.exists():
            files.add(path)
    return sorted(files)


def provenance(
    root: Path, case: str, kernels: Path | None = None, allow_selected=False
):
    source = (kernels or root / "build/kernels") / case
    board = source / "board"
    paths = {
        "image": board / (case + ".bin"),
        "elf": board / (case + ".elf"),
        "config": board / "board_config.json",
        "audit": board / "elf-audit.json",
        "frontend": source / "frontend.json",
    }
    values = {
        key: json.loads(paths[key].read_text())
        for key in ("config", "audit", "frontend")
    }
    hashes = {key + "_sha256": sha(path) for key, path in paths.items()}
    if (
        values["audit"].get("status") != "PASS"
        or values["audit"].get("elf_sha256") != hashes["elf_sha256"]
    ):
        raise ValueError(case + ": ELF audit is missing, failed or stale")
    frontend = values["frontend"]
    if frontend["kernel_module_sha256"] != sha(root / case / "kernel.py"):
        raise ValueError(case + ": compiled kernel differs from current kernel.py")
    if frontend.get("adapter_sha256") != sha(source / "adapter.c"):
        raise ValueError(case + ": frontend adapter hash is stale")
    config = values["config"]
    if config.get("launch_source_sha256") != sha(root / case / "launch.c"):
        raise ValueError(
            case + ": compiled output configuration has a stale launch source hash"
        )
    if config.get("host_support_sha256") != sha(root / "support.c"):
        raise ValueError(
            case + ": compiled output configuration has a stale host support hash"
        )
    values["frontend"] = normalized_frontend(root, case, frontend)
    hashes["source_sha256"] = {
        str(path.relative_to(REPO)): sha(path) for path in source_files(root, source)
    }
    return (
        paths,
        values,
        hashes,
        output_contract(config, values["frontend"], allow_selected),
    )


def ensure_same_provenance(record, hashes, directory):
    for key, value in hashes.items():
        if record.get(key) != value:
            raise ValueError(
                f"Existing result provenance changed ({key}); use a new --output directory"
            )
    frozen = {
        "image": directory / "image.bin",
        "elf": directory / "image.elf",
        "config": directory / "board_config.json",
        "audit": directory / "elf-audit.json",
        "frontend": directory / "frontend.json",
    }
    for key, path in frozen.items():
        if not path.exists() or sha(path) != hashes[key + "_sha256"]:
            raise ValueError("Frozen evidence is missing or changed: " + str(path))
    for path, digest in hashes["source_sha256"].items():
        frozen_source = directory / "sources" / path
        if not frozen_source.exists() or sha(frozen_source) != digest:
            raise ValueError("Frozen source is missing or changed: " + path)


def validate_artifact_metadata(items, expected):
    if not isinstance(items, list):
        raise ValueError("Invalid artifact list")
    actual = {}
    for item in items:
        name = item.get("name")
        if name in actual or name not in expected:
            raise ValueError("Duplicate or unexpected output " + str(name))
        for key in ("dtype", "shape", "nbytes"):
            if item.get(key) != expected[name][key]:
                raise ValueError(f"Output {name} has unexpected {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            raise ValueError("Invalid artifact SHA256 for " + name)
        actual[name] = item
    if set(actual) != set(expected):
        raise ValueError("Expected output declarations or artifacts are missing")
    return actual


def verify_uart_declarations(uart, expected, actual):
    declarations = []
    for value in re.findall(r"@@FPGA_OUTPUT\s+(\{[^\r\n]*\})", uart):
        declarations.append(json.loads(value))
    if len(declarations) != len(expected):
        raise ValueError("UART does not declare every expected output exactly once")
    seen = set()
    for item in declarations:
        name = item.get("name")
        if item.get("version") != 1 or name not in expected or name in seen:
            raise ValueError("Invalid or duplicate UART output declaration")
        seen.add(name)
        for key in ("dtype", "shape", "nbytes", "address"):
            if item.get(key) != actual[name].get(key):
                raise ValueError(f"UART and artifact disagree for {name}/{key}")


def files_match(record, directory):
    for item in record.get("artifacts", []):
        path = directory / "artifacts" / (item["name"] + ".bin")
        if (
            not path.exists()
            or path.stat().st_size != item["nbytes"]
            or sha(path) != item["sha256"]
        ):
            return False
    report = directory / "local-validation.json"
    oracle_report = directory / "local-validation/report.json"
    manifest = directory / "artifacts/manifest.json"
    verifier = Path(__file__).with_name("verify_returned_outputs.py")
    return (
        bool(record.get("artifacts"))
        and report.exists()
        and sha(report) == record.get("local_validation_sha256")
        and oracle_report.exists()
        and sha(oracle_report) == record.get("local_oracle_report_sha256")
        and manifest.exists()
        and sha(manifest) == record.get("artifact_manifest_sha256")
        and verifier.exists()
        and sha(verifier) == record.get("local_validation_verifier_sha256")
    )


def local_oracle_pass(report_path, task_id, manifest_sha=None):
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "LOCAL_ORACLE_PASS"
        or report.get("oracle_executed") is not True
        or report.get("returned_artifacts_compared") is not True
        or report.get("artifact_manifest_verified") is not True
        or report.get("scheduler_job_id") != task_id
        or (report.get("replay") or {}).get("exit_code") != 0
    ):
        return False
    if (
        manifest_sha is not None
        and report.get("artifact_manifest_sha256") != manifest_sha
    ):
        return False
    return True


def replay_locally(root, case, directory, record):
    local = directory / "local-validation"
    record_path = directory / "result.json"
    try:
        _, _, current_hashes, _ = provenance(root, case, Path(record["kernels"]))
        ensure_same_provenance(record, current_hashes, directory)
    except (ValueError, KeyError, OSError) as error:
        record.update(status="SOURCE_CHANGED", error=str(error))
        save(record_path, record)
        raise StopSubmissions(
            "Sources changed before native replay: " + record["task_id"]
        ) from error
    try:
        verifier = Path(__file__).with_name("verify_returned_outputs.py")
        record["local_validation_verifier_sha256"] = sha(verifier)
        command = [
            sys.executable,
            str(verifier),
            "--model",
            str(root),
            "--case",
            case,
            "--artifacts",
            str(directory / "artifacts"),
            "--config",
            str(directory / "board_config.json"),
            "--output",
            str(local),
        ]
        record["local_validation_command"] = command
        checked = subprocess.run(command, capture_output=True, text=True)
        (directory / "local-validation.stdout").write_text(checked.stdout)
        (directory / "local-validation.stderr").write_text(checked.stderr)
        local_report = {
            "command": command,
            "returncode": checked.returncode,
            "stdout": checked.stdout,
            "stderr": checked.stderr,
        }
        oracle_report = local / "report.json"
        if oracle_report.exists():
            local_report["oracle_report"] = json.loads(oracle_report.read_text())
            record["local_oracle_report_sha256"] = sha(oracle_report)
        save(directory / "local-validation.json", local_report)
        record["local_validation_sha256"] = sha(directory / "local-validation.json")
        passed = (
            checked.returncode == 0
            and oracle_report.exists()
            and local_oracle_pass(
                oracle_report, record["task_id"], record["artifact_manifest_sha256"]
            )
        )
        record["status"] = "PASS" if passed else "LOCAL_VALIDATION_FAIL"
    except (ValueError, KeyError, OSError) as error:
        record.update(status="LOCAL_VALIDATION_FAIL", error=str(error))
    try:
        _, _, current_hashes, _ = provenance(root, case, Path(record["kernels"]))
        ensure_same_provenance(record, current_hashes, directory)
    except (ValueError, KeyError, OSError) as error:
        record.update(status="SOURCE_CHANGED", error=str(error))
        save(record_path, record)
        raise StopSubmissions(
            "Sources changed during native replay: " + record["task_id"]
        ) from error


def run_case(
    root: Path,
    case: str,
    output: Path,
    timeout: int,
    kernels: Path | None = None,
    defer_local_validation=False,
):
    kernels = kernels or root / "build/kernels"
    paths, values, hashes, expected = provenance(
        root, case, kernels, defer_local_validation
    )
    directory = output / case
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "result.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        ensure_same_provenance(record, hashes, directory)
        if record.get("status") == "PASS" and files_match(record, directory):
            print("CACHED", root.name, case, "PASS", flush=True)
            return record
        if defer_local_validation and record.get("status") == "DOWNLOADED_FOR_MERGE":
            actual = validate_artifact_metadata(record.get("artifacts"), expected)
            if all(
                (directory / "artifacts" / (name + ".bin")).exists()
                and sha(directory / "artifacts" / (name + ".bin")) == item["sha256"]
                and (directory / "artifacts" / (name + ".bin")).stat().st_size
                == item["nbytes"]
                for name, item in actual.items()
            ):
                print("CACHED", root.name, case, "DOWNLOADED_FOR_MERGE", flush=True)
                return record
        if record.get("state") in TERMINAL and record.get("status") in {
            "TRAP",
            "APPLICATION_FAIL",
            "PLATFORM_ERROR",
            "CANCELLED",
        } | {"TIMEOUT_" + str(record.get("execution_timeout_seconds")) + "S"}:
            print("CACHED", root.name, case, record["status"], flush=True)
            return record
        task = cli(["job", record["task_id"]], directory, "recovery-query")
        if task.get("state") not in TERMINAL:
            raise StopSubmissions(
                "Existing nonterminal job requires recovery; do not resubmit: "
                + record["task_id"]
            )
    else:
        wait_ready(directory)
        # Input files may have changed while waiting for a free board. Freeze
        # and audit the exact set that is available at the actual submission.
        paths, values, hashes, expected = provenance(
            root, case, kernels, defer_local_validation
        )
        marker = stop_marker(directory)
        if marker is not None:
            raise StopSubmissions(
                "New submissions stopped by local marker: " + str(marker)
            )
        for key, name in (
            ("image", "image.bin"),
            ("elf", "image.elf"),
            ("config", "board_config.json"),
            ("audit", "elf-audit.json"),
            ("frontend", "frontend.json"),
        ):
            shutil.copyfile(paths[key], directory / name)
        for path in hashes["source_sha256"]:
            destination = directory / "sources" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / path, destination)
        ident = uuid.uuid4().hex
        record = {
            "model": root.name,
            "case": case,
            "server": "fpga112",
            "endpoint": ENDPOINT,
            "task_id": ident,
            "started_utc": now(),
            "status": "SUBMITTING",
            "kernels": str(kernels),
            "execution_timeout_seconds": timeout,
            "queue_timeout_seconds": 3600,
            "expected_invocations": values["config"]["expected_invocations"],
            "expected_outputs": list(expected.values()),
            **hashes,
        }
        command = BASE + [
            "run",
            str(directory / "image.bin"),
            "--id",
            ident,
            "--name",
            root.name + "/returned-outputs/" + case,
            "--timeout",
            str(timeout),
            "--queue-timeout",
            "3600",
        ]
        record["command"] = command
        save(record_path, record)
        print("RUN", root.name, case, ident, flush=True)
        begin = time.monotonic()
        with (
            (directory / "client.stdout").open("w") as stdout,
            (directory / "client.stderr").open("w") as stderr,
        ):
            # Scheduler owns the application deadline; this foreground client
            # owns the live lease. No outer timeout includes queue/load time.
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                time.sleep(2)
        record.update(
            client_returncode=process.returncode,
            wall_seconds_including_queue_load=time.monotonic() - begin,
        )
        try:
            task = json.loads((directory / "client.stdout").read_text())
            if task.get("id") != ident or "state" not in task:
                raise ValueError("Missing terminal task")
        except (json.JSONDecodeError, ValueError):
            # This lookup is recovery of the same ID, never a new submission.
            try:
                task = cli(["job", ident], directory, "connection-recovery")
            finally:
                record["status"] = "CONNECTION_ERROR"
                save(record_path, record)
            if task.get("state") not in TERMINAL:
                raise StopSubmissions(
                    "CLI disconnected; original job is nonterminal: " + ident
                )
            save(directory / "task.json", task)
            raise StopSubmissions(
                "CLI disconnected; terminal evidence recovered. Stop new submissions and resume the same output directory: "
                + ident
            )
    if task.get("id") != record["task_id"] or task.get("state") not in TERMINAL:
        raise StopSubmissions(
            "Unexpected task ID or nonterminal scheduler response: " + record["task_id"]
        )
    save(directory / "task.json", task)
    result = task.get("result") or {}
    details = result.get("details") or {}
    uart = result.get("uart_output") or result.get("output") or ""
    (directory / "uart.log").write_text(uart)
    record.update(
        state=task.get("state"),
        fpga=task.get("fpga"),
        node=task.get("node"),
        execution_status=result.get("execution_status"),
        exit_code=result.get("exit_code"),
        app_exit_code=result.get("app_exit_code"),
        details=details,
        scheduler_seconds=result.get("seconds"),
        finished_utc=now(),
    )
    record["kernel_cycles"] = [
        int(x, 16) for x in re.findall(r"kernel_cycles=0x([0-9a-fA-F]+)", uart)
    ]
    record["invocation_ordinals"] = [
        int(x) for x in re.findall(r"invocation=(\d+)\s+kernel_cycles=", uart)
    ]
    record["status"] = "APPLICATION_FINISHED"
    save(record_path, record)
    if details.get("image_sha256") != record["image_sha256"]:
        record["status"] = "PLATFORM_ERROR"
        record["error"] = "Server image SHA256 is missing or differs"
        save(record_path, record)
        raise StopSubmissions("Server image hash mismatch: " + record["task_id"])
    status = result.get("execution_status")
    if details.get("artifact_error"):
        record.update(status="ARTIFACT_FAIL", error=str(details["artifact_error"]))
    elif (
        details.get("application_started") is not True
        or details.get("reset_confirmed") is not True
    ):
        record["status"] = "PLATFORM_ERROR"
    elif status == "timeout":
        record["status"] = "TIMEOUT_" + str(record["execution_timeout_seconds"]) + "S"
    elif status == "trap":
        record["status"] = "TRAP"
    elif task.get("state") == "cancelled":
        record["status"] = "CANCELLED"
    elif status != "exited":
        record["status"] = "PLATFORM_ERROR"
    elif (
        task.get("state") != "completed"
        or result.get("exit_code") != 0
        or result.get("app_exit_code") != 0
        or details.get("ddr_readback_matches") is not True
        or len(record["kernel_cycles"]) != record["expected_invocations"]
        or record["invocation_ordinals"] != list(range(record["expected_invocations"]))
    ):
        record["status"] = "APPLICATION_FAIL"
    else:
        try:
            if details.get("artifacts_skipped"):
                raise ValueError("Scheduler output capture failed: " + str(details))
            listed = cli(["artifacts", record["task_id"]], directory, "artifacts-list")
            if listed.get("id") != record["task_id"]:
                raise ValueError("Artifact list task ID mismatch")
            actual = validate_artifact_metadata(listed.get("artifacts"), expected)
            if result.get("artifacts") != listed["artifacts"]:
                # Array ordering is immaterial; all metadata must still match.
                from_task = validate_artifact_metadata(
                    result.get("artifacts"), expected
                )
                if from_task != actual:
                    raise ValueError("Task and artifact endpoint metadata disagree")
            verify_uart_declarations(uart, expected, actual)
            artifacts = directory / "artifacts"
            artifacts.mkdir(exist_ok=True)
            save(
                artifacts / "manifest.json",
                {"id": record["task_id"], "artifacts": list(actual.values())},
            )
            record["artifact_manifest_sha256"] = sha(artifacts / "manifest.json")
            for name, item in actual.items():
                path = artifacts / (name + ".bin")
                if (
                    not path.exists()
                    or path.stat().st_size != item["nbytes"]
                    or sha(path) != item["sha256"]
                ):
                    cli(
                        ["download", record["task_id"], name, str(path)],
                        directory,
                        "download-" + name,
                    )
                if path.stat().st_size != item["nbytes"] or sha(path) != item["sha256"]:
                    raise ValueError("Downloaded length or SHA256 mismatch: " + name)
            record["artifacts"] = list(actual.values())
            record["status"] = "ARTIFACTS_DOWNLOADED"
            save(record_path, record)
        except StopSubmissions:
            record["status"] = "CONNECTION_ERROR"
            save(record_path, record)
            raise
        except (ValueError, KeyError, OSError) as error:
            record.update(status="ARTIFACT_FAIL", error=str(error))
        else:
            # Native replay uses the original model sources; confirm they still
            # match this compiled image before and after executing the oracle.
            if defer_local_validation:
                record["status"] = "DOWNLOADED_FOR_MERGE"
            else:
                replay_locally(root, case, directory, record)
    save(record_path, record)
    print(
        "RESULT",
        root.name,
        case,
        record["status"],
        "fpga=" + str(record.get("fpga")),
        "cycles=" + str(record["kernel_cycles"]),
        flush=True,
    )
    if record["status"] == "PLATFORM_ERROR":
        raise StopSubmissions(
            "Platform error; inspect health before new submissions: "
            + record["task_id"]
        )
    return record


def main():
    global BASE, ENDPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--kernels",
        type=Path,
        help="Compiled case directory root (default MODEL/build/kernels)",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--client", default=CLIENT, help="scheduler executable (FPGA_SCHEDULER_CLIENT)"
    )
    parser.add_argument(
        "--endpoint",
        default=ENDPOINT,
        help="selected fpga112 scheduler URL (FPGA_SCHEDULER_URL)",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=TOKEN,
        help="local fpga112 token file (FPGA_SCHEDULER_TOKEN_FILE); contents are never logged",
    )
    args = parser.parse_args()
    ENDPOINT = args.endpoint
    BASE = [
        args.client,
        "--endpoint",
        ENDPOINT,
        "--token-file",
        str(args.token_file.expanduser().resolve()),
        "--json",
    ]
    if not 1 <= args.timeout <= 86400:
        parser.error("--timeout must be between 1 and 86400")
    root = REPO / "examples/BOSCAMEFPGA" / args.model
    kernels = (args.kernels or root / "build/kernels").resolve()
    output = (
        args.output or root / "build/board-runs/fpga112-returned-outputs-20261103"
    ).resolve()
    legacy_roots = [
        (
            REPO
            / "examples/BOSCAMEFPGA"
            / model
            / "build/board-runs/fpga112-60s-20261103"
        ).resolve()
        for model in MODELS
    ]
    if any(output == legacy or legacy in output.parents for legacy in legacy_roots):
        parser.error(
            "--output must not modify the immutable original 136-case evidence"
        )
    output.mkdir(parents=True, exist_ok=True)
    cases = args.case or sorted(path.parent.name for path in root.glob("*/kernel.py"))
    if len(cases) != len(set(cases)) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", case) for case in cases
    ):
        parser.error("--case values must be unique directory names")
    success = True
    try:
        for case in cases:
            record = run_case(root, case, output, args.timeout, kernels)
            success = success and record["status"] == "PASS"
    except (StopSubmissions, ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr, flush=True)
        success = False
    finally:
        records = [
            json.loads(path.read_text())
            for path in sorted(output.glob("*/result.json"))
        ]
        save(
            output / "report.json",
            {
                "server": "fpga112",
                "endpoint": ENDPOINT,
                "counts": dict(Counter(r["status"] for r in records)),
                "cases": records,
            },
        )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

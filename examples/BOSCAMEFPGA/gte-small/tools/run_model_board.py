#!/usr/bin/env python3
"""Run the real full GTE image; download and validate every result locally."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from validate_model_emitted import check_reference, validate

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SIBLING_CLIENT = REPO.parent / "fpga-scheduler/.venv/bin/fpga-scheduler"
BASE = [
    str(SIBLING_CLIENT) if SIBLING_CLIENT.is_file() else "fpga-scheduler",
    "--endpoint",
    "http://127.0.0.1:18762",
    "--token-file",
    str(REPO / ".agent/secrets/fpga112.token"),
    "--json",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2) + "\n")


def query(args, out, name):
    p = subprocess.run(BASE + args, capture_output=True, text=True, check=False)
    (out / (name + ".stdout")).write_text(p.stdout)
    (out / (name + ".stderr")).write_text(p.stderr)
    if p.returncode:
        raise RuntimeError(
            "Scheduler connection/query failed; stop new submissions and recover same ID: "
            + p.stderr[-1000:]
        )
    return json.loads(p.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--client",
        default=os.environ.get("FPGA_SCHEDULER_CLIENT", BASE[0]),
        help="scheduler client executable; defaults to sibling checkout or PATH",
    )
    p.add_argument(
        "--endpoint",
        default=os.environ.get("FPGA_SCHEDULER_URL", BASE[2]),
    )
    p.add_argument(
        "--token-file",
        type=Path,
        default=os.environ.get("FPGA_SCHEDULER_TOKEN_FILE", BASE[4]),
    )
    p.add_argument("--build", type=Path, default=ROOT / "build/full-model")
    p.add_argument(
        "--output", type=Path, default=ROOT / "build/full-model-board/fpga112-attempt1"
    )
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument(
        "--capture-only",
        action="store_true",
        help="Download diagnostic outputs without full-model comparison",
    )
    p.add_argument(
        "--emitted-reference", type=Path, default=ROOT / "build/full-model-host-emitted"
    )
    p.add_argument(
        "--kernels", type=Path, default=ROOT / "build/scheduler-output/kernels"
    )
    p.add_argument(
        "--validate-only",
        type=Path,
        help="Offline: compare an existing downloaded run without scheduler access",
    )
    p.add_argument(
        "--validation-output",
        type=Path,
        help="New offline report path; preserve previous evidence",
    )
    a = p.parse_args()
    a.build = a.build.resolve()
    a.kernels = a.kernels.resolve()
    if a.validate_only:
        if not a.validation_output:
            p.error("--validate-only requires --validation-output")
        return validate(
            a.validate_only.resolve(),
            a.emitted_reference.resolve(),
            a.kernels,
            a.validation_output.resolve(),
        )
    BASE[:] = [
        a.client,
        "--endpoint",
        a.endpoint,
        "--token-file",
        str(a.token_file),
        "--json",
    ]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = out / "result.json"
    if state.exists():
        record = json.loads(state.read_text())
        ident = record["task_id"]
        for name, digest in record["frozen_files"].items():
            if sha(out / name) != digest:
                raise ValueError("Frozen model evidence changed: " + name)
        task = query(["job", ident], out, "recovery-query")
        if task.get("state") not in ("completed", "failed", "cancelled"):
            raise RuntimeError(
                "Existing model job nonterminal; never duplicate " + ident
            )
    else:
        build = json.loads((a.build / "build-report.json").read_text())
        image = a.build / "gte-small-full.bin"
        elf = a.build / "gte-small-full.elf"
        audit = json.loads((a.build / "elf-audit.json").read_text())
        assert (
            sha(image) == build["image_sha256"]
            and sha(elf) == build["elf_sha256"] == audit["elf_sha256"]
            and audit["status"] == "PASS"
        )
        for name, digest in build["sources"].items():
            assert sha(REPO / name) == digest, ("source changed", name)
        frozen = {}
        for name in (
            "gte-small-full.bin",
            "gte-small-full.elf",
            "build-report.json",
            "elf-audit.json",
            "model-input.json",
            "operator_trace.json",
            "model_launcher.c",
            "kernel_wrappers.c",
        ):
            shutil.copyfile(a.build / name, out / name)
            frozen[name] = sha(out / name)
        if (a.build / "trace-layout.json").exists():
            name = "trace-layout.json"
            shutil.copyfile(a.build / name, out / name)
            frozen[name] = sha(out / name)
        for name in build["sources"]:
            dest = out / "sources" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, dest)
        if not a.capture_only:
            host = a.emitted_reference.resolve()
            contract = json.loads((out / "model-input.json").read_text())
            oracle = check_reference(
                host, a.kernels, build, contract, out / "model_launcher.c"
            )
            names = (
                ["manifest.json"]
                + [item["name"] + ".bin" for item in oracle["files"]]
                + list(oracle["frozen_source_files"].values())
            )
            for name in names:
                dest = out / "reference-emitted" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(host / name, dest)
                frozen["reference-emitted/" + name] = sha(dest)
        health = query(["health"], out, "health-before")
        devices = query(["devices"], out, "devices-before")
        if (
            health.get("stopping")
            or health.get("scheduler_error")
            or not health.get("dispatcher_alive")
        ):
            raise RuntimeError("Service unhealthy; no submission")
        if not any(
            d.get("available") and d.get("state") == "READY" for d in devices["devices"]
        ):
            raise RuntimeError("No READY device; no submission")
        ident = uuid.uuid4().hex
        command = BASE + [
            "run",
            str(out / "gte-small-full.bin"),
            "--id",
            ident,
            "--name",
            "gte-small/full-checkpoint-12layers",
            "--timeout",
            str(a.timeout),
            "--queue-timeout",
            "3600",
        ]
        record = {
            "task_id": ident,
            "server": "fpga112",
            "endpoint": a.endpoint,
            "status": "SUBMITTING",
            "execution_timeout_seconds": a.timeout,
            "image_sha256": sha(image),
            "image_bytes": image.stat().st_size,
            "frozen_files": frozen,
            "command": command,
        }
        save(state, record)
        print("MODEL_RUN", ident, "bytes", image.stat().st_size, flush=True)
        begin = time.monotonic()
        heartbeat = begin
        with (
            (out / "client.stdout").open("w") as stdout,
            (out / "client.stderr").open("w") as stderr,
        ):
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while proc.poll() is None:
                time.sleep(2)
                if time.monotonic() - heartbeat >= 60:
                    print(
                        "MODEL_CLIENT_ALIVE",
                        ident,
                        "wall_seconds",
                        round(time.monotonic() - begin),
                        flush=True,
                    )
                    heartbeat = time.monotonic()
        record.update(
            client_returncode=proc.returncode,
            wall_seconds_including_load_queue=time.monotonic() - begin,
        )
        try:
            task = json.loads((out / "client.stdout").read_text())
            assert task["id"] == ident and "state" in task
        except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
            record["status"] = "CONNECTION_ERROR"
            save(state, record)
            task = query(["job", ident], out, "connection-recovery")
            save(out / "task.json", task)
            raise RuntimeError(
                "CLI disconnected; stop new submissions. Preserved ID " + ident
            )
    save(out / "task.json", task)
    result = task.get("result") or {}
    details = result.get("details") or {}
    uart = result.get("uart_output") or result.get("output") or ""
    (out / "uart.log").write_text(uart)
    record.update(
        state=task.get("state"),
        fpga=task.get("fpga"),
        execution_status=result.get("execution_status"),
        exit_code=result.get("exit_code"),
        app_exit_code=result.get("app_exit_code"),
        details=details,
        scheduler_seconds=result.get("seconds"),
    )
    match = re.search(
        r"model_calls=(\d+) active_case=(\d+) kernel_cycles=0x([0-9a-fA-F]+) model_cycles=0x([0-9a-fA-F]+)",
        uart,
    )
    if match:
        record.update(
            operator_calls=int(match[1]),
            active_case=int(match[2]),
            kernel_cycles=int(match[3], 16),
            model_cycles=int(match[4], 16),
        )
    expected_calls = json.loads((out / "build-report.json").read_text())[
        "expected_calls"
    ]
    if not (
        task.get("state") == "completed"
        and result.get("execution_status") == "exited"
        and result.get("app_exit_code") == 0
        and result.get("exit_code") == 0
        and details.get("application_started") is True
        and details.get("reset_confirmed") is True
        and details.get("ddr_readback_matches") is True
        and details.get("image_sha256") == record["image_sha256"]
        and record.get("operator_calls") == expected_calls
    ):
        record["status"] = (
            "BOARD_" + str(result.get("execution_status", "FAIL")).upper()
        )
        save(state, record)
        print(json.dumps(record, indent=2))
        return 1
    listing = query(["artifacts", ident], out, "artifacts-list")
    items = listing["artifacts"]
    build = json.loads((out / "build-report.json").read_text())
    expected = {
        k: (v["shape"], v["nbytes"])
        for k, v in build.get(
            "outputs",
            {
                "sentence_embedding": {"shape": [384], "nbytes": 1536},
                "last_hidden_state": {"shape": [128, 384], "nbytes": 196608},
            },
        ).items()
    }
    assert len(items) == len(expected) and {d["name"] for d in items} == set(expected)
    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    save(artifacts / "manifest.json", listing)
    for item in items:
        shape, nbytes = expected[item["name"]]
        assert (
            item["shape"] == shape
            and item["nbytes"] == nbytes
            and item["dtype"] == "<f4"
        )
        dest = artifacts / (item["name"] + ".bin")
        if not dest.exists() or sha(dest) != item["sha256"]:
            query(
                ["download", ident, item["name"], str(dest)],
                out,
                "download-" + item["name"],
            )
        assert dest.stat().st_size == nbytes and sha(dest) == item["sha256"]
    record["artifacts"] = items
    record["status"] = "DOWNLOADED"
    save(state, record)
    if a.capture_only:
        record["status"] = "CAPTURED_DIAGNOSTIC_OUTPUTS"
        save(state, record)
        print(
            "MODEL_DIAGNOSTIC_CAPTURE",
            ident,
            "fpga",
            record["fpga"],
            "calls",
            record["operator_calls"],
            flush=True,
        )
        return 0
    host = out / "reference-emitted"
    if not (host / "manifest.json").exists():
        host = a.emitted_reference.resolve()
    validation = validate(out, host, a.kernels)
    record["status"] = "PASS" if validation == 0 else "LOCAL_VALIDATION_FAIL"
    record["local_validation_report"] = "emitted-validation.json"
    save(state, record)
    print("MODEL_RESULT", ident, record["status"], "fpga", record["fpga"], flush=True)
    return validation


if __name__ == "__main__":
    raise SystemExit(main())

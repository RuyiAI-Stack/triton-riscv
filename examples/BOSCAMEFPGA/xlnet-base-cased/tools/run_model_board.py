#!/usr/bin/env python3
"""Run XLNet via the selected fpga112 scheduler; check downloaded tensors locally."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SIBLING_CLIENT = REPO.parent / "fpga-scheduler/.venv/bin/fpga-scheduler"
DEFAULT_CLIENT = os.environ.get(
    "FPGA_SCHEDULER_CLIENT",
    str(SIBLING_CLIENT) if SIBLING_CLIENT.is_file() else "fpga-scheduler",
)
DEFAULT_ENDPOINT = os.environ.get("FPGA_SCHEDULER_URL", "http://127.0.0.1:18762")
DEFAULT_TOKEN_FILE = Path(
    os.environ.get(
        "FPGA_SCHEDULER_TOKEN_FILE", str(REPO / ".agent/secrets/fpga112.token")
    )
)
# Diagnostic runners import this list too; keep its identity when configuring it.
BASE = [
    DEFAULT_CLIENT,
    "--endpoint",
    DEFAULT_ENDPOINT,
    "--token-file",
    str(DEFAULT_TOKEN_FILE),
    "--json",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def query(arguments, out, label):
    result = subprocess.run(
        BASE + arguments, check=False, text=True, capture_output=True
    )
    (out / (label + ".stdout")).write_text(result.stdout)
    (out / (label + ".stderr")).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(
            "scheduler query/connection failed; preserve the same task ID: "
            + result.stderr[-1200:]
        )
    return json.loads(result.stdout)


def validate(out, report_path=None, record=None):
    """Compare frozen emitted outputs only; this function never submits a job."""
    layout = json.loads((out / "output-layout.json").read_text())
    contract = json.loads((out / "validation-contract.json").read_text())
    record = record or json.loads((out / "result.json").read_text())
    assert record["state"] == "completed" and record["execution_status"] == "exited"
    assert record["app_exit_code"] == record["exit_code"] == 0
    listing = json.loads((out / "artifacts/manifest.json").read_text())
    assert listing["id"] == record["task_id"]
    artifacts = {item["name"]: item for item in listing["artifacts"]}
    assert set(artifacts) == {entry["name"] for entry in layout}
    records = []
    passed = True
    for entry in layout:
        name = entry["name"]
        artifact_path = out / "artifacts" / (name + ".bin")
        reference_path = out / "reference-emitted" / (name + ".bin")
        item = artifacts[name]
        assert all(item[key] == entry[key] for key in ("shape", "dtype", "nbytes"))
        assert (
            artifact_path.stat().st_size == entry["nbytes"]
            and sha(artifact_path) == item["sha256"]
        )
        assert reference_path.stat().st_size == entry["nbytes"]
        frozen_reference = record["frozen_files"]["reference-emitted/" + name + ".bin"]
        assert sha(reference_path) == frozen_reference
        actual = np.fromfile(artifact_path, dtype="<f4").reshape(entry["shape"])
        emitted = np.fromfile(reference_path, dtype="<f4").reshape(entry["shape"])
        differences = np.argwhere(actual.view("<u4") != emitted.view("<u4"))
        bit_exact = len(differences) == 0
        conforms = bool(np.isfinite(actual).all() and bit_exact)
        delta = actual.astype(np.float64) - emitted.astype(np.float64)
        records.append(
            {
                "name": name,
                "status": "PASS" if conforms else "FAIL",
                "bit_exact": bit_exact,
                "different_elements": len(differences),
                "downloaded_sha256": sha(artifact_path),
                "reference_sha256": frozen_reference,
                "vs_emitted_scalar_exact_integer_dot": {
                    "max_abs": float(np.max(np.abs(delta))),
                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                },
                "first_differences": [
                    {
                        "index": i.tolist(),
                        "actual_bits": f"0x{int(actual.view('<u4')[tuple(i)]):08x}",
                        "expected_bits": f"0x{int(emitted.view('<u4')[tuple(i)]):08x}",
                    }
                    for i in differences[:8]
                ],
            }
        )
        passed &= conforms
    report = {
        "status": "PASS" if passed else "FAIL",
        "full_layers": 12,
        "evidence": "actual scheduler-downloaded full-model tensors; all numerical checks local",
        "conformance_reference": "actual C graph and emitted scalar LLVM on host; AME replaced by exact signed-i8/i32 dot",
        "lm_head": "logits" in {e["name"] for e in layout},
        "contract_declared_before_submission": contract,
        "strict_bitwise_audit_required_for_this_report": True,
        "board_oracle_executed": False,
        "files": records,
    }
    report_path = report_path or out / "local-validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--client",
        default=DEFAULT_CLIENT,
        help="scheduler CLI; default: FPGA_SCHEDULER_CLIENT or sibling checkout/PATH",
    )
    p.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="fpga112 URL; default: FPGA_SCHEDULER_URL or local port 18762",
    )
    p.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="fpga112 token path; default: FPGA_SCHEDULER_TOKEN_FILE or repository secret path",
    )
    p.add_argument("--build", type=Path, default=ROOT / "build/model-bin")
    p.add_argument("--reference-emitted", type=Path, default=ROOT / "build/model-host")
    # Keep historical commands parseable without requiring auxiliary references.
    p.add_argument("--reference-numpy", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--model-input", type=Path, help=argparse.SUPPRESS)
    p.add_argument(
        "--weights-audit",
        type=Path,
        help="freeze a checkpoint/q/scale provenance audit",
    )
    p.add_argument(
        "--validate-only",
        type=Path,
        help="offline: compare an existing downloaded run without scheduler access",
    )
    p.add_argument(
        "--validation-output",
        type=Path,
        help="write the offline comparison to a new report path",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "build/model-board/fpga112-encoder"
    )
    p.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args()
    if a.validate_only:
        if not a.validation_output:
            p.error(
                "--validate-only requires --validation-output to preserve the original report"
            )
        if (
            a.validation_output.resolve()
            == a.validate_only.resolve() / "local-validation.json"
        ):
            p.error("offline review must use a new report path")
        report = validate(a.validate_only.resolve(), a.validation_output.resolve())
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
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
        for name, checksum in record["frozen_files"].items():
            if sha(out / name) != checksum:
                raise ValueError("frozen evidence changed: " + name)
        task = query(["job", ident], out, "recovery-query")
        if task.get("state") not in ("completed", "failed", "cancelled"):
            raise RuntimeError("existing task nonterminal; do not duplicate " + ident)
    else:
        build = json.loads((a.build / "manifest.json").read_text())
        image = a.build / "xlnet_model.bin"
        assert sha(image) == build["bin_sha256"] and build["audit"]["status"] == "PASS"
        for name, checksum in build["source_sha256"].items():
            assert sha(ROOT / name) == checksum, "model source changed: " + name
        frozen = {}

        def freeze(source, target):
            destination = out / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            frozen[target] = sha(destination)

        for name in [
            "xlnet_model.bin",
            "xlnet_model.elf",
            "input.bin",
            "manifest.json",
            "elf-audit.json",
            "output-layout.json",
            "params-layout.json",
            "model_artifacts.h",
            "kernel-case-map.json",
        ]:
            freeze(a.build / name, name)
        for name in build["source_sha256"]:
            freeze(ROOT / name, "sources/" + name)
        layout = json.loads((a.build / "output-layout.json").read_text())
        for entry in layout:
            name = entry["name"]
            freeze(
                a.reference_emitted / (name + ".bin"),
                "reference-emitted/" + name + ".bin",
            )
        freeze(a.reference_emitted / "report.json", "reference-emitted/report.json")
        if a.weights_audit:
            audit = json.loads(a.weights_audit.read_text())
            assert audit["status"] == "PASS"
            assert audit["lm_head"] == build.get("lm_head", False)
            freeze(a.weights_audit, "weights-audit.json")
        freeze(Path(__file__), "run_model_board.py")
        save(
            out / "validation-contract.json",
            {
                "comparison": "bitwise equality",
                "atol": 0,
                "rtol": 0,
                "required": "all elements finite and bytes equal for every returned tensor against frozen emitted-scalar/exact-integer-dot reference",
            },
        )
        frozen["validation-contract.json"] = sha(out / "validation-contract.json")
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
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
        )
        (out / "ssh-preflight.stderr").write_text(ssh.stderr)
        if ssh.returncode:
            raise RuntimeError(
                "ascend via host to fpga112 SSH unreachable; restore the host tunnel: "
                + ssh.stderr[-500:]
            )
        health = query(["health"], out, "health-before")
        devices = query(["devices"], out, "devices-before")
        if (
            health.get("stopping")
            or health.get("scheduler_error")
            or not health.get("dispatcher_alive")
        ):
            raise RuntimeError("scheduler unhealthy; no submission")
        if not any(
            d.get("available") and d.get("state") == "READY" for d in devices["devices"]
        ):
            raise RuntimeError("no READY device; no submission")
        ident = uuid.uuid4().hex
        command = BASE + [
            "run",
            str(out / "xlnet_model.bin"),
            "--id",
            ident,
            "--name",
            "xlnet-base-cased/12layers-real-checkpoint",
            "--timeout",
            str(a.timeout),
            "--queue-timeout",
            "3600",
        ]
        record = {
            "task_id": ident,
            "server": "fpga112",
            "status": "SUBMITTING",
            "frozen_files": frozen,
            "image_sha256": build["bin_sha256"],
            "image_bytes": image.stat().st_size,
            "execution_timeout_seconds": a.timeout,
            "command": command,
        }
        save(state, record)
        print("MODEL_RUN", ident, "bytes", image.stat().st_size, flush=True)
        start = last_update = time.monotonic()
        with (
            (out / "client.stdout").open("w") as stdout,
            (out / "client.stderr").open("w") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                time.sleep(2)
                if time.monotonic() - last_update >= 60:
                    print(
                        "MODEL_CLIENT_ALIVE",
                        ident,
                        round(time.monotonic() - start),
                        flush=True,
                    )
                    last_update = time.monotonic()
        record.update(
            client_returncode=process.returncode,
            wall_seconds_including_queue_and_load=time.monotonic() - start,
        )
        try:
            task = json.loads((out / "client.stdout").read_text())
            assert task["id"] == ident and "state" in task
        except Exception:  # noqa: BLE001 - Persist task recovery state before failing.
            record["status"] = "CONNECTION_ERROR"
            save(state, record)
            raise RuntimeError(
                "CLI disconnected; stop new submissions and query preserved task "
                + ident
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
    )
    match = re.search(
        r"layers=(\d+) model_cycles=0x([0-9a-fA-F]+) kernel_calls=0x([0-9a-fA-F]+)",
        uart,
    )
    if match:
        record.update(
            layers=int(match[1]),
            model_cycles=int(match[2], 16),
            kernel_calls=int(match[3], 16),
        )
    ok = (
        task.get("state") == "completed"
        and result.get("execution_status") == "exited"
        and result.get("exit_code") == 0
        and result.get("app_exit_code") == 0
        and details.get("application_started") is True
        and details.get("reset_confirmed") is True
        and details.get("ddr_readback_matches") is True
        and details.get("image_sha256") == record["image_sha256"]
        and record.get("layers") == 12
    )
    if not ok:
        record["status"] = (
            "BOARD_" + str(result.get("execution_status", "FAIL")).upper()
        )
        save(state, record)
        print(json.dumps(record, indent=2))
        return 1
    listing = query(["artifacts", ident], out, "artifacts-list")
    expected = {
        e["name"]: e for e in json.loads((out / "output-layout.json").read_text())
    }
    items = listing["artifacts"]
    assert len(items) == len(expected) and {x["name"] for x in items} == set(expected)
    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    save(artifacts / "manifest.json", listing)
    for item in items:
        entry = expected[item["name"]]
        assert all(item[key] == entry[key] for key in ["shape", "dtype", "nbytes"])
        path = artifacts / (item["name"] + ".bin")
        if not path.exists() or sha(path) != item["sha256"]:
            query(
                ["download", ident, item["name"], str(path)],
                out,
                "download-" + item["name"],
            )
        assert path.stat().st_size == item["nbytes"] and sha(path) == item["sha256"]
    record["artifacts"] = items
    report = validate(out, record=record)
    record["status"] = report["status"]
    record["local_validation"] = report
    save(state, record)
    print("MODEL_RESULT", ident, record["status"], "fpga", record["fpga"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

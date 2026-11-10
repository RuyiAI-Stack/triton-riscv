#!/usr/bin/env python3
"""Run/collect a pinned model image on the explicitly selected fpga112 server.

The foreground CLI remains alive to maintain its lease. Existing jobs can be
collected without resubmission. Full-model phase completion is an explicit
flag, never inferred from a successful image build or prefix probe.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BASE = ROOT / "build/model-integration"
CLIENT = Path("/home/chh/gitprojects/fpga-scheduler/.venv/bin/fpga-scheduler")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def threshold_interpretation(result):
    status = result.get("execution_status")
    if status == "cancelled":
        reason = str(
            (result.get("details") or {}).get("cancel_reason") or "unspecified reason"
        )
        return "scheduler cancellation (" + reason + "); not kernel hang evidence"
    if status == "platform_error":
        return "platform/transport failure; not application hang evidence"
    if status == "timeout":
        return "execution deadline reached; no completed output verification"
    return "completed execution; runtime threshold and numeric checks are reported separately"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + chr(10))


def query(base, args, out, name):
    p = subprocess.run([*base, *map(str, args)], capture_output=True, text=True)
    (out / (name + ".stdout")).write_text(p.stdout)
    (out / (name + ".stderr")).write_text(p.stderr)
    if p.returncode:
        raise RuntimeError(
            "fpga112 scheduler request failed: " + p.stderr[-1000:] + p.stdout[-1000:]
        )
    return json.loads(p.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, default=BASE / "model-image")
    p.add_argument("--reference-dir", type=Path, default=BASE / "physical-plan")
    p.add_argument("--output", type=Path, default=BASE / "board")
    p.add_argument("--endpoint", default="http://127.0.0.1:18762")
    p.add_argument(
        "--token-file", type=Path, default=REPO / ".agent/secrets/fpga112.token"
    )
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--queue-timeout", type=int, default=3600)
    p.add_argument(
        "--lease-seconds",
        type=float,
        help="requested scheduler client lease, without changing daemon settings",
    )
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        help="requested scheduler client heartbeat interval",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        help="HTTP request deadline; separate from FPGA execution timeout",
    )
    p.add_argument("--job")
    p.add_argument("--phase-gates-complete", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.image_dir / "image-manifest.json").read_text())
    if manifest["full_model"] and not a.phase_gates_complete and not a.job:
        raise ValueError(
            "whole-model submission is gated: confirm parent phases 1/2 complete with --phase-gates-complete"
        )
    base = [
        str(CLIENT),
        "--endpoint",
        a.endpoint,
        "--token-file",
        str(a.token_file),
        "--json",
    ]
    if a.lease_seconds is not None:
        base += ["--lease-seconds", str(a.lease_seconds)]
    if a.heartbeat_seconds is not None:
        base += ["--heartbeat-seconds", str(a.heartbeat_seconds)]
    if a.request_timeout is not None:
        base += ["--request-timeout", str(a.request_timeout)]
    if not a.token_file.is_file() or not a.token_file.stat().st_size:
        raise ValueError("selected fpga112 token file missing")
    if a.job:
        ident = a.job
        out = a.output / ident
        out.mkdir(exist_ok=True)
        if (out / "image-manifest.json").exists():
            manifest = json.loads((out / "image-manifest.json").read_text())
    else:
        check = subprocess.run(
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
        )
        if check.returncode:
            raise RuntimeError(
                "ascend 经 host 到 fpga112 的 SSH 链路不可达，请检查并恢复 host 上的隧道。入口127.0.0.1:12222; "
                + check.stderr[-800:]
            )
        query(base, ["health"], a.output, "health")
        devices = query(base, ["devices"], a.output, "devices")
        if not any(d["state"] == "READY" for d in devices.get("devices", [])):
            raise RuntimeError(
                "fpga112 scheduler has no READY device; do not submit until READY"
            )
        image = a.image_dir / "efficientnet_b3_model.bin"
        if sha(image) != manifest["image_sha256"]:
            raise ValueError("image differs from audited manifest")
        ident = uuid.uuid4().hex
        out = a.output / ident
        out.mkdir()
        snapshot = out / image.name
        shutil.copy2(image, snapshot)
        shutil.copy2(a.image_dir / "image-manifest.json", out / "image-manifest.json")
        if sha(snapshot) != manifest["image_sha256"]:
            raise ValueError("image changed during snapshot")
        submission = {
            "task_id": ident,
            "server": "fpga112",
            "full_model": manifest["full_model"],
            "physical_calls": manifest["physical_calls"],
            "timeout_seconds": a.timeout,
            "requested_client_lease_seconds": a.lease_seconds,
            "requested_client_heartbeat_seconds": a.heartbeat_seconds,
            "http_request_timeout_seconds": a.request_timeout,
            "image_sha256": manifest["image_sha256"],
            "image_bytes": snapshot.stat().st_size,
            "submitted_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "image_manifest_sha256": sha(a.image_dir / "image-manifest.json"),
        }
        save(out / "submission.json", submission)
        cmd = [
            *base,
            "run",
            str(snapshot),
            "--id",
            ident,
            "--name",
            ("efficientnet-model" if manifest["full_model"] else "efficientnet-prefix")
            + "-c",
            "--timeout",
            str(a.timeout),
            "--queue-timeout",
            str(a.queue_timeout),
        ]
        print(
            "SUBMIT",
            ident,
            "calls",
            manifest["physical_calls"],
            "timeout",
            a.timeout,
            "sha",
            manifest["image_sha256"],
            flush=True,
        )
        # Never kill this process to replace it with wait: it owns the lease.
        with (
            (out / "run.stdout").open("w") as stdout,
            (out / "run.stderr").open("w") as stderr,
        ):
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)
            submission["client_pid"] = proc.pid
            save(out / "submission.json", submission)
            submission["client_exit_code"] = proc.wait()
        save(out / "submission.json", submission)
    task = query(base, ["job", ident], out, "job")
    save(out / "task.json", task)
    result = task.get("result") or {}
    details = result.get("details") or {}
    uart = result.get("uart_output", result.get("output", "")) or ""
    (out / "uart.log").write_text(uart)
    artifact = query(base, ["artifacts", ident], out, "artifacts")
    save(out / "artifacts.json", artifact)
    downloads = []
    download_dir = out / "downloads"
    download_dir.mkdir(exist_ok=True)
    for item in artifact.get("artifacts", []):
        name = item["name"]
        destination = download_dir / Path(name).name
        query(
            base,
            ["download", ident, name, destination],
            out,
            "download-" + Path(name).name,
        )
        downloads.append(
            {
                "name": name,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha(destination),
            }
        )
    comparisons = []
    references = {
        "logits_f32": a.reference_dir / "w8a8-logits.f32",
        "stem_f32": a.reference_dir / "stem.f32",
        "depthwise_f32": a.reference_dir / "depthwise.f32",
        "pointwise_acc_i32": a.reference_dir / "pointwise-acc.i32",
    }
    for item in downloads:
        expected = references.get(item["name"])
        if expected is None or not expected.exists():
            continue
        dtype = "<i4" if item["name"].endswith("_i32") else "<f4"
        x = np.fromfile(item["path"], dtype=dtype)
        y = np.fromfile(expected, dtype=dtype)
        if x.shape != y.shape:
            raise ValueError("download shape mismatch " + item["name"])
        error = np.abs(x.astype(np.float64) - y.astype(np.float64))
        comp = {
            "name": item["name"],
            "elements": x.size,
            "expected_sha256": sha(expected),
            "bit_exact": bool(np.array_equal(x.view(np.uint8), y.view(np.uint8))),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
        if dtype == "<f4":
            threshold = manifest["diagnostic_validation_threshold"]
            comp.update(
                {
                    "finite": bool(np.isfinite(x).all()),
                    "diagnostic_allclose_atol_0_05_rtol_0_01": bool(
                        np.allclose(x, y, atol=0.05, rtol=0.01)
                    ),
                    "declared_threshold_passed": bool(
                        np.isfinite(x).all()
                        and np.allclose(
                            x, y, atol=threshold["atol"], rtol=threshold["rtol"]
                        )
                    ),
                    "tighter_allclose_atol_2e_4_rtol_2e_4": bool(
                        np.allclose(x, y, atol=2e-4, rtol=2e-4)
                    ),
                    "strict_allclose_atol_2e_6_rtol_2e_6": bool(
                        np.allclose(x, y, atol=2e-6, rtol=2e-6)
                    ),
                    "actual_top1": int(np.argmax(x)),
                    "expected_top1": int(np.argmax(y)),
                }
            )
        comparisons.append(comp)
    cycles = re.search(r"model_cycles=(0x[0-9a-fA-F]+|[0-9]+)", uart)
    kernels = re.findall(
        re.escape("[efficientnet-cycle]")
        + r" call=([0-9]+) cycles=(0x[0-9a-fA-F]+|[0-9]+)",
        uart,
    )
    samples = {}
    for i, c in kernels:
        samples.setdefault(int(i), []).append(int(c, 0))
    per_call = [
        {
            "call": i,
            "cycles": values[-1],
            "valid": values[-1] > 0,
            "samples_consistent": len(set(values)) == 1,
        }
        for i, values in sorted(samples.items())
    ]
    submission_path = out / "submission.json"
    submitted = (
        json.loads(submission_path.read_text()) if submission_path.exists() else {}
    )
    summary = {
        "task_id": ident,
        "server": "fpga112",
        "fpga": task.get("fpga"),
        "state": task.get("state"),
        "full_model": manifest["full_model"],
        "physical_calls": manifest["physical_calls"],
        "execution_status": result.get("execution_status"),
        "app_exit_code": result.get("app_exit_code"),
        "exit_code": result.get("exit_code"),
        "scheduler_seconds": result.get("seconds"),
        "details": details,
        "image_sha256": manifest["image_sha256"],
        "scheduler_image_hash_match": details.get("image_sha256")
        == manifest["image_sha256"],
        "model_cycles": int(cycles.group(1), 0) if cycles else None,
        "per_call_cycles": per_call,
        "timeout_seconds": submitted.get("timeout_seconds"),
        "requested_client_lease_seconds": submitted.get(
            "requested_client_lease_seconds"
        ),
        "requested_client_heartbeat_seconds": submitted.get(
            "requested_client_heartbeat_seconds"
        ),
        "http_request_timeout_seconds": submitted.get("http_request_timeout_seconds"),
        "downloads": downloads,
        "comparisons": comparisons,
        "threshold_interpretation": threshold_interpretation(result),
        "scope": "whole-model FPGA run"
        if manifest["full_model"]
        else "bounded operator/model-prefix probe; not whole-model evidence",
        "image_scope": manifest.get("scope"),
    }
    summary["numerical_validation_location"] = manifest.get(
        "numerical_validation_location", "inline and host"
    )
    summary["reference_provenance"] = manifest.get("reference_provenance")
    summary["reference_logits_sha256"] = manifest.get("reference_logits_sha256")
    summary["required_artifacts_present"] = set(
        manifest.get("output_artifacts", ["logits_f32", "call_cycles_u64"])
    ) <= {item["name"] for item in downloads}
    save(out / "summary.json", summary)
    save(a.output / "latest-summary.json", summary)
    concise = {k: v for k, v in summary.items() if k != "per_call_cycles"}
    concise["per_call_cycles_count"] = len(per_call)
    concise["invalid_live_or_final_cycle_observations"] = sum(
        not r["valid"] or not r["samples_consistent"] for r in per_call
    )
    print(json.dumps(concise, indent=2, ensure_ascii=False))
    if (
        summary["state"] != "completed"
        or summary["execution_status"] != "exited"
        or summary["app_exit_code"] != 0
        or summary["exit_code"] != 0
        or not summary["required_artifacts_present"]
        or not comparisons
        or any(
            not r.get("declared_threshold_passed", r["bit_exact"]) for r in comparisons
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

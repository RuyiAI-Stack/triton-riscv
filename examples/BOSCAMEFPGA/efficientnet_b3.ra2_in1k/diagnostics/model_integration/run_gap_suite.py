#!/usr/bin/env python3
"""Queue and retain full-shape GAP probe leases, then download every output."""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent / "conv_profile")]
from cases import inventory  # noqa: E402
from build_model_image import sha  # noqa: E402
from run_profile import preflight  # noqa: E402

CLIENT = Path("/home/chh/gitprojects/fpga-scheduler/.venv/bin/fpga-scheduler")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--probes",
        type=Path,
        default=ROOT / "build/model-integration/gap-all13-probes-20261104",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/model-integration/gap-all13-board-20261104",
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--overlay",
        type=Path,
        default=ROOT / "build/model-integration/gap-phase-overlay",
    )
    p.add_argument("--source", type=Path, default=HERE / "gap_phase_candidate.c")
    p.add_argument(
        "--reference-semantics", default="original emitted LLVM reduction order"
    )
    p.add_argument("--lease-seconds", type=float, default=300)
    p.add_argument("--heartbeat-seconds", type=float, default=2)
    p.add_argument("--request-timeout", type=float, default=180)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    base = [
        str(CLIENT),
        "--endpoint",
        "http://127.0.0.1:18762",
        "--token-file",
        str(REPO / ".agent/secrets/fpga112.token"),
        "--json",
        "--lease-seconds",
        str(a.lease_seconds),
        "--heartbeat-seconds",
        str(a.heartbeat_seconds),
        "--request-timeout",
        str(a.request_timeout),
    ]
    preflight(base, a.output, 600)

    def one(case):
        name = case["name"]
        dest = a.output / name
        dest.mkdir(exist_ok=True)
        image_dir = a.probes / name / "phase-image"
        manifest = json.loads((image_dir / "image-manifest.json").read_text())
        image = image_dir / "efficientnet_b3_model.bin"
        if sha(image) != manifest["image_sha256"]:
            raise ValueError("image changed " + name)
        if (dest / "latest-summary.json").is_file():
            prior = json.loads((dest / "latest-summary.json").read_text())
            if (
                prior["image_sha256"] == manifest["image_sha256"]
                and prior["state"] == "completed"
                and prior["comparisons"][0]["bit_exact"]
            ):
                prior.update(
                    case=name,
                    kernel_o_sha256=sha(image_dir / "kernel.o"),
                    source_sha256=sha(a.source),
                    reference_semantics=a.reference_semantics,
                    call_cycles=prior["per_call_cycles"][0]["cycles"],
                )
                return name, prior
        ident = uuid.uuid4().hex
        out = dest / ident
        out.mkdir()
        snapshot = out / image.name
        shutil.copy2(image, snapshot)
        shutil.copy2(image_dir / "image-manifest.json", out / "image-manifest.json")
        record = {
            "task_id": ident,
            "server": "fpga112",
            "full_model": False,
            "physical_calls": 1,
            "timeout_seconds": 60,
            "image_sha256": sha(snapshot),
            "image_bytes": snapshot.stat().st_size,
            "submitted_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }
        command = [
            *base,
            "run",
            str(snapshot),
            "--id",
            ident,
            "--name",
            "efficientnet-gap-" + name,
            "--timeout",
            "60",
            "--queue-timeout",
            "3600",
        ]
        with (
            (out / "run.stdout").open("w") as stdout,
            (out / "run.stderr").open("w") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            record["client_pid"] = process.pid
            (out / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
            record["client_exit_code"] = process.wait()
        (out / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
        collected = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_collect.py"),
                "--image-dir",
                str(image_dir),
                "--reference-dir",
                str(a.probes / name),
                "--output",
                str(dest),
                "--job",
                ident,
                "--request-timeout",
                str(a.request_timeout),
            ],
            capture_output=True,
            text=True,
        )
        (dest / "collect.log").write_text(collected.stdout + collected.stderr)
        if collected.returncode:
            raise RuntimeError(
                "collection failed " + name + " " + collected.stderr[-1000:]
            )
        summary = json.loads((dest / "latest-summary.json").read_text())
        assert (
            summary["state"] == "completed"
            and summary["execution_status"] == "exited"
            and summary["app_exit_code"] == summary["exit_code"] == 0
        ), name
        assert summary["comparisons"][0]["bit_exact"], name
        expected = a.overlay / name / "kernel.o"
        actual = image_dir / "kernel.o"
        if sha(expected) != sha(actual):
            raise ValueError("probe object differs " + name)
        summary.update(
            case=name,
            kernel_o_sha256=sha(actual),
            source_sha256=sha(a.source),
            reference_semantics=a.reference_semantics,
            call_cycles=summary["per_call_cycles"][0]["cycles"],
        )
        print("PASS", name, summary["call_cycles"], ident, flush=True)
        return name, summary

    results = {}
    abort = threading.Event()

    def guarded(case):
        if abort.is_set():
            raise RuntimeError(
                "GAP suite stopped after an earlier failure; pending case not submitted"
            )
        try:
            return one(case)
        except BaseException:
            abort.set()
            raise

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed(
            [pool.submit(guarded, c) for c in inventory() if c["family"] == "gap"]
        ):
            name, result = future.result()
            results[name] = result
            (a.output / "result-summary.json").write_text(
                json.dumps(
                    {
                        "status": "PASS" if len(results) == 13 else "RUNNING",
                        "server": "fpga112",
                        "results": [results[k] for k in sorted(results)],
                        "expected": 13,
                        "completed": len(results),
                        "reference_semantics": a.reference_semantics,
                        "scope": "all13 full-shape phase-batched GAP objects; complete DDR outputs",
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

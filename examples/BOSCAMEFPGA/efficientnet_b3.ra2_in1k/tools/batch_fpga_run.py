#!/usr/bin/env python3
"""Run all efficientnet_b3 board bins via fpga-scheduler; classify by UART."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

VERIFY_RE = re.compile(
    r"verify\s+(\S+)\s*:\s*(PASS|FAIL)\s+errors=([0-9a-fA-F]+)"
    r"(?:\s+max_abs_error_f32_bits=([0-9a-fA-F]+))?"
)
CYCLES_RE = re.compile(r"cycles=([0-9a-fA-F]+)")
PLATFORM_RE = re.compile(r"\[effnet\]\s*(PASS|FAIL|TIMEOUT|E6 TRAP)", re.I)
SIMCHECK_RE = re.compile(r"numerical acceptance is simcheck", re.I)


def classify(uart: str) -> dict:
    text = uart or ""
    cycles_m = list(CYCLES_RE.finditer(text))
    verify_m = list(VERIFY_RE.finditer(text))
    plat_m = list(PLATFORM_RE.finditer(text))
    cycles = int(cycles_m[-1].group(1), 16) if cycles_m else None
    platform = plat_m[-1].group(1).upper() if plat_m else None
    if verify_m:
        m = verify_m[-1]
        status = "PASS" if m.group(2) == "PASS" else "NUMERICAL_FAIL"
        return {
            "status": status,
            "cycles": cycles,
            "errors": int(m.group(3), 16),
            "max_abs_error_f32_bits": int(m.group(4), 16) if m.group(4) else None,
            "platform": platform,
        }
    if SIMCHECK_RE.search(text) and platform == "PASS":
        return {
            "status": "PASS",
            "cycles": cycles,
            "platform": platform,
            "note": "onboard numerical check deferred to simcheck",
        }
    if platform == "TIMEOUT":
        return {"status": "TIMEOUT", "cycles": cycles, "platform": platform}
    if platform == "PASS" and cycles is not None:
        # Kernel returned 0 and printed cycles, but no verify/simcheck marker.
        return {
            "status": "PASS",
            "cycles": cycles,
            "platform": platform,
            "note": "platform PASS with cycles; no verify line",
        }
    if platform in ("FAIL", "E6 TRAP"):
        return {"status": "NUMERICAL_FAIL", "cycles": cycles, "platform": platform}
    return {
        "status": "TIMEOUT",
        "cycles": cycles,
        "platform": platform,
        "note": "no PASS/FAIL/TIMEOUT in UART within timeout",
    }


def run_one(client, image: Path, timeout: int, idle_seconds: int) -> dict:
    from fpga_scheduler.client import SchedulerError

    t0 = time.monotonic()
    uart_parts: list[str] = []
    task_id = None
    cancel_reason = None
    try:
        stream = client.run_test(
            image,
            timeout=timeout,
            idle_seconds=idle_seconds,
            queue_timeout=3600,
            stream=True,
            poll_interval=0.2,
        )
        task_id = stream.id
        decided = None
        try:
            for chunk in stream:
                uart_parts.append(chunk)
                blob = "".join(uart_parts)
                # App report: platform line, then verify|simcheck, then cycles=.
                has_accept = bool(VERIFY_RE.search(blob) or SIMCHECK_RE.search(blob))
                has_cycles = bool(CYCLES_RE.search(blob))
                bad_plat = bool(
                    re.search(r"\[effnet\]\s*(TIMEOUT|E6 TRAP)\b", blob, re.I)
                )
                if bad_plat or (has_accept and has_cycles):
                    decided = classify(blob)
                    try:
                        client.cancel(task_id)
                        cancel_reason = "uart_verdict"
                    except SchedulerError:
                        pass
                    break
        except SchedulerError as err:
            uart_parts.append(f"\n[client] {err}\n")
        finally:
            try:
                stream.close()
            except Exception:
                pass
            if cancel_reason is None:
                try:
                    client.cancel(task_id)
                    cancel_reason = "cleanup"
                except SchedulerError:
                    pass

        uart = "".join(uart_parts)
        result = getattr(stream, "result", None)
        exec_status = exit_code = None
        if result is not None:
            out = (
                getattr(result, "uart_output", None)
                or getattr(result, "output", "")
                or ""
            )
            if out and len(out) >= len(uart):
                uart = out
            exec_status = getattr(result, "execution_status", None)
            exit_code = getattr(result, "exit_code", None)
        else:
            try:
                task = client.wait(task_id, timeout=30, poll_interval=0.5)
                if task.result is not None:
                    out = task.result.uart_output or task.result.output or ""
                    if out and len(out) >= len(uart):
                        uart = out
                    exec_status = task.result.execution_status
                    exit_code = task.result.exit_code
            except SchedulerError:
                try:
                    task = client.job(task_id)
                    if task.result is not None:
                        out = task.result.uart_output or task.result.output or ""
                        if out:
                            uart = out
                        exec_status = task.result.execution_status
                        exit_code = task.result.exit_code
                except SchedulerError:
                    pass
        info = decided or classify(uart)
        # If we cancelled right after verify but cycles arrived in same UART, re-parse.
        if info.get("cycles") is None and CYCLES_RE.search(uart):
            info = classify(uart)
        info.update(
            {
                "task_id": task_id,
                "uart": uart,
                "wall_seconds": round(time.monotonic() - t0, 3),
                "execution_status": exec_status,
                "exit_code": exit_code,
                "cancel_reason": cancel_reason,
            }
        )
        return info
    except Exception as err:  # noqa: BLE001 - batch must continue
        if task_id:
            try:
                client.cancel(task_id)
            except Exception:
                pass
        return {
            "status": "ERROR",
            "error": f"{type(err).__name__}: {err}",
            "task_id": task_id,
            "uart": "".join(uart_parts),
            "wall_seconds": round(time.monotonic() - t0, 3),
            "cycles": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--skip-pass",
        action="store_true",
        help="skip cases already PASS in existing report",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="skip PASS and TIMEOUT already recorded (retry ERROR only)",
    )
    parser.add_argument(
        "--retries", type=int, default=2, help="extra attempts on connection ERROR"
    )
    args = parser.parse_args()

    root = args.root
    bins = sorted(root.glob("build/kernels/*/*.bin"))
    if args.only:
        want = set(args.only)
        bins = [b for b in bins if b.stem in want]

    report = {
        "cases": {},
        "meta": {
            "timeout": args.timeout,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root": str(root),
        },
    }
    if args.out.exists() and (args.skip_pass or args.skip_done):
        prev = json.loads(args.out.read_text())
        keep_status = {"PASS", "TIMEOUT"} if args.skip_done else {"PASS"}
        report["cases"] = {
            k: v
            for k, v in prev.get("cases", {}).items()
            if v.get("status") in keep_status
        }

    from fpga_scheduler.client import Client

    endpoint = os.environ.get("FPGA_SCHEDULER_URL", "http://127.0.0.1:18765")
    token_file = os.environ.get("FPGA_SCHEDULER_TOKEN_FILE")

    def make_client():
        return Client(endpoint, token_file=token_file, request_timeout=360)

    client = make_client()

    total = len(bins)
    for i, image in enumerate(bins, 1):
        name = image.stem
        prev = report["cases"].get(name, {})
        if args.skip_done and prev.get("status") in ("PASS", "TIMEOUT"):
            print(f"[{i}/{total}] SKIP {prev['status']} {name}", flush=True)
            continue
        if args.skip_pass and prev.get("status") == "PASS":
            print(f"[{i}/{total}] SKIP PASS {name}", flush=True)
            continue
        print(f"[{i}/{total}] RUN {name} ...", flush=True)
        info = None
        for attempt in range(args.retries + 1):
            try:
                info = run_one(client, image, args.timeout, args.idle_seconds)
            except Exception as err:  # noqa: BLE001
                info = {
                    "status": "ERROR",
                    "error": f"{type(err).__name__}: {err}",
                    "cycles": None,
                    "wall_seconds": 0,
                }
            if info.get("status") != "ERROR":
                break
            err = (info.get("error") or "").lower()
            if (
                "connection" not in err
                and "urlerror" not in err
                and "timeout" not in err
            ):
                break
            print(
                f"[{i}/{total}] {name}: connection ERROR, refresh tunnel/client "
                f"(attempt {attempt + 1}/{args.retries + 1})",
                flush=True,
            )
            time.sleep(2)
            try:
                client.close()
            except Exception:
                pass
            # Best-effort: reopen local SSH forward if env helper is available.
            env_file = Path("/home/caihanyi/work/fpga113-client.env")
            if env_file.exists():
                os.system(f"bash -lc 'source {env_file}' >/dev/null 2>&1")
            client = make_client()
        report["cases"][name] = info
        cyc = info.get("cycles")
        cyc_s = str(cyc) if cyc is not None else "-"
        print(
            f"[{i}/{total}] {name}: {info['status']} cycles={cyc_s} "
            f"wall={info.get('wall_seconds')}s",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # Summary
    counts = {}
    for info in report["cases"].values():
        counts[info["status"]] = counts.get(info["status"], 0) + 1
    report["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["meta"]["counts"] = counts
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("COUNTS:", counts, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

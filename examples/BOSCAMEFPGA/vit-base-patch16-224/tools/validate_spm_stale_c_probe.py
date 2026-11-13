"""Verify existing normal-SPM stale-C task and readmem evidence offline."""

import argparse
import hashlib
from pathlib import Path
import re
import struct

from accumulator_validation import digest, write_json
from board_accumulator import fnv64
from collect_accumulator_evidence import (
    calibration_outputs,
    read_json,
    require,
    verified_task,
)

CALL_PATTERN = re.compile(
    r"\[spm-stale-c\] invocation=(\d+) cached_stale=(\d+) "
    r"cached_current=(\d+) cached_other=(\d+) prewarm_errors=(\d+) "
    r"errors=(\d+) input_errors=(\d+) guard_errors=(\d+) "
    r"output_fnv64=0x([0-9a-fA-F]+) kernel_cycles=0x([0-9a-fA-F]+)"
)


def independent_golden():
    values = []
    for stress in (0, 1):
        for row in range(32):
            for column in range(64):
                value = (0x7FFFFF00 if stress else 0x6A020000) + (
                    row * 768 + column
                ) * 17
                for q in range(768):
                    if stress:
                        a = (127, -128, -1, 0, 1)[(row * 3 + q) % 5]
                        b = (-128, 127, 1, -1, 0)[(column * 7 + q * 3) % 5]
                    else:
                        a = (row * 37 + q * 13 + 7) % 256
                        b = (column * 17 + q * 11 + 211) % 256
                        a = a - 256 if a >= 128 else a
                        b = b - 256 if b >= 128 else b
                    value += a * b
                values.append(value & 0xFFFFFFFF)
    return struct.pack("<" + "I" * len(values), *values)


def validate(args):
    directory = args.manifest.resolve().parent
    manifest = read_json(args.manifest)
    require(
        manifest.get("kind") == "normal-spm-stale-c-correctness-probe"
        and manifest.get("mode") == "normal-post-add"
        and manifest.get("post_add") is True
        and manifest.get("shape") == [32, 64, 768]
        and manifest.get("calls") == 4,
        "unexpected stale-C probe semantic contract",
    )
    image = directory / "spm-stale-c-probe.bin"
    task, uart = verified_task(args.task, image, manifest["image_sha256"])
    require(
        digest(directory / "hot_spm.o") == manifest["target_object_sha256"]
        and digest(directory / "hot_spm.emitted.mlir")
        == manifest["target_emitted_sha256"]
        and digest(directory / "seed.mlir") == manifest["seed_source_sha256"]
        and digest(directory / "seed.o") == manifest["seed_object_sha256"]
        and digest(directory / "harness.c") == manifest["harness_sha256"],
        "frozen target, seed or harness identity changed",
    )
    require(
        all(
            digest(directory / name) == expected
            for name, expected in manifest["objects"].items()
        ),
        "frozen probe object identity changed",
    )
    gold = independent_golden()
    require(
        (directory / "golden.bin").read_bytes() == gold,
        "independent modulo-i32 golden differs",
    )
    expected_hashes = [fnv64(gold[s * 8192 : (s + 1) * 8192]) for s in (0, 1)]
    calls = []
    for match in CALL_PATTERN.findall(uart):
        q, stale, current, other, prewarm, errors, inputs, guards = map(int, match[:8])
        calls.append(
            {
                "invocation": q,
                "stress": int(q >= 2),
                "cached_stale": stale,
                "cached_current": current,
                "cached_other": other,
                "prewarm_errors": prewarm,
                "errors": errors,
                "input_errors": inputs,
                "guard_errors": guards,
                "output_fnv64": match[8].lower().zfill(16),
                "cycles": int(match[9], 16),
            }
        )
    require(
        [entry["invocation"] for entry in calls] == [0, 1, 2, 3],
        "four complete probe invocations are required",
    )
    for call in calls:
        require(
            call["cached_stale"] + call["cached_current"] == 2048
            and call["cached_other"] == call["prewarm_errors"] == 0,
            "cached control/prewarm coverage is invalid",
        )
        require(
            call["cycles"] > 0
            and call["errors"] == call["input_errors"] == call["guard_errors"] == 0
            and call["output_fnv64"] == expected_hashes[call["stress"]],
            "normal-SPM output/input/guard check failed",
        )
    summary = re.findall(
        r"\[spm-stale-c\] exposed_calls=(\d+) returned=(\d+) validated=(\d+) errors=(\d+)",
        uart,
    )
    require(len(summary) == 1, "one final stale-C summary is required")
    exposed = sum(call["cached_stale"] > 0 for call in calls)
    require(
        list(map(int, summary[0])) == [exposed, 4, 4, 0],
        "probe completion/exposure summary differs",
    )
    outputs = calibration_outputs(
        manifest, task, uart, calls, directory, args.artifacts
    )
    require(outputs["status"] == "PASS", "readmem outputs are not bit-exact")
    result = {
        "schema": 1,
        "status": "PASS" if exposed else "INCOMPLETE",
        "scope": "normal fixed-SPM mixed old-C visibility; not profitability calibration",
        "task_id": task["id"],
        "image_sha256": manifest["image_sha256"],
        "manifest_sha256": digest(args.manifest),
        "independent_golden_sha256": hashlib.sha256(gold).hexdigest(),
        "numerical_status": "PASS",
        "cached_stale_exposed_calls": exposed,
        "exposure_status": "PASS" if exposed else "NOT_OBSERVED",
        "calls": calls,
        "ddr_validation": outputs,
        "scheduler_evidence": {
            key: task.get(key) for key in ("id", "state", "started", "finished")
        },
        "interpretation": "seed/control runs outside timing; unchanged target entered from cached control; require cached_stale>0 for a physical stale-cache conclusion",
    }
    write_json(args.output, result)
    return 0 if exposed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return validate(args)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        write_json(args.output, {"schema": 1, "status": "FAIL", "reason": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare observed final logical-node checkpoints to an explicit CPU trace.

Only emitted checkpoints are checked. A matching sparse trace is not a
full-output comparison or evidence of completed model execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.reference_trace.read_text())
    references = {
        item["final_physical_call"]: item
        for item in trace
        if item.get("final_physical_call") is not None
    }
    # A live UART chunk may end halfway through a hash. Such a partial value
    # must not become a transient false mismatch after zero-padding.
    samples = re.findall(
        r"\[efficientnet-checkpoint\] call=(\d+) span=(\d+)"
        r"(?: bytes=(\d+))? fnv64=0x([0-9a-fA-F]{16})(?![0-9a-fA-F])",
        args.uart.read_text(),
    )
    observations = {}
    skipped_intermediate = 0
    for call, span, size, value in samples:
        call = int(call)
        if call not in references:
            skipped_intermediate += 1
            continue
        key = (call, int(span))
        observations.setdefault(key, []).append((value.lower().zfill(16), size))
    rows = []
    for (call, span), values in sorted(observations.items()):
        reference = references[call]
        actual_hashes = sorted({value for value, size in values})
        expected = reference["output_fnv64"].lower().zfill(16)
        expected_bytes = 4
        for dimension in reference["shape"]:
            expected_bytes *= dimension
        sizes = sorted({int(size) for value, size in values if size})
        rows.append(
            {
                "call": call,
                "logical_node": reference["node"],
                "span": span,
                "expected_fnv64": expected,
                "actual_fnv64": actual_hashes,
                "observations_consistent": len(actual_hashes) == 1,
                "recorded_bytes": sizes,
                "expected_bytes": expected_bytes,
                "matches": actual_hashes == [expected]
                and (not sizes or sizes == [expected_bytes]),
            }
        )
    mismatches = [row for row in rows if not row["matches"]]
    report = {
        "scope": "observed sparse final logical-node hashes only; not full-model/output acceptance",
        "status": "OBSERVED_MISMATCH"
        if mismatches
        else ("OBSERVED_MATCH" if rows else "NO_COMPLETE_OBSERVATIONS"),
        "uart_sha256": sha(args.uart),
        "reference_trace_sha256": sha(args.reference_trace),
        "reference_logical_outputs": len(references),
        "observed_logical_outputs": len(rows),
        "matching": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "skipped_intermediate_observations": skipped_intermediate,
        "observations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "observations"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

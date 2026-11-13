#!/usr/bin/env python3
"""Compare selected sparse physical output hashes to an independent CPU oracle.

Does not interpret these observations as final logical outputs or model PASS.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    references = {
        (item["call"], item["span"]): item for item in reference["output_spans"]
    }
    if len(references) != len(reference["output_spans"]):
        raise ValueError("duplicate physical reference spans")
    samples = re.findall(
        r"\[efficientnet-checkpoint\] call=(\d+) span=(\d+)"
        r"(?: bytes=(\d+))? fnv64=0x([0-9a-fA-F]{16})(?![0-9a-fA-F])",
        args.uart.read_text(),
    )
    observations = {}
    skipped = 0
    for call, span, size, value in samples:
        key = (int(call), int(span))
        if key not in references:
            skipped += 1
            continue
        observations.setdefault(key, []).append(
            (value.lower(), int(size) if size else None)
        )
    rows = []
    for key, values in sorted(observations.items()):
        ref = references[key]
        actual = sorted({value for value, _ in values})
        sizes = sorted({size for _, size in values if size is not None})
        rows.append(
            {
                "call": key[0],
                "span": key[1],
                "logical_node": ref["logical_node"],
                "physical_value": ref["value"],
                "dtype": ref["dtype"],
                "shape": ref["shape"],
                "expected_bytes": ref["bytes"],
                "recorded_bytes": sizes,
                "expected_fnv64": ref["expected_fnv64"],
                "actual_fnv64": actual,
                "observations_consistent": len(actual) == 1,
                "matches": actual == [ref["expected_fnv64"]]
                and (not sizes or sizes == [ref["bytes"]]),
            }
        )
    mismatches = [row for row in rows if not row["matches"]]
    report = {
        "scope": reference["scope"],
        "status": "OBSERVED_MISMATCH"
        if mismatches
        else ("OBSERVED_MATCH" if rows else "NO_COMPLETE_OBSERVATIONS"),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "uart_sha256": hashlib.sha256(args.uart.read_bytes()).hexdigest(),
        "expected_output_spans": len(references),
        "observed_physical_outputs": len(rows),
        "matching": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "skipped_unreferenced_observations": skipped,
        "observed_physical_calls": sorted({row["call"] for row in rows}),
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

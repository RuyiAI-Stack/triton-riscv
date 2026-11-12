#!/usr/bin/env python3
"""Alternating controlled measurements of tile-entry candidates with changed calls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / "run_profile.py"


def sequence(pair, output, timeout):
    left = Path(pair["baseline_manifest"])
    right = Path(pair["candidate_manifest"])
    a = json.loads(left.read_text())
    b = json.loads(right.read_text())
    for key in ("case", "selected_tile_pairs", "selected_stage"):
        if a[key] != b[key]:
            raise ValueError("uncontrolled " + key)
    if (
        a["fixture"]["kind"] != b["fixture"]["kind"]
        or a["fixture"]["kind"] != "deterministic_nonzero_v1"
    ):
        raise ValueError("fixture")
    records = []
    for index, (label, manifest) in enumerate(
        [
            ("baseline", left),
            ("candidate", right),
            ("baseline", left),
            ("candidate", right),
        ],
        1,
    ):
        d = output / pair["case"] / ("run-" + str(index) + "-" + label)
        d.mkdir(parents=True, exist_ok=True)
        with (
            (d / "driver.stdout").open("w") as stdout,
            (d / "driver.stderr").open("w") as stderr,
        ):
            p = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(RUNNER),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(d),
                    "--workers",
                    "1",
                    "--timeout",
                    str(timeout),
                    "--ready-wait",
                    "3600",
                ],
                stdout=stdout,
                stderr=stderr,
            )
        if p.returncode:
            raise RuntimeError(str(d))
        rows = json.loads((d / "result-summary.json").read_text())["results"]
        if len(rows) != 1 or not rows[0]["profile_counter_check_passed"]:
            raise RuntimeError("invalid measurement " + str(d))
        row = rows[0]
        row["abab_variant"] = label
        row["abab_sequence"] = index
        records.append(row)
        (output / pair["case"] / "results.json").write_text(
            json.dumps(records, indent=2) + chr(10)
        )
        print(
            "PASS",
            pair["case"],
            index,
            label,
            row["profile"]["adapter_total_cycles"],
            flush=True,
        )
    metrics = {}
    for key in ("adapter_total_cycles", "summed_stage_cycles", "launch_cycles"):
        values = {
            label: [r["profile"][key] for r in records if r["abab_variant"] == label]
            for label in ("baseline", "candidate")
        }
        means = {k: sum(v) / len(v) for k, v in values.items()}
        metrics[key] = {
            "cycles": values,
            "means": means,
            "speedup": means["baseline"] / means["candidate"],
            "spread": {k: max(v) - min(v) for k, v in values.items()},
        }
    return {"case": pair["case"], "metrics": metrics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(a.pairs.read_text())
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(
            pool.map(lambda x: sequence(x, a.output.resolve(), a.timeout), pairs)
        )
    (a.output / "comparison.json").write_text(json.dumps(results, indent=2) + chr(10))


if __name__ == "__main__":
    main()

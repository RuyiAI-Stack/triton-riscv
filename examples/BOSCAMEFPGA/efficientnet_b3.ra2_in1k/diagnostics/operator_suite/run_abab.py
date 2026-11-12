#!/usr/bin/env python3
"""Run two frozen nonzero profile pairs as baseline,candidate,baseline,candidate.

Every sequence entry uses run_profile's foreground scheduler lease and unique
evidence directory. Existing phase evidence is read-only and never overwritten.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
PROFILE_RUNNER = HERE.parent / "conv_profile/run_profile.py"
spec = importlib.util.spec_from_file_location("profile_helpers", PROFILE_RUNNER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="JSON list of case,baseline_manifest,candidate_manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ready-wait", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())
    selected = []
    for pair in pairs:
        baseline = Path(pair["baseline_manifest"]).resolve()
        candidate = Path(pair["candidate_manifest"]).resolve()
        left, right = (
            json.loads(baseline.read_text()),
            json.loads(candidate.read_text()),
        )
        for key in (
            "case",
            "fixture",
            "selected_stage",
            "selected_tile_pairs",
            "expected_stage_calls",
        ):
            if left.get(key) != right.get(key):
                raise ValueError(
                    pair["case"] + ": unequal ABAB control metadata " + key
                )
        if (
            left["case"] != pair["case"]
            or left["mode"] != "fused"
            or right["mode"] != "fused"
        ):
            raise ValueError("ABAB requires matching fused case manifests")
        for label, path in [("baseline", baseline), ("candidate", candidate)]:
            manifest = json.loads(path.read_text())
            image = path.parent / (pair["case"] + ".bin")
            if (
                manifest.get("fixture", {}).get("kind") != "deterministic_nonzero_v1"
                or helpers.sha256(image) != manifest["bin_sha256"]
            ):
                raise ValueError("unfrozen or zero-fixture profile: " + str(path))
        selected.append((pair["case"], baseline, candidate))
    if args.dry_run:
        print(
            json.dumps(
                [
                    {"case": case, "baseline": str(left), "candidate": str(right)}
                    for case, left, right in selected
                ],
                indent=2,
            )
        )
        return 0
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case, baseline, candidate in selected:
        for index, (label, manifest) in enumerate(
            [
                ("baseline", baseline),
                ("candidate", candidate),
                ("baseline", baseline),
                ("candidate", candidate),
            ],
            1,
        ):
            output = args.output / case / ("run-" + str(index) + "-" + label)
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-B",
                str(PROFILE_RUNNER),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--workers",
                "1",
                "--timeout",
                str(args.timeout),
                "--ready-wait",
                str(args.ready_wait),
            ]
            print(f"ABAB {case} sequence={index} variant={label}", flush=True)
            with (
                (output / "driver.stdout").open("w") as stdout,
                (output / "driver.stderr").open("w") as stderr,
            ):
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
            if process.returncode:
                raise RuntimeError(f"ABAB sequence failed: {output}")
            result = json.loads((output / "result-summary.json").read_text())["results"]
            if len(result) != 1 or not result[0]["profile_counter_check_passed"]:
                raise RuntimeError(
                    "ABAB lacks one counter-checked result: " + str(output)
                )
            row = result[0]
            row["abab_variant"] = label
            row["abab_sequence"] = index
            records.append(row)
            helpers.write_json(
                args.output / "results.json",
                {"pairs": str(args.pairs.resolve()), "runs": records},
            )
    comparison = []
    for case, baseline, candidate in selected:
        rows = [row for row in records if row["case"] == case]
        stages = rows[0]["profile"]["stages"]
        metrics = {}
        for stage in stages:
            variants = {
                label: [
                    row["profile"]["stages"][stage]["cycles"]
                    for row in rows
                    if row["abab_variant"] == label
                ]
                for label in ("baseline", "candidate")
            }
            means = {
                label: sum(values) / len(values) for label, values in variants.items()
            }
            metrics[stage] = {
                "cycles": variants,
                "mean_cycles": means,
                "speedup": means["baseline"] / means["candidate"],
            }
        comparison.append({"case": case, "stage_comparison": metrics})
    helpers.write_json(args.output / "comparison.json", {"cases": comparison})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

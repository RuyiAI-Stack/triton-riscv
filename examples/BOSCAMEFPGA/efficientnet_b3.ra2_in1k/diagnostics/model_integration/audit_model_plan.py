#!/usr/bin/env python3
"""Independently validate physical call pointer disjointness and tensor liveness."""

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIZE = {"f32": 4, "i32": 4, "i8": 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan-dir", type=Path, default=ROOT / "build/model-integration/physical-plan"
    )
    a = p.parse_args()
    d = json.loads((a.plan_dir / "physical-plan.json").read_text())
    pm = json.loads((a.plan_dir / "parameters.json").read_text())
    ps = {e["name"]: e for e in pm["entries"]}
    ts = d["tensors"]
    w = d["workspace"]
    data = (a.plan_dir / "parameters.bin").read_bytes()
    if hashlib.sha256(data).hexdigest() != pm["sha256"]:
        raise ValueError("parameter blob sha mismatch")
    for name, s in ps.items():
        if s["offset"] % 64 or s["offset"] + s["bytes"] > len(data):
            raise ValueError("invalid parameter offset " + name)
        if (
            hashlib.sha256(data[s["offset"] : s["offset"] + s["bytes"]]).hexdigest()
            != s["sha256"]
        ):
            raise ValueError("parameter entry mismatch " + name)
    param_sorted = sorted(ps.items(), key=lambda e: e[1]["offset"])
    for (ln, left), (rn, r) in zip(param_sorted, param_sorted[1:]):
        if left["offset"] + left["bytes"] > r["offset"]:
            raise ValueError("parameter overlap " + ln + " " + rn)
    roots = {
        v["storage"]["root"]: v
        for v in ts.values()
        if v["storage"]["kind"] == "workspace"
    }
    live_pair_checks = 0
    values = list(roots.items())
    for i, (an, a0) in enumerate(values):
        aa = a0["storage"]
        assert aa["offset"] % 64 == 0
        if aa["offset"] + aa["reserved_bytes"] > w["activation_high_water_bytes"]:
            raise ValueError("activation exceeds bounds")
        for bn, b0 in values[i + 1 :]:
            bb = b0["storage"]
            # Intervals include both producer and final consumer calls. Every
            # call allocates outputs before releasing its final-use inputs.
            live = max(aa["born"], bb["born"]) <= min(aa["last_use"], bb["last_use"])
            overlap = max(aa["offset"], bb["offset"]) < min(
                aa["offset"] + aa["reserved_bytes"], bb["offset"] + bb["reserved_bytes"]
            )
            if live and overlap:
                raise ValueError("live workspace alias " + an + " " + bn)
            live_pair_checks += 1

    def span(arg):
        v = arg["value"]
        if arg["source"] == "parameter":
            s = ps[v]
            return "params", s["offset"], s["bytes"]
        s = ts[v]
        st = s["storage"]
        return (
            ("input", 0, s["bytes"])
            if st["kind"] == "external_input"
            else ("workspace", st["offset"], s["bytes"])
        )

    pair_checks = 0
    for call in d["calls"]:
        if call["index"] != d["calls"].index(call):
            raise ValueError("call ordering")
        seen = []
        for arg in call["arguments"]:
            storage, off, n = span(arg)
            if math.prod(arg["shape"]) * SIZE[arg["dtype"]] != n:
                raise ValueError("physical byte count " + call["target"])
            if off % 64:
                raise ValueError("unaligned call pointer " + call["target"])
            for prev, pstore, poff, pn in seen:
                pair_checks += 1
                if pstore == storage and max(off, poff) < min(off + n, poff + pn):
                    raise ValueError(
                        "ABI pointers alias: "
                        + call["target"]
                        + " "
                        + prev
                        + " "
                        + arg["value"]
                    )
            seen.append((arg["value"], storage, off, n))
    report = {
        "status": "PASS",
        "scope": "independent static physical-ABI and live-memory validation; not numerical or FPGA evidence",
        "physical_calls": len(d["calls"]),
        "call_argument_pairs_checked": pair_checks,
        "live_workspace_pairs_checked": live_pair_checks,
        "workspace_roots": len(roots),
        "parameter_entries_verified": len(ps),
        "workspace_bytes": w["provisional_total_bytes"],
        "plan_sha256": hashlib.sha256(
            (a.plan_dir / "physical-plan.json").read_bytes()
        ).hexdigest(),
    }
    (a.plan_dir / "independent-plan-audit.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

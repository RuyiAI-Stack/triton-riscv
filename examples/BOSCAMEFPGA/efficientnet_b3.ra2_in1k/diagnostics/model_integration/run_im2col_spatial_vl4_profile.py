#!/usr/bin/env python3
"""One isolated spatial-VL4 profile; keep explicit fpga112 client deadlines."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "conv_profile"))
import run_profile as profile  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["case"] != "dwconv2d_72x72x192_k3_s1":
        raise ValueError("this isolated runner permits only the first K3/S1 case")
    options = [
        "--lease-seconds",
        "300",
        "--heartbeat-seconds",
        "2",
        "--request-timeout",
        "180",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "server": "fpga112",
        "execution_seconds": 600,
        "client_lease_seconds": 300,
        "heartbeat_seconds": 2,
        "request_timeout_seconds": 180,
        "client_options": options,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": profile.sha256(args.manifest),
        "foreground_client_until_terminal": True,
    }
    (args.output / "scheduler-client-options.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )

    def with_options(base):
        return base if "--lease-seconds" in base else [*base, *options]

    original_invoke, original_run = profile.invoke, profile.run_one

    def invoke(base, *arguments, **kwargs):
        return original_invoke(with_options(base), *arguments, **kwargs)

    def run_one(item, arguments, base):
        return original_run(item, arguments, with_options(base))

    profile.invoke, profile.run_one = invoke, run_one
    sys.argv = [
        str(HERE / "run_im2col_spatial_vl4_profile.py"),
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output),
        "--timeout",
        "600",
        "--workers",
        "1",
    ]
    return profile.main()


if __name__ == "__main__":
    raise SystemExit(main())

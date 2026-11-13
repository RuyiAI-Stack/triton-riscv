#!/usr/bin/env python3
"""Read bounded scheduler UART chunks without changing the owning lease."""

import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, "/home/chh/gitprojects/fpga-scheduler/src")
from fpga_scheduler.client import Client

REPO = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18762")
    parser.add_argument(
        "--token-file", type=Path, default=REPO / ".agent/secrets/fpga112.token"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cursor = args.output / "uart-cursor.json"
    state = (
        json.loads(cursor.read_text())
        if cursor.exists()
        else {"task_id": args.task_id, "offset": 0}
    )
    if state["task_id"] != args.task_id:
        raise ValueError("output directory already belongs to a different task")
    with Client(args.endpoint, token_file=args.token_file) as client:
        chunk = client._request(
            "/jobs/"
            + args.task_id
            + "/output?offset="
            + str(state["offset"])
            + "&limit=65536"
        )
    data = base64.b64decode(chunk["data"])
    with (args.output / "live-uart.log").open("ab") as stream:
        stream.write(data)
    state.update(offset=chunk["next_offset"], done=chunk["done"])
    cursor.write_text(json.dumps(state, indent=2) + "\n")
    (args.output / "live-task.json").write_text(
        json.dumps(chunk["task"], indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "state": chunk["task"]["state"],
                "fpga": chunk["task"].get("fpga"),
                "offset": state["offset"],
                "done": state["done"],
                "latest_uart": data.decode("utf-8", errors="replace").splitlines()[
                    -12:
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

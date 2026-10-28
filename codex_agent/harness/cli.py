"""Command-line pilot for the DeepSeek Harness integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.runtime import HarnessAgent, HarnessUnavailableError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Triton-RISCV task written in natural language")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--session-id")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call the configured model; the default only prints the bounded prompt",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = HarnessSettings.from_env(args.repo_root)
    agent = HarnessAgent(settings)

    if not args.live:
        print(
            json.dumps(
                {
                    "mode": "prepared",
                    "provider": settings.provider,
                    "model": settings.model,
                    "session_root": str(settings.session_root),
                    "prompt": agent.prepare(args.task),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        try:
            result = agent.run(args.task, session_id=args.session_id)
        except (HarnessUnavailableError, ValueError) as error:
            print(json.dumps({"mode": "failed", "error": str(error)}, ensure_ascii=False))
            return 2
    finally:
        agent.close()

    print(
        json.dumps(
            {
                "mode": "completed",
                "session_id": result.session_id,
                "finish_reason": result.finish_reason,
                "event_count": len(result.events),
                "final_response": result.final_response,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

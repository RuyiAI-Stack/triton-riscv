"""No-model demonstration of the Triton-RISCV MCP tool boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters


async def run_mcp_demo(
    repo_root: Path,
    operator_name: str,
    *,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Launch the stdio server, list tools, and call discover_operator."""

    resolved_root = repo_root.resolve()
    child_env = os.environ.copy()
    child_env["TRITON_RISCV_REPO_ROOT"] = str(resolved_root)
    parameters = StdioServerParameters(
        command=python_executable,
        args=["-m", "codex_agent.harness.mcp_server"],
        cwd=resolved_root,
        env=child_env,
    )

    async with Client(parameters) as client:
        listed = await client.list_tools()
        result = await client.call_tool(
            "discover_operator",
            {"operator_name": operator_name},
        )

    if result.is_error:
        raise RuntimeError("discover_operator returned an MCP tool error")

    return {
        "transport": "stdio",
        "server": "triton-riscv-tools",
        "available_tools": [item.name for item in listed.tools],
        "tool_call": {
            "name": "discover_operator",
            "arguments": {"operator_name": operator_name},
        },
        "result": result.structured_content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the read-only Triton-RISCV MCP tool without a model API."
    )
    parser.add_argument("operator", help="Exact existing FlagGems operator name.")
    parser.add_argument("--repo-root", default=".", help="Triton-RISCV repository.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = asyncio.run(
        run_mcp_demo(
            Path(args.repo_root),
            args.operator,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

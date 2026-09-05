"""MCP server exposing guarded Triton-RISCV repository tools."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from codex_agent.operator_lifecycle import (
    ApplyRepairResult,
    DiagnosisToolResult,
    RepairProposalResult,
    ValidationToolResult,
    apply_operator_repair,
    diagnose_failure_run,
    propose_operator_repair,
    validate_operator_target,
)
from codex_agent.operator_tools import (
    DiscoverOperatorResult,
    discover_operator_evidence,
)


server = MCPServer(
    name="triton-riscv-tools",
    title="Triton-RISCV Operator Tools",
    description="Guarded operator discovery, validation, diagnosis, and repair tools.",
)


def repository_root() -> Path:
    """Resolve the repository selected by the Harness host."""

    return Path(os.environ.get("TRITON_RISCV_REPO_ROOT", Path.cwd())).resolve()


@server.tool(
    name="discover_operator",
    title="Discover Triton-RISCV operator",
    description=(
        "Inspect one existing FlagGems operator and return its implementation file, "
        "mapped tests, validation command, Torch references, Triton operations, "
        "and static risk hints. This tool never executes tests or modifies files."
    ),
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
    structured_output=True,
)
def discover_operator_tool(operator_name: str) -> DiscoverOperatorResult:
    """Discover repository evidence for an exact operator name."""

    return discover_operator_evidence(repository_root(), operator_name)


@server.tool(
    name="validate_operator",
    title="Validate Triton-RISCV operator",
    description=(
        "Create a validation plan by default. Set execute=true only after user "
        "approval; the host must also enable TRITON_RISCV_ALLOW_VALIDATION=1."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def validate_operator_tool(
    operator_name: str,
    execute: bool = False,
    source_env: bool = True,
    timeout_seconds: int = 900,
) -> ValidationToolResult:
    return validate_operator_target(
        repository_root(),
        operator_name,
        execute=execute,
        source_env=source_env,
        timeout_seconds=timeout_seconds,
    )


@server.tool(
    name="diagnose_failure",
    title="Diagnose validation failure",
    description="Classify one stored validation receipt and recommend the next action.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
    structured_output=True,
)
def diagnose_failure_tool(run_id: str) -> DiagnosisToolResult:
    return diagnose_failure_run(repository_root(), run_id)


@server.tool(
    name="propose_repair",
    title="Propose operator source repair",
    description=(
        "Store a replacement for one failed operator implementation as a pending "
        "proposal. This tool does not modify source or tests."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def propose_repair_tool(
    run_id: str,
    replacement_source: str,
    rationale: str,
) -> RepairProposalResult:
    return propose_operator_repair(
        repository_root(),
        run_id,
        replacement_source,
        rationale,
    )


@server.tool(
    name="apply_repair",
    title="Apply approved operator repair",
    description=(
        "Apply a separately approved repair proposal. The host must enable "
        "TRITON_RISCV_ALLOW_REPAIR_APPLY=1; tests are integrity locked."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def apply_repair_tool(proposal_id: str) -> ApplyRepairResult:
    return apply_operator_repair(repository_root(), proposal_id)


def main() -> None:
    """Run the local MCP server over stdio for DeepSeek Harness."""

    server.run(transport="stdio")


if __name__ == "__main__":
    main()

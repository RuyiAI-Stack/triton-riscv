"""Typed, side-effect-free Triton-RISCV domain tools."""

from __future__ import annotations

from dataclasses import asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from codex_agent.discover_operators import (
    OPERATOR_ROOT,
    discover_operator,
    is_valid_operator_name,
)


class OperatorEvidence(BaseModel):
    """Repository evidence returned for one discovered operator."""

    name: str
    visibility: str
    implementation_file: str
    test_files: list[str]
    test_nodes: list[str]
    validation_command: str
    triton_kernels: list[str]
    public_functions: list[str]
    tl_ops: list[str]
    torch_references: list[str]
    parametrize: list[str]
    risk_hints: list[str]


class DiscoverOperatorResult(BaseModel):
    """Stable structured output for the discover_operator tool."""

    status: Literal["found", "not_found", "not_operator"]
    query: str
    message: str
    operator: OperatorEvidence | None = None
    suggestions: list[str] = Field(default_factory=list)


def _operator_names(repo_root: Path) -> list[str]:
    root = repo_root / OPERATOR_ROOT
    if not root.is_dir():
        return []
    return sorted(
        path.stem
        for path in root.glob("*.py")
        if path.name != "__init__.py"
        and not path.name.startswith("test_")
        and is_valid_operator_name(path.stem)
    )


def _suggestions(repo_root: Path, query: str) -> list[str]:
    names = _operator_names(repo_root)
    contains = [name for name in names if query.lower() in name.lower()]
    if contains:
        return contains[:5]
    return get_close_matches(query, names, n=5, cutoff=0.4)


def discover_operator_evidence(
    repo_root: Path,
    operator_name: str,
) -> DiscoverOperatorResult:
    """Return implementation and test evidence without executing repository code."""

    query = operator_name.strip()
    if not query:
        raise ValueError("operator_name cannot be empty")
    if not is_valid_operator_name(query):
        raise ValueError(
            "operator_name must be a Python identifier containing only letters, "
            "numbers, and underscores"
        )

    resolved_root = repo_root.resolve()
    implementation = resolved_root / OPERATOR_ROOT / f"{query}.py"
    if not implementation.is_file():
        return DiscoverOperatorResult(
            status="not_found",
            query=query,
            message=f"No operator implementation file was found for {query}.",
            suggestions=_suggestions(resolved_root, query),
        )

    target = discover_operator(implementation, resolved_root)
    if target is None:
        return DiscoverOperatorResult(
            status="not_operator",
            query=query,
            message=(
                f"{implementation.relative_to(resolved_root)} exists but contains "
                "no discoverable Triton JIT kernel."
            ),
        )

    return DiscoverOperatorResult(
        status="found",
        query=query,
        message=f"Discovered repository evidence for {query}.",
        operator=OperatorEvidence.model_validate(asdict(target)),
    )

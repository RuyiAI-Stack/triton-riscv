"""Guarded operator validation, diagnosis, and repair lifecycle tools."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from codex_agent.operator_tools import discover_operator_evidence
from codex_agent.validate_operator import run_operator


ARTIFACT_ROOT = Path("agent-results/operator-lifecycle")
REPAIRABLE_STAGES = {"correctness", "pytest", "triton-frontend"}
MAX_REPLACEMENT_BYTES = 1_000_000


class ValidationToolResult(BaseModel):
    """Structured validation plan or execution receipt."""

    run_id: str
    operator: str
    status: str
    command: str | None = None
    exit_code: int | None = None
    failure_stage: str | None = None
    likely_reason: str | None = None
    error_excerpt: list[str] = Field(default_factory=list)
    implementation_file: str | None = None
    test_files: list[str] = Field(default_factory=list)
    log_path: str | None = None
    receipt_path: str


class DiagnosisToolResult(BaseModel):
    """Actionable interpretation of one validation receipt."""

    run_id: str
    operator: str
    status: Literal["passed", "failed", "planned", "blocked", "unknown"]
    failure_stage: str | None = None
    likely_reason: str | None = None
    recommended_action: str
    source_repair_allowed: bool
    evidence: list[str] = Field(default_factory=list)


class RepairProposalResult(BaseModel):
    """A source replacement waiting for human approval."""

    proposal_id: str
    run_id: str
    operator: str
    status: str
    implementation_file: str
    rationale: str
    diff_preview: str
    proposal_path: str


class ApplyRepairResult(BaseModel):
    """Result of attempting to apply an approved proposal."""

    proposal_id: str
    operator: str
    status: str
    implementation_file: str
    message: str
    patch_path: str | None = None


def _safe_id(value: str, label: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
        raise ValueError(f"{label} contains unsupported characters")
    return value


def _artifact_dir(repo_root: Path, name: str) -> Path:
    path = repo_root.resolve() / ARTIFACT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _relative_file(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("repository file escaped the workspace") from error
    return candidate


def _validate_replacement_source(source: str) -> None:
    if len(source.encode("utf-8")) > MAX_REPLACEMENT_BYTES:
        raise ValueError("replacement source exceeds the 1 MB limit")
    ast.parse(source)
    if "@triton.jit" not in source:
        raise ValueError("replacement must retain at least one @triton.jit kernel")


def _load_receipt(repo_root: Path, run_id: str) -> dict[str, Any]:
    safe_run_id = _safe_id(run_id, "run_id")
    return _read_json(_artifact_dir(repo_root, "receipts") / f"{safe_run_id}.json")


def _load_proposal(repo_root: Path, proposal_id: str) -> tuple[Path, dict[str, Any]]:
    safe_proposal_id = _safe_id(proposal_id, "proposal_id")
    path = _artifact_dir(repo_root, "proposals") / f"{safe_proposal_id}.json"
    return path, _read_json(path)


def validate_operator_target(
    repo_root: Path,
    operator_name: str,
    *,
    execute: bool = False,
    source_env: bool = True,
    timeout_seconds: int = 900,
) -> ValidationToolResult:
    """Plan validation by default; execute only when the host explicitly enables it."""

    root = repo_root.resolve()
    evidence = discover_operator_evidence(root, operator_name)
    if evidence.status != "found" or evidence.operator is None:
        raise ValueError(evidence.message)
    if timeout_seconds < 1 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be between 1 and 3600")
    if execute and os.environ.get("TRITON_RISCV_ALLOW_VALIDATION") != "1":
        raise PermissionError(
            "live validation is disabled; the host must set "
            "TRITON_RISCV_ALLOW_VALIDATION=1"
        )

    operator = evidence.operator.model_dump()
    result = run_operator(
        operator,
        repo_root=root,
        results_dir=_artifact_dir(root, "validation"),
        source_env=source_env,
        dry_run=not execute,
        timeout_seconds=timeout_seconds,
    )
    run_id = _new_id("run")
    receipt_path = _artifact_dir(root, "receipts") / f"{run_id}.json"
    payload = {
        **asdict(result),
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(receipt_path, payload)
    return ValidationToolResult(
        run_id=run_id,
        operator=result.operator,
        status=result.status,
        command=result.command,
        exit_code=result.exit_code,
        failure_stage=result.failure_stage,
        likely_reason=result.likely_reason,
        error_excerpt=result.error_excerpt,
        implementation_file=result.implementation_file,
        test_files=result.test_files,
        log_path=result.log_path,
        receipt_path=receipt_path.relative_to(root).as_posix(),
    )


def diagnose_failure_run(repo_root: Path, run_id: str) -> DiagnosisToolResult:
    """Diagnose a stored validation result without rerunning the operator."""

    receipt = _load_receipt(repo_root, run_id)
    status = receipt.get("status", "unknown")
    stage = receipt.get("failure_stage")
    reason = receipt.get("likely_reason")
    has_tests = bool(receipt.get("test_files"))
    repairable = status == "failed" and stage in REPAIRABLE_STAGES and has_tests

    if status == "passed":
        action = "No repair is needed; preserve the passing receipt as evidence."
    elif status == "planned":
        action = "Request approval and run live validation before diagnosing a failure."
    elif status == "failed" and stage in REPAIRABLE_STAGES and not has_tests:
        action = "Define and review an acceptance test before allowing source repair."
    elif repairable:
        action = "Inspect the implementation and prepare a bounded source repair proposal."
    elif stage in {"environment", "import", "build", "timeout"}:
        action = "Repair the environment or retry policy; do not change operator source."
    elif stage in {"triton-shared-opt", "buddy-opt", "mlir-translate", "llc", "compilation"}:
        action = "Report a compiler-pipeline limitation before considering a source workaround."
    else:
        action = "Collect a fuller log and request human triage before editing source."

    normalized_status = status if status in {"passed", "failed", "planned", "blocked"} else "unknown"
    return DiagnosisToolResult(
        run_id=receipt["run_id"],
        operator=receipt["operator"],
        status=normalized_status,
        failure_stage=stage,
        likely_reason=reason,
        recommended_action=action,
        source_repair_allowed=repairable,
        evidence=list(receipt.get("error_excerpt", [])),
    )


def propose_operator_repair(
    repo_root: Path,
    run_id: str,
    replacement_source: str,
    rationale: str,
) -> RepairProposalResult:
    """Store a source-only repair proposal without changing repository files."""

    root = repo_root.resolve()
    receipt = _load_receipt(root, run_id)
    diagnosis = diagnose_failure_run(root, run_id)
    if not diagnosis.source_repair_allowed:
        raise ValueError(
            f"source repair is not allowed for stage {diagnosis.failure_stage!r}"
        )
    if not rationale.strip():
        raise ValueError("rationale cannot be empty")
    _validate_replacement_source(replacement_source)

    relative_implementation = receipt["implementation_file"]
    implementation = _relative_file(root, relative_implementation)
    allowed_root = (root / "python/examples/flaggems").resolve()
    if implementation.parent != allowed_root or implementation.name.startswith("test_"):
        raise ValueError("repair target must be a FlagGems implementation file")
    current_source = implementation.read_text(encoding="utf-8")
    if current_source == replacement_source:
        raise ValueError("replacement source does not change the implementation")

    test_hashes = {
        relative: _sha256(_relative_file(root, relative))
        for relative in receipt.get("test_files", [])
    }
    diff = "".join(
        unified_diff(
            current_source.splitlines(keepends=True),
            replacement_source.splitlines(keepends=True),
            fromfile=f"a/{relative_implementation}",
            tofile=f"b/{relative_implementation}",
        )
    )
    proposal_id = _new_id("repair")
    proposal_path = _artifact_dir(root, "proposals") / f"{proposal_id}.json"
    payload = {
        "proposal_id": proposal_id,
        "run_id": run_id,
        "operator": receipt["operator"],
        "status": "pending_approval",
        "implementation_file": relative_implementation,
        "source_sha256": _sha256(implementation),
        "test_sha256": test_hashes,
        "replacement_source": replacement_source,
        "rationale": rationale.strip(),
        "diff": diff,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(proposal_path, payload)
    return RepairProposalResult(
        proposal_id=proposal_id,
        run_id=run_id,
        operator=receipt["operator"],
        status="pending_approval",
        implementation_file=relative_implementation,
        rationale=rationale.strip(),
        diff_preview=diff[:20_000],
        proposal_path=proposal_path.relative_to(root).as_posix(),
    )


def get_repair_proposal(repo_root: Path, proposal_id: str) -> dict[str, Any]:
    """Return a proposal without exposing its full replacement source."""

    _, proposal = _load_proposal(repo_root, proposal_id)
    return {
        key: value
        for key, value in proposal.items()
        if key not in {"replacement_source", "source_sha256", "test_sha256"}
    }


def decide_repair_proposal(
    repo_root: Path,
    proposal_id: str,
    *,
    approve: bool,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    """Record a host-side human decision; this function is not exposed as an MCP tool."""

    if not reviewer.strip():
        raise ValueError("reviewer cannot be empty")
    path, proposal = _load_proposal(repo_root, proposal_id)
    if proposal["status"] != "pending_approval":
        raise ValueError(f"proposal is already {proposal['status']}")
    proposal["status"] = "approved" if approve else "rejected"
    proposal["review"] = {
        "reviewer": reviewer.strip(),
        "note": note.strip(),
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(path, proposal)
    return get_repair_proposal(repo_root, proposal_id)


def apply_operator_repair(repo_root: Path, proposal_id: str) -> ApplyRepairResult:
    """Apply an approved proposal after source and acceptance-test integrity checks."""

    root = repo_root.resolve()
    _, proposal = _load_proposal(root, proposal_id)
    relative_implementation = proposal["implementation_file"]
    implementation = _relative_file(root, relative_implementation)

    if proposal["status"] != "approved":
        return ApplyRepairResult(
            proposal_id=proposal_id,
            operator=proposal["operator"],
            status="not_approved",
            implementation_file=relative_implementation,
            message="The proposal must be approved by the host before it can be applied.",
        )
    if os.environ.get("TRITON_RISCV_ALLOW_REPAIR_APPLY") != "1":
        return ApplyRepairResult(
            proposal_id=proposal_id,
            operator=proposal["operator"],
            status="blocked",
            implementation_file=relative_implementation,
            message="The host has not enabled TRITON_RISCV_ALLOW_REPAIR_APPLY=1.",
        )
    if _sha256(implementation) != proposal["source_sha256"]:
        raise RuntimeError("implementation changed after the proposal was created")
    for relative, expected_hash in proposal["test_sha256"].items():
        if _sha256(_relative_file(root, relative)) != expected_hash:
            raise RuntimeError("an acceptance test changed after the proposal was created")

    replacement = proposal["replacement_source"]
    _validate_replacement_source(replacement)
    original = implementation.read_text(encoding="utf-8")
    patch_path = _artifact_dir(root, "patches") / f"{proposal_id}.diff"
    patch_path.write_text(proposal["diff"], encoding="utf-8")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=implementation.parent,
            prefix=f".{implementation.name}.",
            delete=False,
        ) as temporary:
            temporary.write(replacement)
            temporary_name = temporary.name
        Path(temporary_name).replace(implementation)
    except Exception:
        implementation.write_text(original, encoding="utf-8")
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise

    proposal_path, updated = _load_proposal(root, proposal_id)
    updated["status"] = "applied"
    updated["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    updated["applied_source_sha256"] = _sha256(implementation)
    _write_json(proposal_path, updated)
    return ApplyRepairResult(
        proposal_id=proposal_id,
        operator=proposal["operator"],
        status="applied",
        implementation_file=relative_implementation,
        message="Approved source replacement was applied; validation must run next.",
        patch_path=patch_path.relative_to(root).as_posix(),
    )

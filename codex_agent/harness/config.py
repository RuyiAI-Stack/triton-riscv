"""Environment-backed configuration for the DeepSeek Harness control plane."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional


def _optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class HarnessSettings:
    """Configuration that is safe to pass to the official Harness SDK."""

    repo_root: Path
    session_root: Path
    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    base_url: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    remote_host: Optional[str] = None
    remote_root: Optional[str] = None
    request_timeout_seconds: float = 900.0

    @classmethod
    def from_env(
        cls,
        repo_root: Path,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "HarnessSettings":
        env = os.environ if environ is None else environ
        resolved_root = repo_root.resolve()
        session_root = Path(
            env.get(
                "DSH_SESSION_ROOT",
                resolved_root / "agent-results" / "deepseek-harness",
            )
        ).expanduser()
        return cls(
            repo_root=resolved_root,
            session_root=session_root.resolve(),
            provider=env.get("DSH_PROVIDER", "deepseek-official").strip(),
            model=env.get("DSH_MODEL", "deepseek-v4-flash").strip(),
            base_url=_optional_text(env.get("DEEPSEEK_BASE_URL")),
            api_key=_optional_text(env.get("DEEPSEEK_API_KEY")),
            remote_host=_optional_text(env.get("RISCV_HOST")),
            remote_root=_optional_text(env.get("RISCV_REPO")),
            request_timeout_seconds=float(
                env.get("DSH_REQUEST_TIMEOUT_SECONDS", "900")
            ),
        )

    def validate_live_run(self) -> None:
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for a live Harness run")
        if not self.repo_root.is_dir():
            raise ValueError(f"repository does not exist: {self.repo_root}")

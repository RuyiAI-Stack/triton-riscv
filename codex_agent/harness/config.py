"""Environment-backed configuration for the DeepSeek Harness control plane."""

from __future__ import annotations

import os
import shutil
import sys
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
    cordis_path: Path
    mcp_python: str
    provider: str = "isrc-proxy"
    model: str = "gpt-5.6-sol"
    base_url: Optional[str] = "https://llmapi.isrc.ac.cn/v1"
    api_key_env: str = "ISRC_API_KEY"
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
        cordis_path = Path(
            env.get(
                "TRITON_RISCV_CORDIS_CONFIG",
                Path(__file__).with_name("cordis.yml"),
            )
        ).expanduser()
        if not cordis_path.is_absolute():
            cordis_path = resolved_root / cordis_path
        api_key_env = env.get("DSH_API_KEY_ENV", "ISRC_API_KEY").strip()
        if not api_key_env:
            raise ValueError("DSH_API_KEY_ENV cannot be empty")
        return cls(
            repo_root=resolved_root,
            session_root=session_root.resolve(),
            cordis_path=cordis_path.resolve(),
            mcp_python=env.get("TRITON_RISCV_MCP_PYTHON", sys.executable),
            provider=env.get("DSH_PROVIDER", "isrc-proxy").strip(),
            model=env.get("DSH_MODEL", "gpt-5.6-sol").strip(),
            base_url=_optional_text(
                env.get("ISRC_BASE_URL", "https://llmapi.isrc.ac.cn/v1")
            ),
            api_key_env=api_key_env,
            api_key=_optional_text(env.get(api_key_env)),
            remote_host=_optional_text(env.get("RISCV_HOST")),
            remote_root=_optional_text(env.get("RISCV_REPO")),
            request_timeout_seconds=float(
                env.get("DSH_REQUEST_TIMEOUT_SECONDS", "900")
            ),
        )

    def validate_runtime_config(self) -> None:
        if not self.repo_root.is_dir():
            raise ValueError(f"repository does not exist: {self.repo_root}")
        if not self.cordis_path.is_file():
            raise ValueError(f"Cordis config does not exist: {self.cordis_path}")
        if not shutil.which(self.mcp_python):
            raise ValueError(f"MCP Python executable was not found: {self.mcp_python}")
        if not self.provider:
            raise ValueError("DSH_PROVIDER cannot be empty")
        if not self.model:
            raise ValueError("DSH_MODEL cannot be empty")

    def validate_live_run(self) -> None:
        self.validate_runtime_config()
        if not self.api_key:
            raise ValueError(
                f"{self.api_key_env} is required for a live Harness run"
            )

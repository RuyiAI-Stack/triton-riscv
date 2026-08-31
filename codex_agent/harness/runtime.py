"""Small boundary around the official DeepSeek Harness Python SDK."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.context import build_task_prompt


class HarnessUnavailableError(RuntimeError):
    """Raised when the optional Harness runtime cannot be started."""


@dataclass(frozen=True)
class HarnessRunResult:
    session_id: str
    final_response: str
    finish_reason: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)


HarnessEventCallback = Callable[[dict[str, Any]], None]


class HarnessBackend(Protocol):
    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[HarnessEventCallback] = None,
    ) -> HarnessRunResult:
        ...

    def close(self) -> None:
        ...


class DeepSeekHarnessBackend:
    """Lazy adapter so normal agent tests do not require the optional SDK."""

    def __init__(self, settings: HarnessSettings) -> None:
        self.settings = settings
        self._harness: Any = None
        self._lock = threading.RLock()

    def _runtime(self) -> Any:
        if self._harness is not None:
            return self._harness

        try:
            from deepseek_harness import DeepSeekHarness
        except ImportError as error:
            raise HarnessUnavailableError(
                "DeepSeek Harness SDK is not installed. Use Python 3.10+ and "
                "install codex_agent/harness/requirements.txt in a separate "
                "virtual environment."
            ) from error

        self._harness = DeepSeekHarness(
            provider=self.settings.provider,
            model=self.settings.model,
            cwd=str(self.settings.repo_root),
            session_root=str(self.settings.session_root),
            cordis=str(self.settings.cordis_path),
            env={
                "TRITON_RISCV_MCP_PYTHON": self.settings.mcp_python,
                "TRITON_RISCV_ALLOW_VALIDATION": os.environ.get(
                    "TRITON_RISCV_ALLOW_VALIDATION", "0"
                ),
                "TRITON_RISCV_ALLOW_REPAIR_APPLY": os.environ.get(
                    "TRITON_RISCV_ALLOW_REPAIR_APPLY", "0"
                ),
            },
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            request_timeout_seconds=self.settings.request_timeout_seconds,
        )
        return self._harness

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[HarnessEventCallback] = None,
    ) -> HarnessRunResult:
        self.settings.validate_live_run()
        self.settings.session_root.mkdir(parents=True, exist_ok=True)

        try:
            def forward_notification(notification: Any) -> None:
                if on_event is not None:
                    on_event(
                        {
                            "method": notification.method,
                            "payload": dict(notification.payload),
                        }
                    )

            # The official client owns one reusable subprocess. Serializing access
            # preserves session shell state and avoids sharing its stdio transport
            # unsafely across worker threads.
            with self._lock:
                result = self._runtime().run(
                    prompt,
                    session_id=session_id,
                    on_notification=forward_notification,
                )
        except Exception as error:
            raise HarnessUnavailableError(f"DeepSeek Harness run failed: {error}") from error

        return HarnessRunResult(
            session_id=result.session_id,
            final_response=result.final_response,
            finish_reason=result.finish_reason,
            events=list(result.events),
        )

    def close(self) -> None:
        with self._lock:
            if self._harness is not None:
                self._harness.close()
                self._harness = None


class HarnessAgent:
    """Domain-facing application service independent of a concrete backend."""

    def __init__(
        self,
        settings: HarnessSettings,
        backend: Optional[HarnessBackend] = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or DeepSeekHarnessBackend(settings)

    def prepare(self, user_request: str) -> str:
        return build_task_prompt(user_request, self.settings)

    def run(
        self,
        user_request: str,
        session_id: Optional[str] = None,
        on_event: Optional[HarnessEventCallback] = None,
    ) -> HarnessRunResult:
        return self.backend.run(
            self.prepare(user_request),
            session_id=session_id,
            on_event=on_event,
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

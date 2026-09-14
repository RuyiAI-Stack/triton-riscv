from __future__ import annotations

import tempfile
import unittest
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import patch

from codex_agent.harness import (
    DeepSeekHarnessBackend,
    HarnessAgent,
    HarnessRunResult,
    HarnessSettings,
)


class FakeHarnessBackend:
    def __init__(self) -> None:
        self.prompt: Optional[str] = None
        self.session_id: Optional[str] = None

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> HarnessRunResult:
        self.prompt = prompt
        self.session_id = session_id
        if on_event is not None:
            on_event({"method": "session.event", "payload": {"type": "step/start"}})
        return HarnessRunResult(
            session_id=session_id or "generated-session",
            final_response="planned operator validation",
            finish_reason="completed",
            events=[{"type": "assistant/message"}],
        )


class HarnessIntegrationTests(unittest.TestCase):
    def test_settings_are_loaded_without_exposing_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = HarnessSettings.from_env(
                root,
                {
                    "DEEPSEEK_API_KEY": "secret-value",
                    "DEEPSEEK_BASE_URL": "https://model.example/v1",
                    "DSH_MODEL": "company-deepseek",
                    "RISCV_HOST": "sg2044",
                    "RISCV_REPO": "/home/user/work/triton-riscv",
                },
            )

        self.assertEqual(settings.model, "company-deepseek")
        self.assertEqual(settings.remote_host, "sg2044")
        self.assertNotIn("secret-value", repr(settings))

    def test_official_runtime_is_reused_and_closed(self) -> None:
        instances: list[Any] = []

        class FakeOfficialHarness:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.closed = False
                self.calls: list[str] = []
                instances.append(self)

            def run(self, prompt: str, *, session_id: str, on_notification: Any) -> Any:
                self.calls.append(prompt)
                on_notification(
                    SimpleNamespace(method="session.event", payload={"sessionId": session_id})
                )
                return SimpleNamespace(
                    session_id=session_id,
                    final_response="ok",
                    finish_reason="completed",
                    events=[],
                )

            def close(self) -> None:
                self.closed = True

        module = types.ModuleType("deepseek_harness")
        module.DeepSeekHarness = FakeOfficialHarness
        with tempfile.TemporaryDirectory() as temporary:
            settings = HarnessSettings.from_env(
                Path(temporary),
                {"DEEPSEEK_API_KEY": "test-key"},
            )
            backend = DeepSeekHarnessBackend(settings)
            events: list[dict[str, Any]] = []
            with patch.dict(sys.modules, {"deepseek_harness": module}):
                backend.run("first", "session-1", events.append)
                backend.run("second", "session-1", events.append)
                backend.close()

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].calls, ["first", "second"])
        self.assertEqual(instances[0].kwargs["session_root"], str(settings.session_root))
        self.assertNotIn("profile", instances[0].kwargs)
        self.assertNotIn("dsh_home", instances[0].kwargs)
        self.assertTrue(instances[0].closed)
        self.assertEqual(events[0]["method"], "session.event")

    def test_prepare_builds_bounded_domain_context(self) -> None:
        settings = HarnessSettings.from_env(
            Path.cwd(),
            {
                "RISCV_HOST": "sg2044",
                "RISCV_REPO": "/home/user/work/triton-riscv",
            },
        )
        prompt = HarnessAgent(settings, FakeHarnessBackend()).prepare(
            "验证 relu_and_mul 算子"
        )

        self.assertIn("验证 relu_and_mul 算子", prompt)
        self.assertIn("sg2044:/home/user/work/triton-riscv", prompt)
        self.assertIn("python -m codex_agent.operator_agent", prompt)
        self.assertIn("Never weaken", prompt)

    def test_agent_delegates_to_backend_and_preserves_session(self) -> None:
        settings = HarnessSettings.from_env(Path.cwd(), {})
        backend = FakeHarnessBackend()
        result = HarnessAgent(settings, backend).run(
            "分析 gelu 的编译失败", session_id="session-7"
        )

        self.assertEqual(backend.session_id, "session-7")
        self.assertIn("分析 gelu 的编译失败", backend.prompt or "")
        self.assertEqual(result.session_id, "session-7")
        self.assertEqual(result.events[0]["type"], "assistant/message")

    def test_empty_request_is_rejected_before_model_call(self) -> None:
        settings = HarnessSettings.from_env(Path.cwd(), {})
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            HarnessAgent(settings, FakeHarnessBackend()).prepare("  ")


if __name__ == "__main__":
    unittest.main()

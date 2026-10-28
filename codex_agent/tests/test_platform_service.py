from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Optional

from codex_agent.harness import HarnessAgent, HarnessRunResult, HarnessSettings
from codex_agent.platform.executor import HarnessRunExecutor
from codex_agent.platform.service import PlatformService
from codex_agent.platform.store import PlatformStore


class FakeHarnessBackend:
    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> HarnessRunResult:
        if on_event is not None:
            on_event(
                {
                    "method": "session.event",
                    "payload": {"event": {"type": "assistant/message"}},
                }
            )
        return HarnessRunResult(
            session_id=session_id or "generated-session",
            final_response="Harness inspected the requested operator.",
            finish_reason="completed",
            events=[{"type": "assistant/message"}],
        )


class PlatformServiceTests(unittest.TestCase):
    def make_components(
        self, root: Path
    ) -> tuple[PlatformStore, PlatformService, HarnessRunExecutor]:
        results = root / "agent-results"
        results.mkdir()
        (results / "operators.json").write_text(
            json.dumps({"summary": {"operators": 1}, "operators": []}),
            encoding="utf-8",
        )
        (results / "project-targets.json").write_text(
            json.dumps({"summary": {"total_targets": 12}}),
            encoding="utf-8",
        )
        settings = HarnessSettings.from_env(
            root,
            {
                "ISRC_API_KEY": "test-key",
                "DSH_MODEL": "test-model",
            },
        )
        agent = HarnessAgent(settings, FakeHarnessBackend())
        store = PlatformStore(results / "platform.sqlite3")
        service = PlatformService(root, store, agent)
        executor = HarnessRunExecutor(root, store, agent, workers=1)
        return store, service, executor

    def test_every_message_creates_a_harness_confirmation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            session = service.create_session()

            result = service.handle_message(session["id"], "验证 relu_and_mul")

            self.assertEqual(result["run"]["status"], "awaiting-confirmation")
            self.assertEqual(result["run"]["request"]["action"], "deepseek-harness")
            self.assertEqual(result["task"]["runtime"], "deepseek-harness")
            self.assertIn("test-model", result["assistant_message"]["content"])

    def test_confirm_runs_harness_and_persists_stream_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            session = service.create_session()
            planned = service.handle_message(session["id"], "分析 gelu 失败")

            executor.confirm(planned["run"]["id"])
            deadline = time.time() + 2
            while time.time() < deadline:
                run = store.get_run(planned["run"]["id"])
                if run["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)

            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["result"]["harness_session_id"], f"triton-ui-{session['id']}")
            event_types = [item["event_type"] for item in store.list_events(run["id"])]
            self.assertIn("harness-event", event_types)
            self.assertIn("completed", event_types)
            self.assertIn(
                "Harness inspected",
                store.list_messages(session["id"])[-1]["content"],
            )

    def test_executor_rejects_old_python_command_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a DeepSeek Harness task"):
            HarnessRunExecutor._validate_request(
                {"action": "validate-operator", "argv": ["-m", "os"]}
            )

    def test_bootstrap_reports_harness_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            bootstrap = service.bootstrap()

            self.assertEqual(bootstrap["harness"]["runtime"], "DeepSeek Harness")
            self.assertEqual(bootstrap["harness"]["model"], "test-model")
            self.assertTrue(bootstrap["harness"]["api_configured"])


if __name__ == "__main__":
    unittest.main()

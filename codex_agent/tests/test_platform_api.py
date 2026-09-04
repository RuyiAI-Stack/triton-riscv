from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi.testclient import TestClient

from codex_agent.harness import HarnessAgent, HarnessRunResult, HarnessSettings
from codex_agent.platform.api import create_app


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
                    "payload": {"event": {"type": "tool/result"}},
                }
            )
        return HarnessRunResult(
            session_id=session_id or "generated-session",
            final_response="HTTP bridge reached the Harness backend.",
            finish_reason="completed",
        )


class PlatformApiTests(unittest.TestCase):
    def test_message_confirmation_and_harness_result_cross_http_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = HarnessSettings.from_env(
                root,
                {"DEEPSEEK_API_KEY": "test-key", "DSH_MODEL": "test-model"},
            )
            agent = HarnessAgent(settings, FakeHarnessBackend())
            app = create_app(root, harness_agent=agent)

            with TestClient(app) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                session = client.post("/api/sessions", json={"title": "HTTP test"}).json()
                planned = client.post(
                    f"/api/sessions/{session['id']}/messages",
                    json={"content": "分析一个没有关键词模板的新任务"},
                ).json()
                self.assertEqual(planned["run"]["status"], "awaiting-confirmation")

                run_id = planned["run"]["id"]
                client.post(f"/api/runs/{run_id}/confirm").raise_for_status()
                deadline = time.time() + 2
                while time.time() < deadline:
                    run = client.get(f"/api/runs/{run_id}").json()
                    if run["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertEqual(run["status"], "completed")
                events = client.get(f"/api/runs/{run_id}/events").json()
                self.assertIn("harness-event", [event["event_type"] for event in events])
                bundle = client.get(f"/api/sessions/{session['id']}").json()
                self.assertIn("HTTP bridge reached", bundle["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()

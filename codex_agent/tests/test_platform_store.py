from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex_agent.platform.store import PlatformStore


class PlatformStoreTests(unittest.TestCase):
    def test_persists_session_messages_runs_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PlatformStore(Path(temp_dir) / "platform.sqlite3")
            session = store.create_session()
            message = store.add_message(
                session["id"], "user", "验证 relu", {"source": "test"}
            )
            run = store.create_run(
                session["id"],
                "validate-operator",
                "relu",
                {"action": "validate-operator", "argv": ["-m", "codex_agent.operator_agent"]},
            )
            store.add_event(run["id"], "output", {"line": "1 passed"})
            store.update_run(
                run["id"], status="completed", phase="completed", result={"exit_code": 0}
            )

            bundle = store.session_bundle(session["id"])

            self.assertEqual(bundle["messages"][0]["id"], message["id"])
            self.assertEqual(bundle["messages"][0]["metadata"]["source"], "test")
            self.assertEqual(bundle["runs"][0]["result"]["exit_code"], 0)
            events = store.list_events(run["id"])
            self.assertEqual([item["sequence"] for item in events], [0, 1])

    def test_rejects_unknown_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PlatformStore(Path(temp_dir) / "platform.sqlite3")
            with self.assertRaises(KeyError):
                store.get_session("missing")


if __name__ == "__main__":
    unittest.main()

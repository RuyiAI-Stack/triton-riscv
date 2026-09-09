from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex_agent.operator_agent import prune_old_logs, select_operators
from codex_agent.operator_status import build_status_report


class OperatorAgentTests(unittest.TestCase):
    def test_selects_only_unvalidated_public_operators(self) -> None:
        operators = [
            {"name": "_internal", "visibility": "internal"},
            {"name": "add", "visibility": "public"},
            {"name": "mul", "visibility": "public"},
        ]
        latest = {"add": {"status": "passed"}}

        selected = select_operators(
            operators,
            latest,
            names=[],
            visibility="public",
            selection="unvalidated",
            contains=None,
            offset=0,
            limit=20,
        )

        self.assertEqual([item["name"] for item in selected], ["mul"])

    def test_status_report_does_not_overclaim_correctness(self) -> None:
        inventory = {
            "operators": [
                {
                    "name": "add",
                    "visibility": "public",
                    "implementation_file": "add.py",
                    "test_files": ["test_add.py"],
                    "test_nodes": ["test_add.py::test_add"],
                    "validation_command": "pytest test_add.py",
                },
                {
                    "name": "mul",
                    "visibility": "public",
                    "implementation_file": "mul.py",
                    "test_files": ["test_mul.py"],
                    "test_nodes": ["test_mul.py::test_mul"],
                    "validation_command": "pytest test_mul.py",
                },
            ]
        }
        report = build_status_report(
            inventory,
            [{"operator": "add", "status": "passed"}],
        )

        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["public_validated"], 1)
        self.assertEqual(report["summary"]["unvalidated"], 1)
        self.assertEqual(report["operators"][0]["correctness"], "basic-tests-passed")
        self.assertEqual(report["operators"][1]["correctness"], "not-established")

    def test_log_retention_protects_latest_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            log_dir = results_dir / "logs"
            log_dir.mkdir()
            old_log = log_dir / "old.log"
            latest_log = log_dir / "latest.log"
            old_log.write_text("old", encoding="utf-8")
            latest_log.write_text("latest", encoding="utf-8")

            removed = prune_old_logs(
                results_dir,
                keep_logs=1,
                latest={"add": {"log_path": latest_log.as_posix()}},
            )

            self.assertEqual(removed, [old_log.resolve().as_posix()])
            self.assertFalse(old_log.exists())
            self.assertTrue(latest_log.exists())


if __name__ == "__main__":
    unittest.main()

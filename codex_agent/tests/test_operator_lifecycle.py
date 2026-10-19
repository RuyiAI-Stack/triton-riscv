from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.operator_lifecycle import (
    apply_operator_repair,
    decide_repair_proposal,
    diagnose_failure_run,
    propose_operator_repair,
    validate_operator_target,
)


ORIGINAL_SOURCE = """import triton
import triton.language as tl

@triton.jit
def demo_kernel(x):
    return x
"""

REPLACEMENT_SOURCE = """import triton
import triton.language as tl

@triton.jit
def demo_kernel(x):
    return x + 1
"""

TEST_SOURCE = """from .demo import demo_kernel

def test_demo():
    assert demo_kernel is not None
"""


class OperatorLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        operator_root = self.root / "python/examples/flaggems"
        operator_root.mkdir(parents=True)
        self.implementation = operator_root / "demo.py"
        self.test_file = operator_root / "test_demo.py"
        self.implementation.write_text(ORIGINAL_SOURCE, encoding="utf-8")
        self.test_file.write_text(TEST_SOURCE, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def failed_receipt(self) -> str:
        planned = validate_operator_target(self.root, "demo")
        receipt_path = self.root / planned.receipt_path
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update(
            {
                "status": "failed",
                "exit_code": 1,
                "failure_stage": "correctness",
                "likely_reason": "result differs from PyTorch reference",
                "error_excerpt": ["mismatched elements: 8 / 8"],
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return planned.run_id

    def test_validation_plans_by_default_and_live_run_needs_host_switch(self) -> None:
        result = validate_operator_target(self.root, "demo")

        self.assertEqual(result.status, "planned")
        self.assertIn("test_demo.py::test_demo", result.command or "")
        self.assertTrue((self.root / result.receipt_path).is_file())
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(PermissionError, "ALLOW_VALIDATION"):
                validate_operator_target(self.root, "demo", execute=True)

    def test_diagnosis_and_approved_repair_lifecycle(self) -> None:
        run_id = self.failed_receipt()
        diagnosis = diagnose_failure_run(self.root, run_id)

        self.assertTrue(diagnosis.source_repair_allowed)
        proposal = propose_operator_repair(
            self.root,
            run_id,
            REPLACEMENT_SOURCE,
            "Correct the deliberately wrong kernel expression.",
        )
        self.assertEqual(proposal.status, "pending_approval")
        self.assertEqual(self.implementation.read_text(encoding="utf-8"), ORIGINAL_SOURCE)
        self.assertEqual(apply_operator_repair(self.root, proposal.proposal_id).status, "not_approved")

        decide_repair_proposal(
            self.root,
            proposal.proposal_id,
            approve=True,
            reviewer="unit-test",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                apply_operator_repair(self.root, proposal.proposal_id).status,
                "blocked",
            )
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}):
            applied = apply_operator_repair(self.root, proposal.proposal_id)

        self.assertEqual(applied.status, "applied")
        self.assertEqual(
            self.implementation.read_text(encoding="utf-8"),
            REPLACEMENT_SOURCE,
        )
        self.assertEqual(self.test_file.read_text(encoding="utf-8"), TEST_SOURCE)
        self.assertTrue((self.root / (applied.patch_path or "")).is_file())

    def test_changed_acceptance_test_blocks_an_approved_repair(self) -> None:
        run_id = self.failed_receipt()
        proposal = propose_operator_repair(
            self.root,
            run_id,
            REPLACEMENT_SOURCE,
            "Repair implementation only.",
        )
        decide_repair_proposal(
            self.root,
            proposal.proposal_id,
            approve=True,
            reviewer="unit-test",
        )
        self.test_file.write_text(TEST_SOURCE + "\n# changed\n", encoding="utf-8")

        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}):
            with self.assertRaisesRegex(RuntimeError, "acceptance test changed"):
                apply_operator_repair(self.root, proposal.proposal_id)

        self.assertEqual(self.implementation.read_text(encoding="utf-8"), ORIGINAL_SOURCE)

    def test_compiler_failure_is_not_silently_rewritten(self) -> None:
        run_id = self.failed_receipt()
        receipt_path = (
            self.root / "agent-results/operator-lifecycle/receipts" / f"{run_id}.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["failure_stage"] = "buddy-opt"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        diagnosis = diagnose_failure_run(self.root, run_id)
        self.assertFalse(diagnosis.source_repair_allowed)
        with self.assertRaisesRegex(ValueError, "source repair is not allowed"):
            propose_operator_repair(
                self.root,
                run_id,
                REPLACEMENT_SOURCE,
                "Do not hide a compiler limitation.",
            )


if __name__ == "__main__":
    unittest.main()

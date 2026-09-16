from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.develop_operator import (
    audit_test_contract,
    choose_references,
    extract_pytest_summary,
    load_operator_spec,
    may_repair,
    update_task_validation_record,
    unauthorized_changes,
)


def valid_spec() -> dict:
    return {
        "schema_version": 1,
        "name": "tanh_and_mul",
        "semantics": "Compute tanh(x) multiplied elementwise by y.",
        "pytorch_reference": "torch.tanh(x) * y",
        "inputs": [
            {"name": "x", "description": "activation input"},
            {"name": "y", "description": "multiplication input"},
        ],
        "output": "tanh(x) * y",
        "shape_cases": [[512], [1023]],
        "input_shape_cases": [{"x": [4, 1], "y": [1, 8]}],
        "dtypes": ["torch.float32", "torch.float16"],
        "tolerances": {"rtol": 0.01, "atol": 0.01},
        "backward": True,
        "reference_operators": ["sigmoid_and_mul"],
    }


class DevelopOperatorTests(unittest.TestCase):
    def test_loads_spec_and_applies_safe_default_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")

            spec = load_operator_spec(spec_path, root)

            self.assertEqual(
                spec.implementation_file,
                "python/examples/flaggems/tanh_and_mul.py",
            )
            self.assertEqual(
                spec.test_file,
                "python/examples/flaggems/test_tanh_and_mul.py",
            )

    def test_contract_audit_requires_reference_coverage_and_backward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            test_path = root / spec.test_file
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                """import pytest
import torch

from .tanh_and_mul import tanh_and_mul, tanh_and_mul_backward


@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tanh_and_mul_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    y = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


def test_tanh_and_mul_broadcast():
    x = torch.randn((4, 1), dtype=torch.float32)
    y = torch.randn((1, 8), dtype=torch.float32)
    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


def test_tanh_and_mul_backward():
    x = torch.randn((512,), requires_grad=True)
    y = torch.randn((512,), requires_grad=True)
    expected = torch.tanh(x) * y
    expected.backward(torch.ones_like(expected))
    dx, dy = tanh_and_mul_backward(torch.ones_like(expected), x.detach(), y.detach())
    torch.testing.assert_close(dx, x.grad, rtol=0.01, atol=0.01)
    torch.testing.assert_close(dy, y.grad, rtol=0.01, atol=0.01)
""",
                encoding="utf-8",
            )

            audit = audit_test_contract(spec, root)

            self.assertEqual(audit.status, "passed")
            self.assertEqual(audit.errors, [])
            self.assertIsNotNone(audit.test_sha256)

    def test_contract_audit_rejects_weakened_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_data = valid_spec()
            spec_data["backward"] = False
            spec_data["input_shape_cases"] = []
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            test_path = root / spec.test_file
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                """import pytest
import torch
from .tanh_and_mul import tanh_and_mul
@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * x
    actual = tanh_and_mul(x, x)
    torch.testing.assert_close(actual, expected, rtol=1.0, atol=1.0)
""",
                encoding="utf-8",
            )

            audit = audit_test_contract(spec, root)

            self.assertEqual(audit.status, "failed")
            self.assertTrue(any("exceeds" in error for error in audit.errors))

    def test_reference_selection_prefers_explicit_and_fused_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            inventory = {
                "operators": [
                    {
                        "name": "add",
                        "implementation_file": "add.py",
                        "torch_references": ["torch.add"],
                        "tl_ops": ["load", "store"],
                        "public_functions": ["add"],
                    },
                    {
                        "name": "sigmoid_and_mul",
                        "implementation_file": "sigmoid_and_mul.py",
                        "torch_references": ["torch.sigmoid"],
                        "tl_ops": ["exp", "load", "store"],
                        "public_functions": ["sigmoid_and_mul_backward"],
                    },
                ]
            }

            references = choose_references(spec, inventory)

            self.assertEqual(references[0]["name"], "sigmoid_and_mul")

    def test_repair_policy_stops_for_environment_and_locks_compiler_by_default(self) -> None:
        self.assertTrue(may_repair("correctness", False))
        self.assertFalse(may_repair("environment", True))
        self.assertFalse(may_repair("buddy-opt", False))
        self.assertTrue(may_repair("buddy-opt", True))

    def test_detects_changes_outside_allowlist(self) -> None:
        before = {"implementation.py": "a", "README.md": "old"}
        after = {"implementation.py": "b", "README.md": "new", "extra.py": "x"}

        changed = unauthorized_changes(before, after, {"implementation.py"})

        self.assertEqual(changed, ["README.md", "extra.py"])

    def test_extracts_latest_pytest_summary(self) -> None:
        text = "first\n2 failed, 3 passed in 1.25s\nretry\n20 passed in 16.79s\n"

        self.assertEqual(extract_pytest_summary(text), "20 passed in 16.79s")

    def test_rejects_operator_path_outside_allowlisted_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_data = valid_spec()
            spec_data["implementation_file"] = "../outside.py"
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec_data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "repository-relative"):
                load_operator_spec(spec_path, root)

    def test_contract_audit_requires_declared_broadcast_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            test_path = root / spec.test_file
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                """import pytest
import torch
from .tanh_and_mul import tanh_and_mul, tanh_and_mul_backward
@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * x
    actual = tanh_and_mul(x, x)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
def test_backward():
    x = torch.randn((512,), requires_grad=True)
    expected = torch.tanh(x) * x
    expected.backward(torch.ones_like(expected))
    dx, dy = tanh_and_mul_backward(torch.ones_like(expected), x, x)
    torch.testing.assert_close(dx, x.grad, rtol=0.01, atol=0.01)
""",
                encoding="utf-8",
            )

            audit = audit_test_contract(spec, root)

            self.assertEqual(audit.status, "failed")
            self.assertTrue(any("input-shape case" in item for item in audit.errors))

    def test_task_validation_record_is_replaced_instead_of_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task_path = Path(temp_dir) / "task.md"
            task_path.write_text("# Task\n", encoding="utf-8")
            final = {
                "status": "passed",
                "locked_test_sha256": "abc",
                "repair_attempts": 1,
                "validations": [
                    {
                        "iteration": 1,
                        "status": "passed",
                        "failure_stage": None,
                        "test_summary": "2 passed in 1.00s",
                        "error_excerpt": [],
                    }
                ],
            }

            update_task_validation_record(task_path, final)
            update_task_validation_record(task_path, final)
            content = task_path.read_text(encoding="utf-8")

            self.assertEqual(content.count("Autonomous Validation Record"), 1)


if __name__ == "__main__":
    unittest.main()

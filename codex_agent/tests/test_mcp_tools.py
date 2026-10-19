from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from codex_agent.harness.mcp_demo import run_mcp_demo
from codex_agent.harness.mcp_server import server
from codex_agent.operator_tools import discover_operator_evidence


class OperatorToolTests(unittest.TestCase):
    def test_discovers_existing_operator_evidence(self) -> None:
        result = discover_operator_evidence(Path.cwd(), "relu_and_mul")

        self.assertEqual(result.status, "found")
        self.assertIsNotNone(result.operator)
        assert result.operator is not None
        self.assertEqual(
            result.operator.implementation_file,
            "python/examples/flaggems/relu_and_mul.py",
        )
        self.assertIn(
            "python/examples/flaggems/test_relu_and_mul.py",
            result.operator.test_files,
        )
        self.assertIn("load", result.operator.tl_ops)
        self.assertTrue(
            result.operator.validation_command.startswith("python -m pytest")
        )

    def test_unknown_operator_returns_bounded_suggestions(self) -> None:
        result = discover_operator_evidence(Path.cwd(), "relu_and_mu")

        self.assertEqual(result.status, "not_found")
        self.assertIn("relu_and_mul", result.suggestions)
        self.assertLessEqual(len(result.suggestions), 5)

    def test_rejects_path_traversal_as_an_operator_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Python identifier"):
            discover_operator_evidence(Path.cwd(), "../relu_and_mul")


class McpOperatorToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_lists_lifecycle_and_calls_discover_operator(self) -> None:
        with patch.dict(
            os.environ,
            {"TRITON_RISCV_REPO_ROOT": str(Path.cwd())},
        ):
            async with Client(server) as client:
                listed = await client.list_tools()
                names = {item.name for item in listed.tools}
                tool = next(
                    item for item in listed.tools if item.name == "discover_operator"
                )
                result = await client.call_tool(
                    "discover_operator",
                    {"operator_name": "relu_and_mul"},
                )

        self.assertEqual(
            names,
            {
                "discover_operator",
                "validate_operator",
                "diagnose_failure",
                "propose_repair",
                "apply_repair",
            },
        )
        self.assertTrue(tool.annotations.read_only_hint)
        self.assertFalse(tool.annotations.destructive_hint)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "found")
        self.assertEqual(
            result.structured_content["operator"]["name"],
            "relu_and_mul",
        )

    async def test_mcp_validation_defaults_to_a_nonexecuting_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator_root = root / "python/examples/flaggems"
            operator_root.mkdir(parents=True)
            (operator_root / "demo.py").write_text(
                "import triton\n@triton.jit\ndef demo_kernel(x):\n    return x\n",
                encoding="utf-8",
            )
            (operator_root / "test_demo.py").write_text(
                "def test_demo():\n    assert True\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TRITON_RISCV_REPO_ROOT": str(root)}):
                async with Client(server) as client:
                    result = await client.call_tool(
                        "validate_operator",
                        {"operator_name": "demo"},
                    )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "planned")

    async def test_stdio_demo_exposes_and_calls_the_same_tool(self) -> None:
        payload = await run_mcp_demo(Path.cwd(), "relu_and_mul")

        self.assertEqual(payload["transport"], "stdio")
        self.assertIn("discover_operator", payload["available_tools"])
        self.assertEqual(payload["result"]["status"], "found")
        self.assertEqual(payload["result"]["operator"]["name"], "relu_and_mul")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from codex_agent.validate_operator import classify_log, extract_error_excerpt


class ValidateOperatorTests(unittest.TestCase):
    def test_classifies_success_and_timeout(self) -> None:
        self.assertEqual(classify_log("20 passed", 0), ("passed", None, None))
        self.assertEqual(classify_log("TIMEOUT", 124)[1], "timeout")

    def test_classifies_environment_and_import_failures(self) -> None:
        environment = "scripts/triton-riscv-env.sh: No such file or directory"
        self.assertEqual(classify_log(environment, 1)[1], "environment")
        self.assertEqual(classify_log("ModuleNotFoundError: triton", 1)[1], "import")

    def test_classifies_build_failure(self) -> None:
        self.assertEqual(classify_log("ERROR: Failed building wheel for triton", 1)[1], "build")

    def test_classifies_target_capability_failure(self) -> None:
        log = "ValueError: type fp8e4nv not supported in this architecture"
        status, stage, reason = classify_log(log, 1)

        self.assertEqual(status, "failed")
        self.assertEqual(stage, "target-capability")
        self.assertIn("FP8", reason)

    def test_classifies_unavailable_triton_math_api(self) -> None:
        log = (
            "triton.compiler.errors.CompilationError\n"
            "AttributeError: module triton.language.math has no attribute acos"
        )

        self.assertEqual(classify_log(log, 1)[1], "triton-frontend")

    def test_classifies_buddy_tptr_failure(self) -> None:
        log = "error: Dialect `tptr' not found for custom op 'tptr.type_offset'"

        self.assertEqual(classify_log(log, 1)[1], "buddy-opt")

    def test_classifies_buddy_ttx_failure(self) -> None:
        log = "error: Dialect `ttx' not found for custom op 'ttx.cumsum'"
        status, stage, reason = classify_log(log, 1)

        self.assertEqual(status, "failed")
        self.assertEqual(stage, "buddy-opt")
        self.assertIn("ttx", reason)

    def test_classifies_unsupported_buddy_reduction(self) -> None:
        log = "error: unsupported linalg.reduce for -lower-linalg-to-vir"
        status, stage, reason = classify_log(log, 1)

        self.assertEqual(status, "failed")
        self.assertEqual(stage, "buddy-opt")
        self.assertIn("linalg.reduce", reason)

    def test_classifies_linalg_translation_failure(self) -> None:
        log = "error: Dialect `linalg' not found for custom op 'linalg.generic'"

        self.assertEqual(classify_log(log, 1)[1], "mlir-translate")

    def test_classifies_correctness_failure(self) -> None:
        log = "torch.testing.assert_close failed: Mismatched elements: 5 / 10"

        self.assertEqual(classify_log(log, 1)[1], "correctness")

    def test_extracts_and_deduplicates_primary_errors(self) -> None:
        log = (
            "/tmp/tmp123/ll.mlir:10: error: Dialect `linalg' not found\n"
            "/tmp/tmp456/ll.mlir:10: error: Dialect `linalg' not found\n"
            "FAILED test_example.py::test_op\n"
        )

        excerpt = extract_error_excerpt(log)

        self.assertEqual(len(excerpt), 1)
        self.assertIn("Dialect `linalg'", excerpt[0])


if __name__ == "__main__":
    unittest.main()

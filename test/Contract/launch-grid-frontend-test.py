# RUN: %PYTHON %s
"""Check that the three real drivers pass their explicit launch and ABI promises."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples/BOSCAMEFPGA"


class LaunchGridFrontendTest(unittest.TestCase):
    def test_exporter_receives_explicit_grid_and_abi(self):
        from triton.backends.compiler import GPUTarget
        from triton.backends.triton_shared import contract
        from triton.compiler import ASTSource

        for name, grid in (
            ("matmul_i8_128x1024x128", (2, 16, 1)),
            ("matmul_i8_196x768x768", (7, 6, 1)),
            ("layernorm_197x768", (197, 1, 1)),
        ):
            with (
                self.subTest(example=name),
                tempfile.TemporaryDirectory() as temporary,
                patch.object(sys, "path", [str(EXAMPLES / name), *sys.path]),
                patch.dict(sys.modules),
            ):
                # The standalone drivers use the same local module names.
                sys.modules.pop("kernels", None)
                sys.modules.pop("kernel", None)
                spec = importlib.util.spec_from_file_location(
                    "driver_" + name, EXAMPLES / name / "build.py"
                )
                driver = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(driver)
                driver.OUT = Path(temporary)
                artifact = Mock(
                    ttir=f'@{driver.SYMBOL}( math.rsqrt "tt.reduce" "tt.reduce" tt.store'
                )
                with patch.object(
                    contract, "export_contract", return_value=artifact
                ) as export:
                    driver.export_frontend()
                export.assert_called_once()
                args, kwargs = export.call_args
                self.assertIsInstance(args[0], ASTSource)
                self.assertEqual(args[0].signature, driver.SIGNATURE)
                self.assertEqual(
                    args[0].constants,
                    {
                        (args[0].fn.arg_names.index(key),): value
                        for key, value in getattr(
                            driver, "CONSTEXPRS", {}
                        ).items()
                    },
                )
                self.assertEqual(
                    kwargs["target"], GPUTarget("boscame", "generic", 1)
                )
                self.assertEqual(kwargs["launch_grid"], grid)
                self.assertEqual(kwargs["entry_name"], "triton_" + name)
                self.assertEqual(
                    kwargs["caller_contract"], "args_aligned64_nonalias"
                )
                self.assertIsNone(kwargs["math_accuracy"])
                artifact.write.assert_called_once_with(Path(temporary))


if __name__ == "__main__":
    unittest.main()

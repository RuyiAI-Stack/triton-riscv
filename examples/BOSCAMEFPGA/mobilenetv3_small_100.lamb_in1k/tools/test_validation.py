#!/usr/bin/env python3
"""Offline regression checks for missed writes and independent host evidence."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from check_model_host import check_model
from cases import inventory
from validation import compare_logits, operator_pass, validate_model_inputs

ROOT = Path(__file__).resolve().parents[1]
CLANG = os.environ.get("CLANG", shutil.which("clang"))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def model_fixture(self):
        golden = struct.pack("<1000f", *range(1000))
        (self.directory / "golden-logits.f32").write_bytes(golden)
        (self.directory / "model-data.bin").write_bytes(golden)
        manifest = dict(
            model="validation-probe",
            used_cases=[],
            blob_bytes=len(golden),
            blob_sha256=hashlib.sha256(golden).hexdigest(),
            constant_offsets={"golden_logits": 0},
            logits_atol=0.002,
            logits_rtol=0.001,
            expected_top5=[999, 998, 997, 996, 995],
        )
        (self.directory / "model-manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_mean_reduction_block_fits_spatial_extent(self):
        for case in inventory():
            if case["family"] == "mean_hw":
                with self.subTest(case=case["name"]):
                    block = case["constexprs"]["BLOCK"]
                    spatial = case["constexprs"]["SPATIAL"]
                    self.assertGreater(block, 0)
                    self.assertLessEqual(block, spatial)
                    self.assertEqual(block & (block - 1), 0)

    def test_operator_evidence_requires_case_and_complete_comparison(self):
        good = (
            "verify relu_count24: PASS errors=00000000 "
            "max_abs_error_f32_bits=00000000 checked=00000018 validation_executed=1\r\n"
        )
        self.assertTrue(operator_pass(good, "relu_count24"))
        for bad in (
            good.replace("relu_count24:", "relu_count144:"),
            good.replace("00000018", "00000001"),
            good.replace("validation_executed=1", "validation_executed=0"),
            good.replace(
                "max_abs_error_f32_bits=00000000", "max_abs_error_f32_bits=7fc00000"
            ),
            good + good,
            ": PASS errors=00000000",
        ):
            self.assertFalse(operator_pass(bad, "relu_count24"), bad)

    @unittest.skipUnless(CLANG, "clang is required for the missed-write probe")
    def test_nan_sentinel_detects_omitted_zero_outputs(self):
        source = self.directory / "stub.c"
        source.write_text("""#include "support.h"
void _mlir_ciface_kernel_relu_count24(MemRef1 *x, MemRef1 *y) {
  float *input = x->aligned, *output = y->aligned;
  for (int i = 0; i < 24; ++i) {
#ifdef OMIT_ZERO
    if (input[i] <= 0) continue;
#endif
    output[i] = input[i] > 0 ? input[i] : 0;
  }
}
int main(void) { return launch(); }
""")
        for omit in (False, True):
            executable = self.directory / ("partial" if omit else "complete")
            command = [
                CLANG,
                "-O2",
                "-DHOST_TEST",
                "-I" + str(ROOT),
                str(ROOT / "support.c"),
                str(ROOT / "relu_count24/launch.c"),
                str(source),
                "-o",
                str(executable),
            ]
            if omit:
                command.append("-DOMIT_ZERO")
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            result = subprocess.run(
                [str(executable)], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, int(omit))
            self.assertEqual(operator_pass(result.stdout, "relu_count24"), not omit)

    def test_golden_blob_and_manifest_must_agree(self):
        manifest = self.model_fixture()
        validate_model_inputs(self.directory, manifest)
        for field, value in (
            ("blob_sha256", "0" * 64),
            ("blob_bytes", 1),
            ("logits_atol", float("inf")),
            ("expected_top5", [999] * 5),
        ):
            changed = {**manifest, field: value}
            with self.assertRaises(ValueError):
                validate_model_inputs(self.directory, changed)
        (self.directory / "golden-logits.f32").write_bytes(bytes(4000))
        with self.assertRaisesRegex(ValueError, "embedded reference"):
            validate_model_inputs(self.directory, manifest)

    def test_nonfinite_and_short_actual_logits_fail(self):
        manifest = self.model_fixture()
        actual = self.directory / "actual.f32"
        actual.write_bytes((self.directory / "golden-logits.f32").read_bytes())
        self.assertTrue(
            compare_logits(actual, self.directory / "golden-logits.f32", manifest)[
                "passed"
            ]
        )
        for value in (float("inf"), float("-inf"), float("nan")):
            actual.write_bytes(struct.pack("<1000f", value, *range(1, 1000)))
            result = compare_logits(
                actual, self.directory / "golden-logits.f32", manifest
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["nonfinite_count"], 1)
            self.assertEqual(result["mismatch_count"], 1)
        actual.write_bytes(bytes(3999))
        with self.assertRaises(struct.error):
            compare_logits(actual, self.directory / "golden-logits.f32", manifest)

    @unittest.skipUnless(CLANG, "clang is required for the external-export probe")
    def test_external_model_directory_and_failure_report(self):
        self.model_fixture()
        (self.directory / "runner.c").write_text("""#include "model_support.h"
extern const float mobilenet_model_blob[];
int launch(void) {
  model_start(0);
  return model_verify(mobilenet_model_blob, mobilenet_model_blob, 1000, 999, 0);
}
""")
        (self.directory / "model_blob.S").write_text(
            ".section .rodata\n.balign 64\n.global mobilenet_model_blob\nmobilenet_model_blob:\n"
            f'.incbin "{self.directory / "model-data.bin"}"\n'
            '.section .note.GNU-stack,"",@progbits\n'
        )
        args = SimpleNamespace(
            model=self.directory,
            kernels=self.directory / "kernels",
            clang=CLANG,
            jobs=2,
            timeout=30,
        )
        with redirect_stdout(io.StringIO()):
            result = check_model(args)
        self.assertTrue(result["passed"], result.get("error"))
        self.assertIn(str(self.directory / "runner.c"), result["sources_sha256"])
        (self.directory / "runner.c").write_text("invalid C source")
        with redirect_stdout(io.StringIO()):
            result = check_model(args)
        self.assertFalse(result["passed"])
        self.assertIn("Compilation failed", result["error"])
        self.assertFalse(
            json.loads((self.directory / "host-model-report.json").read_text())[
                "passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()

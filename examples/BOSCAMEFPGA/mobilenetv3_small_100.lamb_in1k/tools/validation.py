"""Independent numerical and provenance checks shared by host/QEMU runners."""

import hashlib
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def path_label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def input_hashes(paths):
    return {
        path_label(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
    }


def operator_pass(stdout, case):
    # Require this case's complete comparison, rather than any PASS substring.
    from cases import describe
    from export import argument_shapes

    count = math.prod(argument_shapes(describe(case))[-1])
    lines = [line for line in stdout.splitlines() if line.startswith("verify ")]
    if len(lines) != 1:
        return False
    match = re.fullmatch(r"verify " + re.escape(case) + r": PASS (.+)", lines[0])
    if not match:
        return False
    fields = {}
    for token in match[1].split():
        if token.count("=") != 1:
            return False
        key, value = token.split("=")
        if key in fields:
            return False
        fields[key] = value
    try:
        maximum = struct.unpack(
            "<f", struct.pack("<I", int(fields["max_abs_error_f32_bits"], 16))
        )[0]
        return (
            fields["errors"] == "00000000"
            and int(fields["checked"], 16) == count
            and fields["validation_executed"] == "1"
            and math.isfinite(maximum)
            and maximum >= 0
        )
    except (KeyError, ValueError, OverflowError, struct.error):
        return False


def validate_model_inputs(model, manifest):
    """Verify that the independent golden file is the one embedded in the image."""
    blob = (model / "model-data.bin").read_bytes()
    golden = (model / "golden-logits.f32").read_bytes()
    if len(golden) != 1000 * 4:
        raise ValueError("golden logits must contain exactly 1000 float32 values")
    if hashlib.sha256(blob).hexdigest() != manifest["blob_sha256"]:
        raise ValueError("model blob does not match model-manifest.json")
    if len(blob) != manifest["blob_bytes"]:
        raise ValueError("model blob byte count does not match model-manifest.json")
    offset = manifest["constant_offsets"]["golden_logits"]
    if (
        type(offset) is not int
        or offset < 0
        or blob[offset : offset + len(golden)] != golden
    ):
        raise ValueError("independent golden logits differ from the embedded reference")
    for name in ("logits_atol", "logits_rtol"):
        value = manifest[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    expected = manifest["expected_top5"]
    if (
        not isinstance(expected, list)
        or len(expected) != 5
        or any(type(i) is not int or not 0 <= i < 1000 for i in expected)
        or len(set(expected)) != 5
    ):
        raise ValueError("expected_top5 must contain five distinct class indices")
    if not all(math.isfinite(x) for x in struct.unpack("<1000f", golden)):
        raise ValueError("golden logits contain nonfinite values")


def compare_logits(actual_path, golden_path, manifest):
    actual = struct.unpack("<1000f", Path(actual_path).read_bytes())
    golden = struct.unpack("<1000f", Path(golden_path).read_bytes())
    differences = [
        abs(a - g) if math.isfinite(a) and math.isfinite(g) else math.inf
        for a, g in zip(actual, golden)
    ]
    errors = [
        i
        for i, (a, g, difference) in enumerate(zip(actual, golden, differences))
        if not math.isfinite(a)
        or not math.isfinite(g)
        or difference > manifest["logits_atol"] + manifest["logits_rtol"] * abs(g)
    ]
    top5 = sorted(range(1000), key=lambda i: actual[i], reverse=True)[:5]
    maximum = max(differences)
    return dict(
        max_abs_error=maximum if math.isfinite(maximum) else None,
        nonfinite_count=sum(
            not math.isfinite(a) or not math.isfinite(g) for a, g in zip(actual, golden)
        ),
        mismatch_count=len(errors),
        top5=top5,
        expected_top5=manifest["expected_top5"],
        actual_logits_sha256=hashlib.sha256(Path(actual_path).read_bytes()).hexdigest(),
        passed=not errors and top5 == manifest["expected_top5"],
    )

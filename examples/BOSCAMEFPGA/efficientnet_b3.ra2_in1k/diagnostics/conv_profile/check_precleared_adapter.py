#!/usr/bin/env python3
"""Native isolated adapter checks for once-per-invocation QPatch initialization."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_launch  # noqa: E402
from export import inventory  # noqa: E402
from quantized_conv_pipeline import pipeline_layout  # noqa: E402


def mock_source(case):
    text = check_launch.composite_mock_source(case)
    width = pipeline_layout(case)["width"]
    marker = "  ++im2col_calls;"
    verification = [
        "  Descriptor1D *patch = (Descriptor1D *)u1->descriptor;",
        "  unsigned char *bytes = patch->aligned;",
        "  for (int i = 0; i < 4096; ++i) {",
        f"    unsigned char expected = im2col_calls && i % 64 < {width} ? 0x5a : 0;",
        '    require(bytes[i] == expected, "QPatch once-per-public-call clear and padding reuse");',
        f"    if (i % 64 < {width}) bytes[i] = 0x5a;",
        "  }",
    ]
    assert text.count(marker) == 1
    text = text.replace(marker, "\n".join(verification) + "\n" + marker)
    pointers = ", ".join(f"a{i}" for i in range(4))
    call = f"  effnet_test_adapter({pointers});"
    wrapper = f"  reject_invalid({pointers});\n" + call
    replacement = (
        wrapper
        + "\n"
        + (
            "  phase = 0; weight_calls = im2col_calls = gemm_calls = dequant_calls = 0;\n"
            "  gemm_total = dequant_total = tile_index = 0;\n"
        )
        + call
    )
    assert text.count(wrapper) == 1
    text = text.replace(wrapper, replacement)
    main = "  if (!setjmp(completed)) {"
    text = text.replace(
        main,
        "  for (size_t i = 0; i < VIT_WORKSPACE_BYTES; ++i) ((unsigned char *)workspace(0))[i] = 0xa9;\n"
        + main,
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    build = args.output / "adapters"
    records = []
    check_launch.mock_source = mock_source
    for case in inventory():
        if case["family"] not in ("conv2d_f32", "dwconv2d_f32"):
            continue
        layout = pipeline_layout(case)
        destination = build / case["name"]
        destination.mkdir(parents=True, exist_ok=True)
        text = (
            ROOT / "build/w8a8-opt/integrated" / case["name"] / "adapter.c"
        ).read_text()
        marker = "  for (int32_t row = 0;"
        insertion = (
            f"  for (int i=0; i<4096; ++i) ((unsigned char *)(scratch + {layout['qpatch']['offset']}u))[i] = 0;\n"
            '  __asm__ volatile("" ::: "memory");\n'
        )
        assert text.count(marker) == 1
        text = text.replace(marker, insertion + marker)
        (destination / "adapter.c").write_text(text)
        options = SimpleNamespace(
            output=args.output / "checks",
            build_dir=build,
            cc=shutil.which("clang") or "cc",
            timeout=60,
        )
        record = check_launch.check_case(case, options)
        record["scratch_contract"] = (
            "dirty arena; two public adapter invocations; first raw sees all-zero QPatch; subsequent raw sees reused logical bytes and zero padding"
        )
        records.append(record)
    (args.output / "report.json").write_text(json.dumps(records, indent=2) + "\n")
    if len(records) != 18 or any(row["status"] != "ABI_SMOKE_PASS" for row in records):
        raise RuntimeError("precleared adapter validation incomplete")


if __name__ == "__main__":
    main()

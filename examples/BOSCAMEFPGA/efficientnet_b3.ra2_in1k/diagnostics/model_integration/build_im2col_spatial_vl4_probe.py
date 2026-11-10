#!/usr/bin/env python3
"""Build one full-output probe and query actual VL outside the adapter timer."""

import argparse
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "conv_profile"))
import output_probe  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = "dwconv2d_72x72x192_k3_s1"
    output_probe.build(name, args.output.resolve(), args.kernel_root.resolve())
    directory = args.output.resolve() / "fused/all/kernels" / name
    launch = directory / "launch.c"
    source = launch.read_text()
    query = """  uintptr_t spatial_actual_vl;
  __asm__ volatile(".option push\\n.option arch,+v\\nvsetivli %0,4,e32,m1,ta,ma\\n.option pop"
                   : "=r"(spatial_actual_vl) :: "memory");
  nr_puts("PROFILE spatial_actual_vl=0x"); nr_hex64(spatial_actual_vl); nr_puts("\\n");
"""
    assert source.count("  return 0;") == 1
    launch.write_text(source.replace("  return 0;", query + "  return 0;"))
    for label in ("launch", "link", "audit", "bin"):
        command_line = (directory / (label + ".log")).read_text().splitlines()[0]
        assert command_line.startswith("$ ")
        output_probe.run(shlex.split(command_line[2:]), directory, "vl-query-" + label)
    manifest_path = directory / "profile-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    image = directory / (name + ".bin")
    manifest["bin_sha256"] = output_probe.digest(image)
    manifest["vl_query"] = {
        "instruction": "vsetivli rd,4,e32,m1,ta,ma",
        "timing_scope": "after full adapter return; outside adapter timer; inside launch timer",
        "kernel_object_unchanged": True,
        "generator_sha256": output_probe.digest(__file__),
        "launch_source_sha256": output_probe.digest(launch),
    }
    output_probe.save(manifest_path, manifest)
    print("PASS actual-VL-query probe", manifest["bin_sha256"])


if __name__ == "__main__":
    main()

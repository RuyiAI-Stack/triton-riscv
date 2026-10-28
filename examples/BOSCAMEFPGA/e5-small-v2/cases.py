"""Read fixed specializations from each kernel.py without importing Triton.

KERNEL_CONFIG must be a literal dict so inventory/model tools can inspect it
with only the Python standard library. Numerical bodies stay in their cases.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POINTER_TYPES = {"f32": "*fp32", "i32": "*i32", "i8": "*i8"}


def describe(name):
    if Path(name).name != name:
        raise ValueError("invalid case name: " + name)
    source = ROOT / name / "kernel.py"
    if not source.is_file():
        raise ValueError("unknown case: " + name)
    module = ast.parse(source.read_text(), filename=str(source))
    configs = [
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "KERNEL_CONFIG"
                for t in node.targets)
    ]
    if len(configs) != 1:
        raise ValueError(f"{source}: expected one literal KERNEL_CONFIG")
    config = ast.literal_eval(configs[0])
    # Descriptive annotations from the former JSON remain beside the kernel,
    # while the public descriptor retains its existing ABI and report schema.
    config.pop("metadata", None)
    for key in ("family", "kernel", "arguments", "shape", "constexprs", "grid", "roles"):
        if key not in config:
            raise ValueError(f"{source}: missing KERNEL_CONFIG[{key!r}]")
    for key in ("grid", "shape"):
        if not config[key] or any(type(v) is not int or v <= 0 for v in config[key]):
            raise ValueError(f"{source}: {key} must contain positive integers")
    if len(config["grid"]) != 3:
        raise ValueError(f"{source}: grid must have three dimensions")
    return {
        "name": name,
        **config,
        "kernel_module": "kernel",
        "kernel_source": str(source),
        "symbol": "triton_" + name,
        "signature": {
            arg["name"]: POINTER_TYPES[arg["dtype"]]
            for arg in config["arguments"]
        },
        "directory": str(source.parent),
        "launch_source": str(source.parent / "launch.c"),
        "validation": "Independent launch oracle; contiguous tensors",
    }


def inventory():
    return [describe(path.parent.name) for path in sorted(ROOT.glob("*/kernel.py"))]


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))

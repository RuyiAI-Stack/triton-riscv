"""Case registry for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.

Inputs: per-case kernel.py files beside this registry; each must define
KERNEL_CONFIG used by export and Make.

Outputs: names, shapes, dtypes, and launch metadata for batch build, simcheck,
and board-run helpers.

When used: model-level export.py / report.py / makefiles walk this list instead
of hard-coding operator names.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POINTER_TYPES = {"f32": "*fp32", "i32": "*i32", "i8": "*i8"}
W8A8_CONV_BINDINGS = {
    "conv2d_f32": (
        "conv2d_im2col_w8a8",
        "im2col_w8a8",
        "quantized_conv_kernels",
    ),
    "dwconv2d_f32": (
        "dwconv2d_im2col_w8a8",
        "im2col_w8a8_depthwise",
        "quantized_conv_kernels",
    ),
}


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
        and any(
            isinstance(t, ast.Name) and t.id == "KERNEL_CONFIG" for t in node.targets
        )
    ]
    if len(configs) != 1:
        raise ValueError(f"{source}: expected one literal KERNEL_CONFIG")
    config = ast.literal_eval(configs[0])
    if config.get("family") == "silu":
        if (
            config.get("implementation"),
            config.get("implementation_source"),
            config.get("implementation_symbol_prefix"),
            config.get("reference_kernel"),
            config.get("kernel_module"),
        ) != (
            "hybrid_rvv_c",
            "diagnostics/model_integration/silu_rvv_direct_output_candidate.c",
            "efficientnet_rvv_",
            "silu_scalar_reference",
            "silu_rvv_kernels",
        ):
            raise ValueError(
                f"{source}: SiLU must bind to the verified native RVV implementation"
            )
        config["implementation_symbol"] = config["implementation_symbol_prefix"] + name
    if config.get("family") == "gap":
        if (
            config.get("kernel"),
            config.get("implementation"),
            config.get("implementation_source"),
            config.get("implementation_symbol_prefix"),
            config.get("reference_kernel"),
            config.get("kernel_module"),
        ) != (
            "gap_phase_ordered",
            "hybrid_ordered_c",
            "diagnostics/model_integration/gap_torch_order_candidate.c",
            "efficientnet_phase_",
            "gap_scalar_reference",
            "gap_phase_kernels",
        ):
            raise ValueError(
                f"{source}: GAP must bind to the verified ordered C implementation"
            )
        if config["constexprs"]["BLOCK"] not in (32, 64, 128):
            raise ValueError(f"{source}: unsupported ordered GAP accumulator width")
        config["implementation_symbol"] = config["implementation_symbol_prefix"] + name
    if config.get("family") == "linear_f32":
        if (
            config.get("kernel"),
            config.get("implementation"),
            config.get("kernel_module"),
        ) != ("linear_w8a8", "w8a8_gemm", "quantized_linear_kernels") or any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(module)
        ):
            raise ValueError(f"{source}: classifier must bind directly to W8A8 stages")
    if config.get("family") in W8A8_CONV_BINDINGS:
        expected_kernel, expected_implementation, expected_module = W8A8_CONV_BINDINGS[
            config["family"]
        ]
        if (
            config.get("kernel"),
            config.get("implementation"),
            config.get("kernel_module"),
        ) != (
            expected_kernel,
            expected_implementation,
            expected_module,
        ):
            raise ValueError(
                f"{source}: FP32 convolution must bind to {expected_kernel}"
            )
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(module)
        ):
            raise ValueError(
                f"{source}: staged W8A8 convolution cases must not define a local kernel"
            )
    # Descriptive annotations from the former JSON remain beside the kernel,
    # while the public descriptor retains its existing ABI and report schema.
    config.pop("metadata", None)
    for key in (
        "family",
        "kernel",
        "arguments",
        "shape",
        "constexprs",
        "grid",
        "roles",
    ):
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
        "kernel_module": config.get("kernel_module", "kernel"),
        "kernel_source": str(source),
        "symbol": "triton_" + name,
        "signature": {
            arg["name"]: POINTER_TYPES[arg["dtype"]] for arg in config["arguments"]
        },
        "directory": str(source.parent),
        "launch_source": str(source.parent / "launch.c"),
        "validation": (
            "Independent C launch oracle; contiguous tensors"
            if config["family"] in ("add", "silu")
            else "C launch smoke test; numerical acceptance requires simcheck"
        ),
    }


def inventory():
    return [describe(path.parent.name) for path in sorted(ROOT.glob("*/kernel.py"))]


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))

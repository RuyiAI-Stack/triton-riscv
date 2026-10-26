"""BOSCAME AOT Contract export. This module never selects or opens a device."""

import hashlib
import json
import math
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import _get_triton_shared_opt_path

AOT_ONLY = (
    "boscame is AOT-only; use export_contract() to export a verified Contract"
)


@dataclass(frozen=True)
class BoscameOptions:
    # Explicit AST frontend defaults, independent of CPU/IME environment options.
    debug: bool = False
    arch: str = "generic"
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    allow_fp_reassoc: bool = False
    extern_libs: tuple = ()
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    supported_fp8_dtypes: tuple = ("fp8e4nv",)
    allow_fp8e4nv: bool = True
    max_num_imprecise_acc_default: int = 0
    allowed_dot_input_precisions: tuple = ("ieee",)
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    fpsan_homomorphic_casts: bool = False
    enable_make_gather_scatter: bool = True
    enable_collapse_shape: bool = False

    @classmethod
    def parse(cls, options):
        options = dict(options or {})
        # Only these knobs have defined Contract frontend semantics.
        allowed = {
            "debug",
            "enable_fp_fusion",
            "allow_fp_reassoc",
            "enable_make_gather_scatter",
            "enable_collapse_shape",
        }
        unknown = options.keys() - allowed
        if unknown:
            raise ValueError(
                "unknown boscame options: " + ", ".join(sorted(unknown))
            )
        if any(type(value) is not bool for value in options.values()):
            raise ValueError("boscame option values must be boolean")
        if options.get("enable_fp_fusion") or options.get("allow_fp_reassoc"):
            raise ValueError(
                "boscame Contract export forbids FP fusion and reassociation"
            )
        return cls(**options)


@dataclass(frozen=True)
class ContractArtifact:
    ttir: str
    contract: str
    manifest: dict

    def write(self, directory):
        """Write a reviewable handoff; no executable or persistent cache is made."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "kernel.ttir").write_text(self.ttir)
        (directory / "kernel.frontend.mlir").write_text(self.contract)
        (directory / "contract-manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
        )


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(argv):
    process = subprocess.run(
        list(map(str, argv)), text=True, capture_output=True
    )
    if process.returncode:
        raise ValueError(process.stderr.strip() or process.stdout.strip())
    return process.stdout


def _math_attribute(accuracy):
    if accuracy is None:
        return None
    if not isinstance(accuracy, dict) or not accuracy:
        raise ValueError("math_accuracy must be a nonempty erf/exp dictionary")
    entries = []
    for name, budget in sorted(accuracy.items()):
        if name not in {"erf", "exp"} or not isinstance(budget, dict):
            raise ValueError("math_accuracy accepts erf/exp dictionaries only")
        if set(budget) != {
            "domain",
            "special_values",
            "max_abs_error",
            "max_rel_error",
        }:
            raise ValueError(
                "math_accuracy requires the closed four-field accuracy schema"
            )
        if budget["domain"] != "all-f32" or budget["special_values"] != "ieee":
            raise ValueError(
                "math accuracy requires domain=all-f32 and special_values=ieee"
            )
        errors = []
        for key in ("max_abs_error", "max_rel_error"):
            value = budget[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "math error budgets must be finite and nonnegative"
                )
            errors.append(f"{key} = {float(value):.17e} : f64")
        entries.append(
            name
            + ' = {domain = "all-f32", special_values = "ieee", '
            + ", ".join(errors)
            + "}"
        )
    return "{" + ", ".join(entries) + "}"


class BoscameDelegate:
    """An ordinary delegate; CPUBackend remains the sole concrete plugin backend."""

    def __init__(self, target):
        if (target.backend, target.arch, target.warp_size) != (
            "boscame",
            "generic",
            1,
        ):
            raise ValueError(
                'boscame requires GPUTarget("boscame", "generic", 1)'
            )
        self.target = target

    def export(
        self,
        backend,
        source,
        *,
        launch_grid,
        entry_name=None,
        caller_contract=None,
        math_accuracy=None,
        options=None,
    ):
        import triton
        import triton.compiler.code_generator as code_generator
        import triton.compiler.compiler as compiler
        from triton._C import libtriton
        from triton._C.libtriton import ir
        from triton.compiler import ASTSource
        from triton.compiler.compiler import Language

        if (
            not isinstance(launch_grid, (tuple, list))
            or len(launch_grid) != 3
            or any(
                type(value) is not int or not 0 < value <= 2147483647
                for value in launch_grid
            )
        ):
            raise ValueError(
                "launch_grid must contain three positive static int32 extents"
            )
        if entry_name is not None and (
            not isinstance(entry_name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$]*", entry_name)
        ):
            raise ValueError("entry_name must be a nonempty MLIR symbol name")
        if caller_contract not in (None, "args_aligned64_nonalias"):
            raise ValueError(
                "unsupported caller_contract; expected args_aligned64_nonalias or None"
            )
        accuracy = _math_attribute(math_accuracy)
        effective = BoscameOptions.parse(options)
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        tool = _get_triton_shared_opt_path()
        tool_sha = _file_sha(tool)
        build_identity = json.loads(_run([tool, "--contract-build-info"]))

        with tempfile.TemporaryDirectory(
            prefix="triton-contract-"
        ) as temporary:
            temporary = Path(temporary)
            raw, prepared, normalized = (
                temporary / name
                for name in ("input.ttir", "prepared.ttir", "normalized.ttir")
            )
            output, report = (
                temporary / "contract.mlir",
                temporary / "report.json",
            )
            if isinstance(source, ASTSource):
                if source.language != Language.TRITON:
                    raise ValueError(
                        "Contract export accepts Triton ASTSource only, not Gluon"
                    )
                module = source.make_ir(
                    self.target,
                    effective,
                    backend.get_codegen_implementation(effective),
                    backend.get_module_map(),
                    context,
                )
                raw.write_text(str(module))
                input_identity = {"kind": "triton-ast", "sha256": source.hash()}
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if path.suffix not in (".ttir", ".mlir"):
                    raise ValueError("Contract export accepts TTIR files only")
                data = path.read_bytes()
                raw.write_bytes(data)
                input_identity = {"kind": "ttir", "sha256": _sha(data)}
            else:
                raise TypeError(
                    "source must be a Triton ASTSource or TTIR file"
                )

            prepare = []
            if entry_name:
                prepare.append("entry-name=" + entry_name)
            if accuracy:
                prepare.append("math-accuracy='" + accuracy + "'")
            # Version checks and symbol API rename happen before any graph rewrite.
            _run(
                [
                    tool,
                    raw,
                    "--triton-contract-prepare=" + " ".join(prepare),
                    "-o",
                    prepared,
                ]
            )
            module = ir.parse_mlir_module(str(prepared), context)
            module.context = context
            name = module.get_entry_func_name()
            module = backend.make_ttir(module, {}, effective)
            ttir = str(module)
            normalized.write_text(ttir)
            flags = [
                "contract=true",
                "launch-grid=" + name + ":" + ",".join(map(str, launch_grid)),
                "enable-make-gather-scatter="
                + str(effective.enable_make_gather_scatter).lower(),
                "enable-collapse-shape="
                + str(effective.enable_collapse_shape).lower(),
            ]
            if caller_contract:
                flags.append("caller-contract=" + caller_contract)
            _run(
                [
                    tool,
                    normalized,
                    "--triton-to-linalg-experimental=" + " ".join(flags),
                    "--kernel-contract-verify",
                    "-o",
                    output,
                ]
            )
            # Reverify the serialized artifact, using a fresh parser and context.
            _run(
                [
                    tool,
                    output,
                    "--triton-contract-report=output=" + str(report),
                    "-o",
                    temporary / "verified.mlir",
                ]
            )
            contract = output.read_text()
            metadata = json.loads(report.read_text())
            entry = next(
                item for item in metadata["functions"] if item["name"] == name
            )
        if _file_sha(tool) != tool_sha:
            raise RuntimeError(
                "frontend tool changed during Contract export; rebuild and retry"
            )
        manifest = {
            "schema_version": 1,
            "contract_schema_version": 1,
            "verified": True,
            "input": input_identity,
            "outputs": {
                "ttir_sha256": _sha(ttir.encode()),
                "contract_sha256": _sha(contract.encode()),
            },
            "target": {"backend": "boscame", "arch": "generic", "warp_size": 1},
            "entry": entry,
            "launch_grid": list(launch_grid),
            "explicit_promises": {
                "caller_contract": caller_contract,
                "math_accuracy": math_accuracy,
            },
            "effective_promises": {
                "caller_contract": metadata["caller_contract"],
                "math_accuracy": entry["math_accuracy"],
            },
            "effective_options": asdict(effective),
            "functions": metadata["functions"],
            "tool": {"sha256": tool_sha, "build": build_identity},
            "triton": {
                "version": triton.__version__,
                "binary_sha256": _file_sha(libtriton.__file__),
                "compiler_sha256": _file_sha(compiler.__file__),
                "code_generator_sha256": _file_sha(code_generator.__file__),
            },
            "exporter_sha256": _file_sha(__file__),
        }
        return ContractArtifact(ttir, contract, manifest)


def export_contract(
    source,
    *,
    target,
    launch_grid,
    entry_name=None,
    caller_contract=None,
    math_accuracy=None,
    options=None,
):
    """Export verified standard MLIR through the registered BOSCAME AOT backend."""
    from triton.compiler.compiler import make_backend

    backend = make_backend(target)
    if target.backend != "boscame":
        raise ValueError("export_contract requires the boscame target")
    return backend.export_contract(
        source,
        launch_grid=launch_grid,
        entry_name=entry_name,
        caller_contract=caller_contract,
        math_accuracy=math_accuracy,
        options=options,
    )


def main():
    import argparse

    from triton.backends.compiler import GPUTarget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--launch-grid", required=True, help="x,y,z")
    parser.add_argument("--entry-name")
    parser.add_argument(
        "--caller-contract", choices=("args_aligned64_nonalias",)
    )
    parser.add_argument(
        "--math-accuracy", type=Path, help="explicit accuracy JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        grid = tuple(int(value) for value in args.launch_grid.split(","))
        artifact = export_contract(
            args.source,
            target=GPUTarget("boscame", "generic", 1),
            launch_grid=grid,
            entry_name=args.entry_name,
            caller_contract=args.caller_contract,
            math_accuracy=json.loads(args.math_accuracy.read_text())
            if args.math_accuracy
            else None,
        )
    except (ValueError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    artifact.write(args.output)


if __name__ == "__main__":
    main()

"""RISC-V ELF linking and QEMU execution helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Sequence

DEFAULT_TRIPLE = "riscv64-unknown-linux-gnu"
DEFAULT_MARCH = "rv64gcv_zfh_zvfh_zba_zbb"
DEFAULT_MABI = "lp64d"
DEFAULT_LLC_FEATURES = "+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb"


def _find_tool(
    configured: str | None, candidates: Sequence[str], description: str
) -> str:
    if configured:
        path = shutil.which(configured) if not os.path.isabs(configured) else configured
        if path and Path(path).is_file():
            return str(path)
        raise RuntimeError(f"Configured {description} does not exist: {configured}")

    for candidate in candidates:
        if path := shutil.which(candidate):
            return path
    raise RuntimeError(
        f"Unable to find {description}; configure its TRITON_RISCV_* variable"
    )


@dataclass(frozen=True)
class RiscvToolchain:
    triple: str = DEFAULT_TRIPLE
    march: str = DEFAULT_MARCH
    mabi: str = DEFAULT_MABI
    llc_features: str = DEFAULT_LLC_FEATURES
    cc: str | None = None
    objdump: str | None = None
    sysroot: str | None = None
    qemu: str | None = None
    qemu_cpu: str | None = None

    @classmethod
    def from_env(cls) -> "RiscvToolchain":
        return cls(
            triple=os.getenv("TRITON_RISCV_TARGET_TRIPLE", DEFAULT_TRIPLE),
            march=os.getenv("TRITON_RISCV_MARCH", DEFAULT_MARCH),
            mabi=os.getenv("TRITON_RISCV_MABI", DEFAULT_MABI),
            llc_features=os.getenv("TRITON_RISCV_LLC_FEATURES", DEFAULT_LLC_FEATURES),
            cc=os.getenv("TRITON_RISCV_CC") or None,
            objdump=os.getenv("TRITON_RISCV_OBJDUMP") or None,
            sysroot=os.getenv("TRITON_RISCV_SYSROOT") or None,
            qemu=os.getenv("TRITON_RISCV_QEMU") or None,
            qemu_cpu=os.getenv("TRITON_RISCV_QEMU_CPU") or None,
        )

    def llc_command(
        self, llc: str, source: str | os.PathLike, output: str | os.PathLike
    ) -> list[str]:
        return [
            llc,
            str(source),
            "-filetype=obj",
            f"-mtriple={self.triple}",
            f"-mattr={self.llc_features}",
            "-relocation-model=pic",
            "-o",
            str(output),
        ]

    def linker(self) -> str:
        return _find_tool(
            self.cc,
            (
                f"{self.triple}-clang",
                "riscv64-linux-gnu-gcc",
                "riscv64-unknown-linux-gnu-gcc",
                "clang",
            ),
            "RISC-V C compiler",
        )

    def objdump_binary(self) -> str:
        if self.objdump:
            return _find_tool(self.objdump, (), "RISC-V objdump")
        if self.cc:
            candidate = Path(self.cc).with_name(
                Path(self.cc).name.replace("gcc", "objdump")
            )
            if candidate.is_file():
                return str(candidate)
        return _find_tool(
            None,
            (
                "riscv64-unknown-linux-gnu-objdump",
                "riscv64-linux-gnu-objdump",
                "llvm-objdump",
            ),
            "RISC-V objdump",
        )

    def qemu_binary(self) -> str:
        return _find_tool(
            self.qemu,
            ("qemu-riscv64", "qemu-riscv64-static"),
            "QEMU RISC-V user emulator",
        )


def link_elf(
    objects: Iterable[str | os.PathLike],
    runners: Iterable[str | os.PathLike],
    output: str | os.PathLike,
    *,
    toolchain: RiscvToolchain | None = None,
    static: bool = True,
    extra_args: Sequence[str] = (),
) -> Path:
    """Link Triton kernel objects and runner sources into a RISC-V ELF."""
    toolchain = toolchain or RiscvToolchain.from_env()
    object_paths = [Path(path) for path in objects]
    runner_paths = [Path(path) for path in runners]
    if not object_paths or not runner_paths:
        raise ValueError("Objects and runner sources are required")
    for path in object_paths + runner_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    cc = toolchain.linker()
    command = [cc]
    if "clang" in Path(cc).name:
        command.append(f"--target={toolchain.triple}")
    if toolchain.sysroot:
        command.append(f"--sysroot={toolchain.sysroot}")
    command.extend([f"-march={toolchain.march}", f"-mabi={toolchain.mabi}", "-O2"])
    if static:
        command.append("-static")
    command.extend(str(path) for path in runner_paths)
    command.extend(str(path) for path in object_paths)
    command.extend(extra_args)
    command.extend(["-o", str(output)])
    subprocess.run(command, check=True)
    return Path(output)


def run_qemu(
    elf: str | os.PathLike,
    args: Sequence[str] = (),
    *,
    toolchain: RiscvToolchain | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a RISC-V Linux ELF with qemu-riscv64."""
    toolchain = toolchain or RiscvToolchain.from_env()
    elf_path = Path(elf)
    if not elf_path.is_file():
        raise FileNotFoundError(elf_path)

    command = [toolchain.qemu_binary()]
    if toolchain.qemu_cpu:
        command.extend(["-cpu", toolchain.qemu_cpu])
    if toolchain.sysroot:
        command.extend(["-L", toolchain.sysroot])
    command.extend([str(elf_path), *args])
    return subprocess.run(command, check=check)


_C_TYPES = {
    "i1": "int32_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "uint32_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
}


def dump_assembly(
    binary: str | os.PathLike,
    output: str | os.PathLike,
    *,
    toolchain: RiscvToolchain | None = None,
) -> Path:
    """Disassemble a RISC-V object or ELF into a text file."""
    toolchain = toolchain or RiscvToolchain.from_env()
    result = subprocess.run(
        [toolchain.objdump_binary(), "-d", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout)
    return output_path


def _c_type(triton_type: str) -> str:
    dtype = triton_type[1:] if triton_type.startswith("*") else triton_type
    try:
        return _C_TYPES[dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported standalone argument type: {triton_type}"
        ) from exc


def _c_literal(value, triton_type: str) -> str:
    dtype = triton_type[1:] if triton_type.startswith("*") else triton_type
    if dtype in ("fp32", "f32"):
        value = float(value)
        if not (-float("inf") < value < float("inf")):
            raise ValueError("Non-finite standalone values are not supported")
        literal = f"{value:.9g}"
        if "." not in literal and "e" not in literal.lower():
            literal += ".0"
        return literal + "f"
    if dtype == "fp64":
        value = float(value)
        if not (-float("inf") < value < float("inf")):
            raise ValueError("Non-finite standalone values are not supported")
        literal = f"{value:.17g}"
        if "." not in literal and "e" not in literal.lower():
            literal += ".0"
        return literal
    return str(int(value))


def _generate_runner(
    kernel_name: str,
    signature: dict,
    arguments: dict,
    grid: Sequence[int],
    expected: dict | None = None,
    atol: float = 1e-5,
) -> str:
    """Generate a self-checking C runner for a compiled Triton kernel."""
    grid = tuple(int(value) for value in grid)
    if not 1 <= len(grid) <= 3 or any(value < 0 for value in grid):
        raise ValueError("grid must contain one to three non-negative dimensions")
    grid = grid + (1,) * (3 - len(grid))
    expected = expected or {}

    declarations = []
    extern_types = []
    call_args = []
    checks = []
    prints = []
    for index, (name, triton_type) in enumerate(signature.items()):
        if triton_type == "constexpr":
            continue
        if name not in arguments:
            raise ValueError(f"Missing standalone argument: {name}")

        if triton_type.startswith("*"):
            values = list(arguments[name])
            c_type = _c_type(triton_type)
            storage_size = max(1, len(values))
            initializer = ", ".join(_c_literal(value, triton_type) for value in values)
            if not initializer:
                initializer = "0"
            declarations.extend(
                [
                    f"  {c_type} arg_{index}[{storage_size}] = {{{initializer}}};",
                    f"  MemRef0 memref_{index} = {{arg_{index}, arg_{index}, 0}};",
                ]
            )
            extern_types.extend(["int64_t", "MemRef0 *"])
            call_args.extend(["0", f"&memref_{index}"])

            format_spec = "%g" if c_type in ("float", "double") else "%lld"
            cast = "(double)" if c_type in ("float", "double") else "(long long)"
            prints.extend(
                [
                    f'  printf("{name}=[");',
                    f"  for (int i = 0; i < {len(values)}; ++i) {{",
                    f'    printf(i ? ",{format_spec}" : "{format_spec}", {cast}arg_{index}[i]);',
                    "  }",
                    '  puts("]");',
                ]
            )

            if name in expected:
                expected_values = list(expected[name])
                if len(expected_values) != len(values):
                    raise ValueError(
                        f"Expected length for {name} does not match its buffer"
                    )
                expected_init = (
                    ", ".join(
                        _c_literal(value, triton_type) for value in expected_values
                    )
                    or "0"
                )
                declarations.append(
                    f"  const {c_type} expected_{index}[{storage_size}] = {{{expected_init}}};"
                )
                if c_type in ("float", "double"):
                    condition = (
                        f"double diff = (double)arg_{index}[i] - expected_{index}[i]; "
                        f"if (diff < 0) diff = -diff; if (diff > {atol:.17g})"
                    )
                else:
                    condition = f"if (arg_{index}[i] != expected_{index}[i])"
                checks.extend(
                    [
                        f"  for (int i = 0; i < {len(values)}; ++i) {{",
                        f"    {condition} {{",
                        f'      fprintf(stderr, "verification failed: {name}[%d]\\n", i);',
                        "      return 1;",
                        "    }",
                        "  }",
                    ]
                )
        else:
            c_type = _c_type(triton_type)
            extern_types.append(c_type)
            call_args.append(_c_literal(arguments[name], triton_type))

    extern_types.extend(["int32_t"] * 6)
    call_prefix = ", ".join(call_args)
    if call_prefix:
        call_prefix += ", "
    gx, gy, gz = grid
    return "\n".join(
        [
            "#include <stdint.h>",
            "#include <stdio.h>",
            "",
            "typedef struct { void *allocated; void *aligned; int64_t offset; } MemRef0;",
            f"extern void {kernel_name}({', '.join(extern_types)});",
            "",
            "int main(void) {",
            *declarations,
            f"  for (int x = 0; x < {gx}; ++x)",
            f"    for (int y = 0; y < {gy}; ++y)",
            f"      for (int z = 0; z < {gz}; ++z)",
            f"        {kernel_name}({call_prefix}{gx}, {gy}, {gz}, x, y, z);",
            *checks,
            *prints,
            '  puts("PASS");',
            "  return 0;",
            "}",
            "",
        ]
    )


def compile_to_elf(
    source,
    arguments: dict,
    grid: Sequence[int],
    output: str | os.PathLike,
    *,
    expected: dict | None = None,
    atol: float = 1e-5,
    toolchain: RiscvToolchain | None = None,
) -> Path:
    """Compile a Triton ASTSource and automatically build a runnable RISC-V ELF."""
    import tempfile

    import triton
    from triton.backends.compiler import GPUTarget

    toolchain = toolchain or RiscvToolchain.from_env()
    updates = {
        "TRITON_RISCV_CROSS_COMPILE": "1",
        "TRITON_RISCV_TARGET_TRIPLE": toolchain.triple,
        "TRITON_RISCV_LLC_FEATURES": toolchain.llc_features,
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        compiled = triton.compile(source, target=GPUTarget("cpu", 0, 0))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    runner = _generate_runner(
        compiled.metadata.name,
        source.signature,
        arguments,
        grid,
        expected=expected,
        atol=atol,
    )
    with tempfile.TemporaryDirectory(prefix="triton-riscv-elf-") as directory:
        directory = Path(directory)
        object_path = directory / "kernel.o"
        runner_path = directory / "runner.c"
        object_path.write_bytes(compiled.asm["obj"])
        runner_path.write_text(runner)
        return link_elf(
            [object_path], [runner_path], output, toolchain=toolchain, static=True
        )


def compile_and_run(
    source,
    arguments: dict,
    grid: Sequence[int],
    output: str | os.PathLike,
    *,
    expected: dict | None = None,
    atol: float = 1e-5,
    toolchain: RiscvToolchain | None = None,
) -> subprocess.CompletedProcess:
    """Compile a Triton kernel to ELF and immediately execute it in QEMU."""
    toolchain = toolchain or RiscvToolchain.from_env()
    elf = compile_to_elf(
        source,
        arguments,
        grid,
        output,
        expected=expected,
        atol=atol,
        toolchain=toolchain,
    )
    return run_qemu(elf, toolchain=toolchain)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    link_parser = commands.add_parser("link", help="link objects and a runner")
    link_parser.add_argument("-o", "--output", required=True)
    link_parser.add_argument("--object", action="append", required=True)
    link_parser.add_argument("--runner", action="append", required=True)
    link_parser.add_argument("--dynamic", action="store_true")
    link_parser.add_argument("--link-arg", action="append", default=[])

    run_parser = commands.add_parser("run", help="execute an ELF with QEMU")
    run_parser.add_argument("elf")
    run_parser.add_argument("args", nargs=argparse.REMAINDER)

    for subparser in (link_parser, run_parser):
        for name in ("triple", "march", "mabi", "sysroot"):
            subparser.add_argument(f"--{name}")
    link_parser.add_argument("--cc")
    run_parser.add_argument("--qemu")
    run_parser.add_argument("--qemu-cpu")

    parsed = parser.parse_args(argv)
    names = ("triple", "march", "mabi", "cc", "sysroot", "qemu", "qemu_cpu")
    overrides = {
        name: getattr(parsed, name)
        for name in names
        if hasattr(parsed, name) and getattr(parsed, name) is not None
    }
    toolchain = replace(RiscvToolchain.from_env(), **overrides)
    if parsed.command == "link":
        link_elf(
            parsed.object,
            parsed.runner,
            parsed.output,
            toolchain=toolchain,
            static=not parsed.dynamic,
            extra_args=parsed.link_arg,
        )
    else:
        run_qemu(parsed.elf, parsed.args, toolchain=toolchain)
    return 0


def _infer_triton_type(value) -> str:
    """Infer a standalone Triton type from a Python scalar or buffer."""
    is_buffer = not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
    dtype_name = str(getattr(value, "dtype", "")).removeprefix("torch.")
    dtype_map = {
        "bool": "i1",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
        "float32": "fp32",
        "float64": "fp64",
    }
    scalar_type = dtype_map.get(dtype_name)

    if is_buffer:
        values = list(value)
        if scalar_type is None:
            if not values:
                raise ValueError(
                    "Cannot infer the type of an empty buffer; provide signature=..."
                )
            scalar_type = _infer_triton_type(values[0]).removeprefix("*")
        return f"*{scalar_type}"

    if scalar_type is not None:
        return scalar_type
    if isinstance(value, bool):
        return "i1"
    if isinstance(value, float):
        return "fp32"
    if isinstance(value, int):
        return "i32" if -(2**31) <= value < 2**31 else "i64"
    raise ValueError(
        f"Cannot infer a Triton type for {type(value).__name__}; provide signature=..."
    )


def make_standalone_source(
    kernel,
    arguments: dict,
    *,
    constexprs: dict | None = None,
    signature: dict | None = None,
):
    """Create ASTSource for a JIT kernel, inferring common argument types."""
    import triton

    constexprs = constexprs or {}
    overrides = signature or {}
    inferred_signature = {}
    for name in kernel.arg_names:
        if name in constexprs:
            inferred_signature[name] = "constexpr"
        elif name in overrides:
            inferred_signature[name] = overrides[name]
        elif name in arguments:
            inferred_signature[name] = _infer_triton_type(arguments[name])
        else:
            raise ValueError(f"Missing standalone argument or constexpr: {name}")
    return triton.compiler.ASTSource(
        fn=kernel,
        signature=inferred_signature,
        constexprs=constexprs,
    )


def compile_kernel_to_elf(
    kernel,
    arguments: dict,
    grid: Sequence[int],
    output: str | os.PathLike,
    *,
    constexprs: dict | None = None,
    signature: dict | None = None,
    expected: dict | None = None,
    atol: float = 1e-5,
    toolchain: RiscvToolchain | None = None,
) -> Path:
    """Compile a @triton.jit function directly into a runnable RISC-V ELF."""
    source = make_standalone_source(
        kernel,
        arguments,
        constexprs=constexprs,
        signature=signature,
    )
    return compile_to_elf(
        source,
        arguments,
        grid,
        output,
        expected=expected,
        atol=atol,
        toolchain=toolchain,
    )


def compile_and_run_kernel(
    kernel,
    arguments: dict,
    grid: Sequence[int],
    output: str | os.PathLike,
    *,
    constexprs: dict | None = None,
    signature: dict | None = None,
    expected: dict | None = None,
    atol: float = 1e-5,
    toolchain: RiscvToolchain | None = None,
) -> subprocess.CompletedProcess:
    """Compile a @triton.jit function to ELF and execute it with QEMU."""
    toolchain = toolchain or RiscvToolchain.from_env()
    elf = compile_kernel_to_elf(
        kernel,
        arguments,
        grid,
        output,
        constexprs=constexprs,
        signature=signature,
        expected=expected,
        atol=atol,
        toolchain=toolchain,
    )
    return run_qemu(elf, toolchain=toolchain)


def standalone_kernel_cli(
    kernel,
    arguments: dict,
    grid: Sequence[int],
    *,
    default_output: str | os.PathLike,
    constexprs: dict | None = None,
    signature: dict | None = None,
    expected: dict | None = None,
    atol: float = 1e-5,
    argv: Sequence[str] | None = None,
) -> Path:
    """Provide the standard compile/run/dump CLI for one standalone kernel case."""
    parser = argparse.ArgumentParser(
        description=f"Build {kernel.__name__} as RVV ELF and execute it with QEMU"
    )
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-asm", action="store_true")
    parser.add_argument("--asm-output", type=Path)
    parsed = parser.parse_args(argv)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)

    compile_fn = (
        compile_kernel_to_elf if parsed.compile_only else compile_and_run_kernel
    )
    compile_fn(
        kernel,
        arguments,
        grid,
        parsed.output,
        constexprs=constexprs,
        signature=signature,
        expected=expected,
        atol=atol,
    )
    print(f"ELF: {parsed.output}")

    if parsed.dump_asm:
        asm_output = parsed.asm_output or parsed.output.with_suffix(".s")
        dump_assembly(parsed.output, asm_output)
        print(f"Assembly: {asm_output}")
    return parsed.output


if __name__ == "__main__":
    raise SystemExit(main())

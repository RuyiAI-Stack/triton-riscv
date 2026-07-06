from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import platform
import re
import shutil
import subprocess
import functools
import triton
from pathlib import Path

from .paths import _get_buddy_opt_path, _get_llvm_bin_path, _get_triton_shared_opt_path
from .riscv import DEFAULT_LLC_FEATURES, RiscvToolchain


def _get_buddy_translate_path() -> str:
    """Path to buddy-translate (from buddy-mlir build)."""
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path == "":
        raise Exception("BUDDY_MLIR_BINARY_DIR is not set.")
    return os.path.join(path, "buddy-translate")


def _get_buddy_llc_path() -> str:
    """Path to buddy-llc (from buddy-mlir build)."""
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path == "":
        raise Exception("BUDDY_MLIR_BINARY_DIR is not set.")
    return os.path.join(path, "buddy-llc")


def _use_ime_pipeline() -> bool:
    """Return True when the IME (RISC-V matrix-extension) lowering path should be used.

    Set the environment variable TRITON_RISCV_USE_IME=1 to activate.  The IME
    pipeline lowers linalg.matmul to ime.vfmadot (fp16) / ime.vmadot (int)
    instructions via buddy-mlir and cross-compiles to a RISC-V ELF object.
    """
    return os.getenv("TRITON_RISCV_USE_IME", "") == "1"


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    for f in files:
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")

    return sanitizer_type


def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        _dump_ir_if_needed([src_path])
        triton_shared_opt_path = _get_triton_shared_opt_path()

        subprocess_args = [
            triton_shared_opt_path,
            src_path,
            "--triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu",
            "--mlir-print-debuginfo",
            "-o",
            dst_path,
        ]

        if _get_sanitizer_type() != "":
            print("Building with sanitizer support...")

            # has to run before the other passes as operates on the tt dialect
            subprocess_args.insert(2, "--add-llvm-debug-info")

        subprocess.check_call(subprocess_args)
        _dump_ir_if_needed([dst_path])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        buddy_opt_path = _get_buddy_opt_path()

        if _use_ime_pipeline():
            # ---------------------------------------------------------------
            # IME path: Triton-Shared MLIR → buddy-mlir IME dialect → LLVM IR
            # Lowers linalg.matmul (f16) to ime.vfmadot via buddy-mlir passes,
            # then cross-compiles to a RISC-V ELF object with +xsmtime.
            # ---------------------------------------------------------------
            subprocess.check_call(
                [
                    buddy_opt_path,
                    ttshared_path,
                    # Bufferize tensor ops before IME lowering
                    "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                    # One-shot bufferization creates heap-backed temporary
                    # memrefs. Insert and lower their deallocations while the
                    # memref-level ownership information is still available.
                    "--buffer-deallocation-pipeline",
                    # Lower linalg.matmul (memref) → ime.vfmadot / ime.vmadot
                    "--lower-linalg-to-ime",
                    # Lower IME dialect → vector / loop ops
                    "--lower-ime",
                    # Lower any remaining linalg / affine / SCF ops
                    "--convert-linalg-to-loops",
                    "--lower-affine",
                    "--expand-strided-metadata",
                    "--convert-scf-to-cf",
                    # Lower to LLVM dialect
                    "--convert-cf-to-llvm",
                    "--convert-arith-to-llvm",
                    "--convert-math-to-llvm",
                    "--convert-complex-to-llvm",
                    "--convert-vector-to-llvm",
                    "--convert-index-to-llvm",
                    "--convert-func-to-llvm",
                    "--memref-expand",
                    "--finalize-memref-to-llvm",
                    # Lowering memrefs creates more affine.apply ops; run again.
                    "--lower-affine",
                    "--convert-arith-to-llvm",
                    "--reconcile-unrealized-casts",
                    "--mlir-print-debuginfo",
                    "-o",
                    llmlir_path,
                ]
            )
            # LLVM-MLIR → LLVM-IR via buddy-translate (handles the IME output)
            buddy_translate_path = _get_buddy_translate_path()
            subprocess.check_call(
                [
                    buddy_translate_path,
                    "--buddy-to-llvmir",
                    llmlir_path,
                    "-o",
                    llir_path,
                ]
            )
        else:
            # ---------------------------------------------------------------
            # Standard path: buddy-mlir vectorisation → host LLVM IR
            # ---------------------------------------------------------------
            subprocess.check_call(
                [
                    buddy_opt_path,
                    ttshared_path,
                    "--convert-linalg-to-affine-loops",
                    # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                    # in a single kernel which are the results of early returns.
                    # See python/examples/test_early_return.py for examples.
                    # We disable this pass for now since performance on CPU isn't the main
                    # focus at the moment.
                    # "--eliminate-empty-tensors",
                    "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                    "--buffer-deallocation-pipeline",
                    "--matmul-vectorization",
                    "--lower-affine",
                    "--convert-linalg-to-loops",
                    "--expand-strided-metadata",
                    "--convert-scf-to-cf",
                    "--convert-arith-to-llvm",
                    "--convert-math-to-llvm",
                    "--convert-complex-to-llvm",
                    "--convert-vector-to-llvm",
                    "--convert-index-to-llvm",
                    "--memref-expand",
                    "--finalize-memref-to-llvm",
                    "--convert-func-to-llvm",
                    "--convert-cf-to-llvm",
                    # Lowering memrefs creates more affine.apply ops.
                    # Lowering these affine ops again creates further arith ops,
                    # so we have to run these two passes again here.
                    "--lower-affine",
                    "--convert-arith-to-llvm",
                    # Remove all unrealized casts created
                    "--reconcile-unrealized-casts",
                    "--mlir-print-debuginfo",
                    "-o",
                    llmlir_path,
                ]
            )
            # LLVM-MLIR to LLVM-IR
            mlir_translate_path = _get_llvm_bin_path("mlir-translate")
            subprocess.check_call(
                [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
            )

        _dump_ir_if_needed([ttshared_path, llmlir_path, llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     llir_path = os.path.join(tmpdir, "ll.ir")
    #     lliropt_path = os.path.join(tmpdir, "llopt.ir")
    #     Path(llir_path).write_text(llir)

    #     opt_path = _get_llvm_bin_path("opt")
    #     subprocess.check_call([opt_path, llir_path, "-O2", "-S", "-o", lliropt_path])

    #     _dump_ir_if_needed([lliropt_path])
    #     return Path(lliropt_path).read_text()
    return llir


def _ttsharedir_to_vectorir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        vector_path = os.path.join(tmpdir, "vector.mlir")
        Path(ttshared_path).write_text(ttsharedir)
        buddy_opt_path = _get_buddy_opt_path()
        subprocess.check_call(
            [
                buddy_opt_path,
                ttshared_path,
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--buffer-deallocation-pipeline",
                "--lower-linalg-to-vir",
                "--lower-vir-to-vector=vector-width=16",
                "--cse",
                "--mlir-print-debuginfo",
                "-o",
                vector_path,
            ]
        )
        _dump_ir_if_needed([vector_path])
        return Path(vector_path).read_text()


def _vectorir_to_llir(vectorir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_path = os.path.join(tmpdir, "vector.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(vector_path).write_text(vectorir)
        buddy_opt_path = _get_buddy_opt_path()
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                buddy_opt_path,
                vector_path,
                "--convert-linalg-to-affine-loops",
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                # "--empty-tensor-to-alloc-tensor",
                # "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                # "--matmul-vectorization",
                "--expand-strided-metadata",
                "--lower-affine",
                "--convert-math-to-llvm",
                "--convert-math-to-libm",
                "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                "--convert-vector-to-scf",
                "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                "--convert-ub-to-llvm",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
                "--reconcile-unrealized-casts",
                "--mlir-print-debuginfo",
                "-o",
                llmlir_path,
            ]
        )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        _dump_ir_if_needed([llmlir_path, llir_path])
        return Path(llir_path).read_text()


_LLVM_SYMBOL = r'(?:"([^"]+)"|([A-Za-z$._][A-Za-z0-9$._-]*))'


def _find_kernel_name(llir: str) -> str:
    """Return the single externally callable function defined in LLVM IR.

    Buffer deallocation may add helper definitions such as ``dealloc_helper``
    to the module.  Helpers are called by the kernel, whereas the kernel is the
    sole definition that is not called from within the module.
    """
    definitions = {
        quoted or unquoted
        for quoted, unquoted in re.findall(
            rf"^define\b[^@\n]*@{_LLVM_SYMBOL}\s*\(", llir, re.MULTILINE
        )
    }
    callees = {
        quoted or unquoted
        for quoted, unquoted in re.findall(
            rf"\b(?:call|invoke)\b[^@\n]*@{_LLVM_SYMBOL}\s*\(", llir
        )
    }
    candidates = sorted(definitions - callees)
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one externally callable kernel definition, "
            f"found {candidates}; all definitions: {sorted(definitions)}"
        )
    return candidates[0]


def _llir_to_bin(llir: str, metadata, options=None):
    metadata["name"] = _find_kernel_name(llir)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)

        sanitizer_type = _get_sanitizer_type()

        if sanitizer_type != "":
            # using a sanitizer
            # invoke pass to append sanitizer attributes
            instrumented_src_path = os.path.join(tmpdir, "kernel-instrumented.ll")

            opt_path = _get_llvm_bin_path("opt")
            top_level_triton_path = os.path.dirname(triton.__file__)
            sanitizer_attributes_pass_path = str(
                next(
                    Path(top_level_triton_path).rglob("libSanitizerAttributes.so"), None
                )
            )

            if not sanitizer_attributes_pass_path:
                raise Exception("libSanitizerAttributes.so does not exist.")

            subprocess.check_call(
                [
                    opt_path,
                    "-load-pass-plugin",
                    sanitizer_attributes_pass_path,
                    "-passes=sanitizer-attributes",
                    f"-sanitizer-type={sanitizer_type}",
                    "-S",
                    src_path,
                    "-o",
                    instrumented_src_path,
                ]
            )

            # compile to object file
            clang_path = _get_llvm_bin_path("clang++")

            subprocess_args = [clang_path, "-c", instrumented_src_path, "-o", dst_path]

            if sanitizer_type == "asan":
                subprocess_args.extend(
                    ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
                )
            elif sanitizer_type == "tsan":
                subprocess_args.extend(["-g", "-fsanitize=thread"])

            subprocess.check_call(subprocess_args)
        elif _use_ime_pipeline():
            # IME path: cross-compile to RISC-V with xsmtime (vfmadot / vmadot).
            # buddy-llc understands the RISC-V IME intrinsics produced by
            # buddy-translate and generates correct machine code.
            buddy_llc_path = _get_buddy_llc_path()
            toolchain = RiscvToolchain.from_env()
            toolchain = replace(
                toolchain, llc_features=f"{toolchain.llc_features},+xsmtime"
            )
            subprocess.check_call(
                toolchain.llc_command(buddy_llc_path, src_path, dst_path)
            )
        else:
            llc_path = _get_llvm_bin_path("llc")
            llc_args = [
                llc_path,
                src_path,
                "-filetype=obj",
                "-relocation-model=pic",
                "-o",
                dst_path,
            ]
            # On RISC-V Linux, the system ABI is lp64d (hardware double-precision float).
            # Without explicitly enabling +f,+d, llc defaults to soft-float, causing
            # "can't link soft-float modules with double-float modules" linker errors.
            target_triple = getattr(options, "target_triple", None)
            if target_triple:
                toolchain = replace(
                    RiscvToolchain.from_env(),
                    triple=target_triple,
                    llc_features=options.target_features,
                )
                llc_args = toolchain.llc_command(llc_path, src_path, dst_path)
            elif platform.machine() == "riscv64":
                llc_args.extend([f"-mattr={DEFAULT_LLC_FEATURES}"])
            subprocess.check_call(llc_args)

        return Path(dst_path).read_bytes()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    target_triple: str = None
    target_features: str = None

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = "obj"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        )
        if (
            os.getenv("TRITON_RISCV_CROSS_COMPILE", "") == "1"
            or _use_ime_pipeline()
            or args.get("target_triple")
        ):
            toolchain = RiscvToolchain.from_env()
            args.setdefault("target_triple", toolchain.triple)
            args.setdefault("target_features", toolchain.llc_features)
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttir")
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(
            _ttir_to_ttsharedir(src)
        )
        if _use_ime_pipeline():
            stages["llir"] = lambda src, metadata: _optimize_llir(
                _ttsharedir_to_llir(src)
            )
        else:
            stages["vectorir"] = lambda src, metadata: _ttsharedir_to_vectorir(src)
            stages["llir"] = lambda src, metadata: _optimize_llir(
                _vectorir_to_llir(src)
            )
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

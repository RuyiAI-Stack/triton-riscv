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


def _get_openmp_num_threads() -> int:
    value = os.getenv("TRITON_RISCV_OPENMP_THREADS", "0")
    if value == "":
        return 0
    try:
        threads = int(value)
    except ValueError as exc:
        raise ValueError(
            "TRITON_RISCV_OPENMP_THREADS must be an integer thread count"
        ) from exc
    if threads < 0:
        raise ValueError("TRITON_RISCV_OPENMP_THREADS must be non-negative")
    return threads


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

        linalg_experimental_opts = (
            "structured-ldst-mode=tensor-first-vector-cpu"
        )
        if platform.machine() == "riscv64":
            # StructuredToMemref tensor-first-vector-cpu emits fixed-width
            # vector<W x T> lowered to LLVM <W x T> (default W=16 for AVX-512).
            # riscv64 llc miscompiles these ops (heap corruption on masked 1D
            # stores in fill_neg_inf / fp8_paged_mqa_logits). W=1 keeps scalar
            # loops; RVV matmul stays on buddy --matmul-vectorization.
            linalg_experimental_opts += " cpu-vector-width=1"

        subprocess_args = [
            triton_shared_opt_path,
            src_path,
            f"--triton-to-linalg-experimental={linalg_experimental_opts}",
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


def _targets_riscv(options=None) -> bool:
    if platform.machine() == "riscv64":
        return True
    triple = getattr(options, "target_triple", None) if options else None
    return bool(triple and "riscv" in triple)


def _cross_compiles_riscv(options=None) -> bool:
    """True when the compile request explicitly targets RISC-V (cross-compile).

    VIR elementwise vectorization is only needed for this case: the RVV object
    test passes target_triple, and x86 hosts cross-compiling disable LLVM's loop
    vectorizer.  Applying VIR on native riscv64 execution miscompiles some tile
    kernels (euclidean_dist: corrupted size vs. prev_size at launch).
    """
    triple = getattr(options, "target_triple", None) if options else None
    return bool(triple and "riscv" in triple)


def _riscv_vir_vector_passes() -> list[str]:
    # VLEN=128 → 4 x f32. Lower elementwise linalg to explicit vector ops so
    # llc emits RVV vfsub/vfmul (FlagGems euclidean_dist RVV object test).
    return [
        "--lower-linalg-to-vir",
        "--lower-vir-to-vector=vector-width=4",
    ]


def _ttsharedir_to_llir(ttsharedir: str, options=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        ime_pre_llvm_path = os.path.join(tmpdir, "ime-pre-llvm.mlir")
        standard_pre_llvm_path = os.path.join(tmpdir, "pre-llvm.mlir")
        pre_llvm_transformed_path = os.path.join(tmpdir, "pre-llvm-transformed.mlir")
        atomic_cas_path = os.path.join(tmpdir, "ttshared-atomic-cas.mlir")
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
            ime_lowering_passes = [
                # Bufferize tensor ops before IME lowering
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                # Hoist loop/allocation pairs before deallocation insertion.
                # CAS while-loops (scatter_reduce) otherwise get per-iteration
                # heap alloc/free that corrupts the heap on riscv64.
                "--buffer-loop-hoisting",
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
            ]
            llvm_lowering_passes = [
                # Lower to LLVM dialect
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-math-to-libm",
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
            ]

            buddy_input_path = ttshared_path
            buddy_passes = [*ime_lowering_passes, *llvm_lowering_passes]
            if "__triton_shared_atomic_cas_" in ttsharedir:
                # LLVM cmpxchg is not bufferizable, so lower helper calls only
                # after Buddy has completed its tensor-to-memref work.
                subprocess.check_call(
                    [
                        buddy_opt_path,
                        ttshared_path,
                        *ime_lowering_passes,
                        "--mlir-print-debuginfo",
                        "-o",
                        ime_pre_llvm_path,
                    ]
                )
                subprocess.check_call(
                    [
                        _get_triton_shared_opt_path(),
                        ime_pre_llvm_path,
                        "--lower-atomic-cas-to-llvm",
                        "--canonicalize",
                        "--mlir-print-debuginfo",
                        "-o",
                        atomic_cas_path,
                    ]
                )
                buddy_input_path = atomic_cas_path
                buddy_passes = llvm_lowering_passes

            subprocess.check_call(
                [
                    buddy_opt_path,
                    buddy_input_path,
                    *buddy_passes,
                    "--mlir-print-debuginfo",
                    "-o",
                    llmlir_path,
                ]
            )
            # LLVM-MLIR → LLVM-IR via buddy-translate (handles buddyext dialect)
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
            # Split bufferize/loop lowering from LLVM conversion so triton-shared
            # can expand FP8 arith.extf/truncf (and lower atomic CAS helpers)
            # before convert-arith-to-llvm. Without this, f32→f8E4M3FN truncf
            # survives into mlir-translate, which does not register the arith
            # dialect (FlagGems test_act_quant / fp8 kernels).
            standard_lowering_passes = [
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
            ]
            # Fuse elementwise chains before bufferization for vanilla kernels.
            # On atomic-CAS while loops (scatter_reduce prod/amax/amin) this pass
            # makes scf.while carry memref values with per-iteration alloc/copy/
            # dealloc; convert-scf-to-cf then double-frees the tile buffer.
            if "__triton_shared_atomic_cas_" not in ttsharedir:
                standard_lowering_passes.append("--linalg-fuse-elementwise-ops")
            standard_lowering_passes.extend(
                [
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                # Before buffer-deallocation: reuse CAS/while tile buffers across
                # iterations instead of heap alloc+free inside scf.while.
                "--buffer-loop-hoisting",
                "--buffer-deallocation-pipeline",
                "--eliminate-memref-copy",
                # Triton programs commonly materialize small, statically
                # sized tiles for loads and scalar broadcasts.  Paying for
                # several heap allocations on every grid invocation
                # dominates elementwise CPU kernels, so keep bounded tile
                # temporaries in the launcher thread's stack frame.
                "--promote-buffers-to-stack=max-alloc-size-in-bytes=65536",
                ]
            )
            # VLEN=128 → 4 x f32 per vector register. Keep matmul-vectorization
            # (RVV path) but size vectors to the host VLEN; default vector-size=32
            # is an AVX-512-oriented width and over-commits RVV LMUL.
            if _targets_riscv(options):
                standard_lowering_passes.append(
                    "--matmul-vectorization=vector-size=4"
                )
            else:
                standard_lowering_passes.append("--matmul-vectorization")
            if _cross_compiles_riscv(options):
                standard_lowering_passes.extend(_riscv_vir_vector_passes())
            standard_lowering_passes.extend(
                [
                "--convert-linalg-to-affine-loops",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--expand-strided-metadata",
                "--convert-scf-to-cf",
                ]
            )
            llvm_lowering_passes = [
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-math-to-libm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-cf-to-llvm",
                "--convert-func-to-llvm",
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
                "--reconcile-unrealized-casts",
            ]

            subprocess.check_call(
                [
                    buddy_opt_path,
                    ttshared_path,
                    *standard_lowering_passes,
                    "--mlir-print-debuginfo",
                    "-o",
                    standard_pre_llvm_path,
                ]
            )
            # Match the vector path: expand FP8 before LLVM dialect conversion.
            pre_llvm_transform_passes = ["--expand-float8-conversions"]
            if "__triton_shared_atomic_cas_" in ttsharedir:
                # LLVM cmpxchg is not bufferizable, so lower helper calls only
                # after Buddy has completed its tensor-to-memref work.
                pre_llvm_transform_passes.insert(0, "--lower-atomic-cas-to-llvm")
            subprocess.check_call(
                [
                    _get_triton_shared_opt_path(),
                    standard_pre_llvm_path,
                    *pre_llvm_transform_passes,
                    "--canonicalize",
                    "--mlir-print-debuginfo",
                    "-o",
                    pre_llvm_transformed_path,
                ]
            )
            subprocess.check_call(
                [
                    buddy_opt_path,
                    pre_llvm_transformed_path,
                    *llvm_lowering_passes,
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


_FADD_INSTRUCTION = re.compile(
    r"^(?P<prefix>\s*%[-\w.$\"]+\s*=\s*fadd)[ \t]+"
    r"(?P<flags>(?:(?:fast|reassoc|nnan|ninf|nsz|arcp|contract|afn)[ \t]+)*)"
    r"(?P<operands>\S.*)$",
    re.MULTILINE,
)

_HEAP_MALLOC = re.compile(
    r"^\s*(?:tail )?call (?:noalias noundef )?ptr @malloc\b", re.MULTILINE
)
_HEAP_FREE = re.compile(
    r"^\s*(?:tail )?call void @free\(ptr [^)]+\)\s*$", re.MULTILINE
)


def _strip_orphan_heap_frees(llir: str) -> str:
    """Remove heap @free calls when LLVM promoted all @malloc to stack.

    riscv64 opt -O1+ with -mtriple=riscv64 leaves orphan frees after promotion,
    which corrupts the heap (scatter_reduce CAS loops).
    """
    if _HEAP_MALLOC.search(llir):
        return llir
    return _HEAP_FREE.sub("", llir)


def _enable_fadd_reassociation(llir: str) -> str:
    """Add reassoc only to complete fadd instructions that do not have it."""

    def add_flag(match):
        flags = match.group("flags").split()
        if "fast" in flags or "reassoc" in flags:
            return match.group(0)
        flags.insert(0, "reassoc")
        return f"{match.group('prefix')} {' '.join(flags)} {match.group('operands')}"

    return _FADD_INSTRUCTION.sub(add_flag, llir)


_ATOMIC_LLIR_MARKERS = (
    "atomicrmw",
    "cmpxchg",
    "__triton_shared_atomic_cas_",
)

_LIBM_LLIR_MARKERS = (
    "@erff",
    "@erf",
    "@sinhf",
    "@sinh",
    "@expf",
    "@exp",
    "@logf",
    "@log",
    "@tanhf",
    "@tanh",
)

# convert-math-to-llvm lowers tl.exp/log/... before libm sees them.
_LLVM_MATH_INTRINSIC_MARKERS = (
    "@llvm.exp.",
    "@llvm.log.",
    "@llvm.log10.",
    "@llvm.log2.",
    "@llvm.pow.",
    "@llvm.sin.",
    "@llvm.cos.",
    "@llvm.tan.",
    "@llvm.sqrt.",
    "@llvm.fabs.",
    "@llvm.erf.",
)

_DOT_KERNEL_NAME_MARKERS = (
    "matmul",
    "gemm",
    "dot",
    "logits",
    "_mm",
)

# Elementwise norms/reductions can contain many fmul/fadd without being matmul.
_DOT_REDUCTION_EXCLUDED_NAME_MARKERS = (
    "euclidean",
    "fill",
    "neg_inf",
)


def _llir_contains_libm_calls(llir: str) -> bool:
    return any(marker in llir for marker in _LIBM_LLIR_MARKERS)


def _llir_contains_llvm_math_intrinsics(llir: str) -> bool:
    return any(marker in llir for marker in _LLVM_MATH_INTRINSIC_MARKERS)


def _llir_contains_llvm_vector_dot_ops(llir: str) -> bool:
    return "@llvm.fmuladd.v" in llir


def _llir_contains_dot_reduction(llir: str) -> bool:
    name_match = re.search(r"define void @(\w+)", llir)
    if name_match:
        name = name_match.group(1).lower()
        if any(marker in name for marker in _DOT_REDUCTION_EXCLUDED_NAME_MARKERS):
            return False
        if any(marker in name for marker in _DOT_KERNEL_NAME_MARKERS):
            return True
    fmul_count = len(re.findall(r"=\s*fmul float", llir))
    fadd_count = len(re.findall(r"=\s*fadd float", llir))
    return fmul_count >= 4 and fadd_count >= 4


def _llir_is_simple_store_only_kernel(llir: str) -> bool:
    name_match = re.search(r"define void @(\w+)", llir)
    if name_match:
        name = name_match.group(1).lower()
        if "fill" in name or "neg_inf" in name:
            return True

    fmul_count = len(re.findall(r"=\s*fmul float", llir))
    fadd_count = len(re.findall(r"=\s*fadd float", llir))
    fcmp_count = len(re.findall(r"=\s*fcmp ", llir))
    call_count = len(re.findall(r"=\s*call ", llir))
    store_count = len(re.findall(r"=\s*store ", llir))

    return (
        store_count >= 1
        and fmul_count == 0
        and fadd_count == 0
        and fcmp_count == 0
        and call_count == 0
    )


def _llir_contains_atomics(llir: str) -> bool:
    return any(marker in llir for marker in _ATOMIC_LLIR_MARKERS)


def _llir_needs_scalar_riscv_codegen(llir: str, *, for_cross_compile: bool = False) -> bool:
    """Return True when riscv64 codegen must avoid LLVM loop vectorization."""
    if _llir_contains_atomics(llir) or _llir_contains_dot_reduction(llir):
        return True
    if for_cross_compile:
        return False
    return (
        _llir_contains_libm_calls(llir)
        or _llir_contains_llvm_math_intrinsics(llir)
        or _llir_contains_llvm_vector_dot_ops(llir)
        or _llir_is_simple_store_only_kernel(llir)
    )


def _optimize_llir(llir: str, options=None):
    # llc's -O3 controls code-generation optimizations but does not run the
    # target-aware LLVM middle-end pipeline.  In particular, scalar linalg
    # loops emitted for elementwise Triton tiles remain scalar unless opt sees
    # the host vector width.  Run the native O3 pipeline for host x86 builds;
    # for RISC-V (native or cross-compile via target_triple) run the RVV-safe
    # opt pipeline so explicit <N x float> ops reach llc intact.
    host_machine = platform.machine()
    target_triple = getattr(options, "target_triple", None)
    target_features = getattr(options, "target_features", None)
    cross_compile_riscv = bool(target_triple and "riscv" in target_triple)
    riscv_opt = host_machine == "riscv64" or cross_compile_riscv

    if (host_machine == "riscv64" or cross_compile_riscv) and _llir_needs_scalar_riscv_codegen(
        llir, for_cross_compile=cross_compile_riscv
    ):
        return llir

    if host_machine not in {"x86_64", "AMD64", "riscv64"} and not riscv_opt:
        return llir
    if target_triple is not None and not riscv_opt:
        return llir

    if getattr(options, "allow_fp_reassoc", False) or cross_compile_riscv:
        # Cross-compiled RVV objects need reassociation so reduction chains can
        # lower to vfadd/vfmul; native opt-in covers other reduction kernels.
        llir = _enable_fadd_reassociation(llir)

    # The launcher constructs unranked-memref descriptors on its stack and
    # keeps them immutable for the complete kernel call. MLIR lowers access to
    # the descriptor's aligned data pointer as a load through `arg + 8`, but
    # LLVM cannot otherwise prove that stores through the loaded data pointer
    # do not modify the descriptor itself. Mark just those descriptor-field
    # loads invariant so LICM can hoist them and the loop vectorizer can see a
    # normal contiguous load/store loop.
    define = re.search(r"^define\b[^\n]*\((.*)\)[^{]*\{", llir, re.MULTILINE)
    if define:
        pointer_args = set(
            re.findall(r"(?:^|,\s*)ptr(?:\s+[^,%]+)*\s+(%[-\w.]+)", define.group(1))
        )
        descriptor_fields = set()
        for match in re.finditer(
            r"^\s*(%[-\w.]+) = getelementptr(?: inbounds)? i8, "
            r"ptr (%[-\w.]+), i64 8\s*$",
            llir,
            re.MULTILINE,
        ):
            if match.group(2) in pointer_args:
                descriptor_fields.add(match.group(1))
        # Before instcombine, the same second descriptor field is expressed
        # as a one-element GEP on `ptr`. A constant index of one is specific to
        # the two-pointer memref descriptor prefix; user data indexing remains
        # variable or is typed by the pointee element.
        for match in re.finditer(
            r"^\s*(%[-\w.]+) = getelementptr(?: inbounds)? ptr, "
            r"ptr (%[-\w.]+), i(?:32|64) 1\s*$",
            llir,
            re.MULTILINE,
        ):
            # Do not classify an arbitrary one-element user-data GEP as a
            # descriptor field. It must be derived directly from a pointer
            # argument of the generated kernel ABI.
            if match.group(2) in pointer_args:
                descriptor_fields.add(match.group(1))
        if descriptor_fields:
            metadata_ids = [
                int(value) for value in re.findall(r"^!(\d+) =", llir, re.MULTILINE)
            ]
            invariant_id = max(metadata_ids, default=-1) + 1
            fields = "|".join(re.escape(value) for value in descriptor_fields)
            load_pattern = re.compile(
                rf"^(\s*%[-\w.]+ = load ptr, ptr (?:{fields})[^\n]*)(?<!\binvariant\.load !\d)$",
                re.MULTILINE,
            )
            llir, replacements = load_pattern.subn(
                rf"\1, !invariant.load !{invariant_id}", llir
            )
            if replacements:
                llir += f"\n!{invariant_id} = !{{}}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel-opt.ll")
        Path(src_path).write_text(llir)
        command = [_get_llvm_bin_path("opt"), "-O3"]
        if cross_compile_riscv:
            command.extend(
                [
                    f"-mtriple={target_triple}",
                    f"-mattr={target_features or DEFAULT_LLC_FEATURES}",
                    "-riscv-v-vector-bits-min=128",
                    "-riscv-v-vector-bits-max=128",
                    "--unroll-threshold=0",
                    "-vectorize-loops=false",
                    "-vectorize-slp=false",
                ]
            )
        elif host_machine in {"x86_64", "AMD64"}:
            command.extend(
                [
                    "-mtriple=x86_64-unknown-linux-gnu",
                    "-mcpu=native",
                    "-vector-library=LIBMVEC",
                ]
            )
        else:
            # Native riscv64: historical opt pipeline (act_quant FP8 quant relies
            # on this).  Cross-compile uses the guarded branch above; disabling
            # the loop vectorizer here miscompiles FP8 bitcast/shift chains.
            command.extend(
                [
                    "-mtriple=riscv64-unknown-linux-gnu",
                    f"-mattr={DEFAULT_LLC_FEATURES}",
                    "-riscv-v-vector-bits-min=128",
                    "-riscv-v-vector-bits-max=128",
                ]
            )
        command.extend(["-S", src_path, "-o", dst_path])
        subprocess.check_call(command)
        optimized = Path(dst_path).read_text()
        if riscv_opt and _llir_contains_atomics(optimized):
            optimized = _strip_orphan_heap_frees(optimized)
            Path(dst_path).write_text(optimized)
        _dump_ir_if_needed([dst_path])
        return optimized


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
                "--buffer-loop-hoisting",
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
        transformed_vector_path = os.path.join(tmpdir, "vector-transformed.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(vector_path).write_text(vectorir)
        transform_passes = []
        if "__triton_shared_atomic_cas_" in vectorir:
            transform_passes.append("--lower-atomic-cas-to-llvm")
        transform_passes.append("--expand-float8-conversions")
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess.check_call(
            [
                triton_shared_opt_path,
                vector_path,
                *transform_passes,
                "--canonicalize",
                "--mlir-print-debuginfo",
                "-o",
                transformed_vector_path,
            ]
        )
        _dump_ir_if_needed([transformed_vector_path])
        vector_path = transformed_vector_path
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
_LLVM_HELPER_SYMBOLS = {"dealloc_helper"}


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
    # LLVM O3 can prove a generated deallocation helper unnecessary and erase
    # all calls without deleting the externally visible helper definition.
    # Such runtime helpers are never Triton kernel entry points.
    candidates = sorted(definitions - callees - _LLVM_HELPER_SYMBOLS)
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
            # IME path: cross-compile to RISC-V with XSMTIME (vfmadot / vmadot).
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
                "-O3",
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
                if _llir_needs_scalar_riscv_codegen(llir):
                    # libm/intrinsic/atomic kernels: RVV vlen hints and -O3 loop
                    # opts still miscompile these loops on native riscv64.
                    llc_args = [
                        llc_path,
                        src_path,
                        "-filetype=obj",
                        "-O2",
                        "-relocation-model=pic",
                        f"-mattr={DEFAULT_LLC_FEATURES}",
                        "-vectorize-loops=false",
                        "-vectorize-slp=false",
                        "-o",
                        dst_path,
                    ]
                else:
                    llc_args.extend(
                        [
                            f"-mattr={DEFAULT_LLC_FEATURES}",
                            "-riscv-v-vector-bits-min=128",
                            "-riscv-v-vector-bits-max=128",
                        ]
                    )
            elif platform.machine() in {"x86_64", "AMD64"}:
                llc_args.extend(["-mcpu=native"])
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
    # The RISC-V backend supports fp8_e4m3fn storage/conversion through
    # Triton's fp8e4nv IR type. Keep the list explicit so unsupported fp8
    # variants still fail at frontend type legalization.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv",)
    allow_fp8e4nv: bool = True
    max_num_imprecise_acc_default: int = 0
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    target_triple: str = None
    target_features: str = None
    openmp_num_threads: int = 0
    allow_fp_reassoc: bool = False

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
        args.setdefault("openmp_num_threads", _get_openmp_num_threads())
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
        stages["llir"] = lambda src, metadata: _optimize_llir(
            _ttsharedir_to_llir(src, options), options
        )
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

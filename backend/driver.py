import hashlib
import tempfile

import os
import subprocess
import platform
import importlib.util
import sys
import triton
from triton._C.libtriton import native_specialize_impl

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from .paths import _get_llvm_bin_path


def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")

    return sanitizer_type


def _sanitizer_available(sanitizer_type):
    if "LD_PRELOAD" not in os.environ:
        return False
    if f"libclang_rt.{sanitizer_type}.so" not in os.environ["LD_PRELOAD"]:
        return False

    return True


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


def _openmp_enabled() -> bool:
    return _get_openmp_num_threads() > 1


def _append_openmp_link_args(subprocess_args):
    if not _openmp_enabled():
        return subprocess_args

    clang_path = _get_llvm_bin_path("clang++")
    llvm_root = Path(clang_path).resolve().parent.parent
    libomp_path = next(llvm_root.rglob("libomp.so"), None)
    if not libomp_path:
        raise Exception("libomp.so does not exist.")

    subprocess_args.extend(
        [
            f"-L{libomp_path.parent}",
            "-fopenmp",
            f"-Wl,-rpath,{libomp_path.parent}",
        ]
    )
    return subprocess_args


def _append_vector_math_link_args(subprocess_args):
    # LLVM's LIBMVEC vector-library mapping emits glibc vector ABI symbols such
    # as _ZGVdN8v_expf. Link them explicitly into the generated launcher.
    if platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"}:
        subprocess_args.append("-lmvec")
    return subprocess_args


def _grid_parallel_pragma() -> str:
    if _openmp_enabled():
        return (
            "#pragma omp parallel for collapse(3) num_threads("
            f"{_get_openmp_num_threads()})"
        )
    if _get_sanitizer_type() == "tsan":
        return "#pragma omp parallel for collapse(3)"
    return ""


# -------------------- Launcher ----------------------------
_OMP_PARALLEL_FOR_PLACEHOLDER = "__TRITON_SHARED_OMP_PARALLEL_FOR__"
_OMP_PARALLEL_FOR = "#pragma omp parallel for collapse(3)"
_LAUNCHER_CACHE_VERSION = b"triton_shared_launcher_v3"


def _get_ordinary_openmp_config(sanitizer_type):
    if platform.system() == "Windows" or sanitizer_type != "":
        return None, False

    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    enabled = omp_num_threads is not None and omp_num_threads != "1"
    return omp_num_threads, enabled


def _finalize_launcher_src(launcher_src, parallel_pragma):
    return launcher_src.replace(_OMP_PARALLEL_FOR_PLACEHOLDER, parallel_pragma)


def _encode_omp_cache_state(omp_num_threads):
    if omp_num_threads is None:
        return b"\x00"
    return b"\x01" + omp_num_threads.encode("utf-8")


def _launcher_cache_key(src, kernel_obj, omp_num_threads):
    cache_input = (
        src.encode("utf-8")
        + kernel_obj
        + _LAUNCHER_CACHE_VERSION
        + b"\x00OMP_NUM_THREADS\x00"
        + _encode_omp_cache_state(omp_num_threads)
    )
    return hashlib.sha256(cache_input).hexdigest()


def _ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
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
        # Proper support for bfloat16 and float16 is not yet handled.
        # https://github.com/microsoft/triton-shared/issues/348
        # "fp16": "TODO",
        # "bf16": "TODO",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _extracted_type(ty):
    if ty[0] == "*":
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join(
        [_format_of(_extracted_type(ty)) for ty in signature.values()]
    )
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    kernel_arg_decls = ", ".join(
        _ty_to_cpp(ty) if ty[0] != "*" else "int64_t, void*"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_arg_decls += ", " if kernel_arg_decls else ""

    kernel_parameters = ", ".join(
        f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_parameters += ", " if kernel_parameters else ""

    launch_args_str = ", ".join(
        f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"
        for i, ty in signature.items()
    )
    get_ptr_checks_str = "; ".join(
        (
            f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); "
            f"if (!ptr_info{i}.valid) return NULL;"
            if ty[0] == "*"
            else ""
        )
        for i, ty in signature.items()
    )

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    // Parallelize independent Triton grid programs when OpenMP is enabled.
    {_OMP_PARALLEL_FOR_PLACEHOLDER}
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          // Use some random type "char" here.
          {" ".join(f"StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};" for i, ty in signature.items() if i not in constants and ty[0] == "*")}
          {kernel_name}({kernel_parameters}
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  static PyObject *data_ptr_name = NULL;
  if (!data_ptr_name)
    data_ptr_name = PyUnicode_InternFromString("data_ptr");
  if (data_ptr_name) {{
    // Call the method through CPython's vectorcall path. This avoids both the
    // temporary bound-method object and the empty argument tuple per pointer.
    PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_name);
    if (!ret) {{
      PyErr_Clear();
    }} else {{
      if (!PyLong_Check(ret)) {{
        PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
        Py_DECREF(ret);
        ptr_info.valid = false;
        return ptr_info;
      }}
      ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
      Py_DECREF(ret);
      return ptr_info;
    }}
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {" ".join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas,
  //      &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {get_ptr_checks_str};
  _launch(gridX, gridY, gridZ, {launch_args_str});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, "include")
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib = "{name}{major}{minor}.lib".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    else:
        py_include_dir = os.path.join(
            sys.base_prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        py_lib_dir = os.path.join(sys.base_prefix, "lib")
        py_lib = "{name}{major}.{minor}".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")
    # Loading an extension module from disk costs O(100 us) even when its
    # compiled shared object is already in Triton's cache. Keep the imported
    # module and bound launch function alive for repeated kernel invocations.
    loaded_launchers = {}
    loaded_kernels = {}

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile one kernel every time.
        # The cu_function parameter actually contains our kernel obj.
        # See CPUUtils.load_binary method.
        kernel_obj = cu_function
        kernel_name = kernel_metadata[6]  # see pack_metadata in compiler.py
        sanitizer_type = _get_sanitizer_type()
        omp_num_threads, ordinary_openmp_enabled = _get_ordinary_openmp_config(
            sanitizer_type
        )
        custom_openmp_enabled = _openmp_enabled()
        kernel_key = (
            kernel_name,
            kernel_obj,
            sanitizer_type,
            omp_num_threads,
            _get_openmp_num_threads(),
        )
        launch_fn = loaded_kernels.get(kernel_key)
        if launch_fn is not None:
            return launch_fn(
                gridX,
                gridY,
                gridZ,
                kernel_metadata,
                launch_metadata,
                launch_enter_hook,
                launch_exit_hook,
                *args,
            )
        if custom_openmp_enabled:
            parallel_pragma = _grid_parallel_pragma()
        elif sanitizer_type == "tsan" or ordinary_openmp_enabled:
            parallel_pragma = _OMP_PARALLEL_FOR
        else:
            parallel_pragma = ""
        src = _finalize_launcher_src(launcher_src, parallel_pragma)
        src = src.replace(kernel_placeholder_name, kernel_name)

        key = _launcher_cache_key(src, kernel_obj, omp_num_threads)
        cache = get_cache_manager(key)
        name = f"__triton_shared_ref_cpu_kernel_launcher_{key[:16]}"
        src = src.replace("__triton_shared_ref_cpu_kernel_launcher", name)

        loaded = loaded_launchers.get(key)
        if loaded is not None:
            return loaded[1](
                gridX,
                gridY,
                gridZ,
                kernel_metadata,
                launch_metadata,
                launch_enter_hook,
                launch_exit_hook,
                *args,
            )

        if platform.system() == "Windows":
            filename = f"{name}.pyd"
        else:
            filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                if platform.system() == "Windows":
                    if sanitizer_type != "":
                        raise Exception(
                            "Sanitizers are not supported on Windows with triton-shared."
                        )

                    obj_path = os.path.join(tmpdir, "kernel.obj")
                    launcher_src_path = os.path.join(tmpdir, "main.cxx")
                    so_path = os.path.join(tmpdir, "kernel.pyd")
                    Path(obj_path).write_bytes(kernel_obj)
                    Path(launcher_src_path).write_text(src)
                    # Compile it together.
                    subprocess.check_call(
                        [
                            "cl",
                            "/LD",
                            "/std:c++17",
                            launcher_src_path,
                            obj_path,
                            f"-I{py_include_dir}",
                            f"-I{include_dir}",
                            "/link",
                            f"/LIBPATH:{py_lib_dir}",
                            "/link",
                            f"{py_lib}",
                            f"/OUT:{so_path}",
                        ]
                    )
                else:
                    obj_path = os.path.join(tmpdir, "kernel.o")
                    launcher_src_path = os.path.join(tmpdir, "main.cxx")
                    so_path = os.path.join(tmpdir, "kernel.so")
                    Path(obj_path).write_bytes(kernel_obj)
                    Path(launcher_src_path).write_text(src)

                    # Compile it together.
                    if sanitizer_type != "":
                        clang_path = _get_llvm_bin_path("clang++")
                        llvm_root = Path(clang_path).resolve().parent.parent

                        subprocess_args = [
                            clang_path,
                            "-std=c++17",
                            launcher_src_path,
                            obj_path,
                            f"-I{py_include_dir}",
                            f"-I{include_dir}",
                            f"-L{py_lib_dir}",
                            f"-Wl,-rpath,{py_lib_dir}",
                            "-shared",
                            f"-l{py_lib}",
                            "-fPIC",
                            "-o",
                            so_path,
                        ]
                        if platform.system() == "Linux":
                            subprocess_args.append("-Wl,-Bsymbolic")

                        if not _sanitizer_available(sanitizer_type):
                            raise Exception(
                                'Use LD_PRELOAD="path/to/libclang_rt.'
                                + sanitizer_type
                                + '.so" TRITON_SHARED_SANITIZER_TYPE='
                                + sanitizer_type
                                + " python ..."
                            )

                        if sanitizer_type == "asan":
                            subprocess_args.extend(
                                ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
                            )
                        elif sanitizer_type == "tsan":
                            # ensure that openmp is available
                            libomp_path = next(
                                llvm_root.rglob("libomp.so"),
                                None,
                            )

                            if not libomp_path:
                                raise Exception("libomp.so does not exist.")

                            libomp_path = str(libomp_path.parent)

                            subprocess_args.extend(
                                [
                                    "-g",
                                    "-fsanitize=thread",
                                    "-fopenmp",
                                    f"-Wl,-rpath,{libomp_path}",
                                ]
                            )

                        if sanitizer_type != "tsan" and _openmp_enabled():
                            subprocess_args = _append_openmp_link_args(subprocess_args)

                        subprocess_args = _append_vector_math_link_args(subprocess_args)

                        subprocess.check_call(subprocess_args)
                    else:
                        compiler = "g++"
                        subprocess_args = [
                            compiler,
                            "-std=c++17",
                            launcher_src_path,
                            obj_path,
                            f"-I{py_include_dir}",
                            f"-I{include_dir}",
                            "-shared",
                            "-fPIC",
                            "-o",
                            so_path,
                        ]
                        if ordinary_openmp_enabled:
                            subprocess_args.append("-fopenmp")
                        if platform.system() == "Linux":
                            subprocess_args.append("-Wl,-Bsymbolic")
                        if custom_openmp_enabled and not ordinary_openmp_enabled:
                            subprocess_args.append("-fopenmp")
                        subprocess_args = _append_vector_math_link_args(subprocess_args)
                        subprocess.check_call(subprocess_args)

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        launch_fn = mod.launch
        loaded_launchers[key] = (mod, launch_fn)
        loaded_kernels[kernel_key] = launch_fn
        return launch_fn(
            gridX,
            gridY,
            gridZ,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            *args,
        )

    return launch


_compiled_launcher_cache = {}
_launcher_signature_cache = {}


class CPULauncher(object):
    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            return src.fn.arg_names.index(i) if isinstance(i, str) else i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_key = (
            tuple(sorted(constants.items())),
            tuple(signature.items()),
            _get_openmp_num_threads(),
            _get_sanitizer_type(),
        )
        self.launch = _launcher_signature_cache.get(launcher_key)
        if self.launch is not None:
            return
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = _compiled_launcher_cache.get(launcher_src)
        if self.launch is None:
            self.launch = compile_module(launcher_src, kernel_placeholder_name)
            _compiled_launcher_cache[launcher_src] = self.launch
        _launcher_signature_cache[launcher_key] = self.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class PreparedCPUKernel:
    """A fixed-specialization CPU launch path with runtime pointer arguments."""

    def __init__(self, jit_kernel, grid, *sample_args, **compile_kwargs):
        # Separate kernel parameters from backend/compiler options before
        # binding the function signature. JITFunction.warmup still receives
        # both sets and remains the authority for specialization and codegen.
        parameter_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key in jit_kernel.arg_names
        }
        bound = jit_kernel.signature.bind(*sample_args, **parameter_kwargs)
        bound.apply_defaults()
        argument_template = [bound.arguments[name] for name in jit_kernel.arg_names]

        compiled = jit_kernel.warmup(
            *sample_args,
            grid=grid,
            **compile_kwargs,
        )
        # Accessing .run initializes the CPU launcher and module lifecycle once.
        launcher = compiled.run

        if callable(grid):
            grid = grid(bound.arguments)
        grid = tuple(grid)
        if not 1 <= len(grid) <= 3:
            raise ValueError("CPU launch grid must have between one and three axes")
        self.grid = grid + (1,) * (3 - len(grid))
        self.compiled = compiled
        self.launcher = launcher
        self.jit_kernel = jit_kernel
        self.original_grid = grid
        self.compile_kwargs = compile_kwargs.copy()
        self.backend_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key not in jit_kernel.arg_names
        }
        self.argument_template = argument_template
        constexprs = set(jit_kernel.constexprs)
        self.runtime_indices = [
            index for index in range(len(argument_template)) if index not in constexprs
        ]
        # Use Triton's generated native binder as the source of truth for
        # dtype, scalar-value and alignment specialization. Keeping only the
        # resulting tuples avoids retaining the sample tensors.
        device = triton.runtime.driver.active.get_current_device()
        _, _, _, backend, _ = jit_kernel.device_caches[device]
        self.runtime_specializations = []
        for index in self.runtime_indices:
            param = jit_kernel.params[index]
            specialize = not param.do_not_specialize
            if param.annotation_type == "u1" or param.annotation_type[:2] in {
                "fp",
                "bf",
            }:
                specialize = False
            self.runtime_specializations.append(
                (
                    backend,
                    param.is_const,
                    specialize,
                    not param.do_not_specialize_on_alignment,
                    native_specialize_impl(
                        backend,
                        argument_template[index],
                        param.is_const,
                        specialize,
                        not param.do_not_specialize_on_alignment,
                    ),
                )
            )
        # Runtime tensors can be very large. They are only examples used for
        # specialization and must not be retained by a long-lived runner.
        for index in self.runtime_indices:
            self.argument_template[index] = None

    def _call_kwargs(self, args):
        kwargs = dict(zip(self.jit_kernel.arg_names, args))
        kwargs.update(self.backend_kwargs)
        return kwargs

    def _standard_launch(self, args):
        return self.jit_kernel.run(
            grid=self.original_grid,
            warmup=False,
            **self._call_kwargs(args),
        )

    def __call__(self, *runtime_args):
        if len(runtime_args) != len(self.runtime_indices):
            raise TypeError(
                f"prepared CPU kernel expects {len(self.runtime_indices)} runtime "
                f"arguments, got {len(runtime_args)}"
            )
        args = self.argument_template.copy()
        for index, value in zip(self.runtime_indices, runtime_args):
            args[index] = value
        from triton import knobs

        if (
            knobs.runtime.launch_enter_hook.calls
            or knobs.runtime.launch_exit_hook.calls
            or self.jit_kernel.pre_run_hooks
        ):
            return self._standard_launch(args)

        for value, guard in zip(runtime_args, self.runtime_specializations):
            backend, is_const, specialize, align, expected = guard
            if (
                native_specialize_impl(backend, value, is_const, specialize, align)
                != expected
            ):
                return self._standard_launch(args)

        not_present = object()
        for (name, _), (
            expected,
            globals_dict,
        ) in self.jit_kernel.used_global_vals.items():
            if globals_dict.get(name, not_present) != expected:
                return self._standard_launch(args)
        compiled = self.compiled
        self.launcher(
            self.grid[0],
            self.grid[1],
            self.grid[2],
            None,
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *args,
        )


def prepare_cpu_kernel(jit_kernel, grid, *sample_args, **compile_kwargs):
    """Compile one specialization and return its low-overhead CPU runner.

    The returned callable accepts only non-constexpr kernel arguments, in the
    original signature order. It intentionally omits launch hooks and dynamic
    grids; callers needing those features should use the standard Triton API.
    """

    return PreparedCPUKernel(jit_kernel, grid, *sample_args, **compile_kwargs)


class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # obj of the kernel so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
            kernel_obj,  # non-null module/lifecycle sentinel
            kernel_obj,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )

    @staticmethod
    def unload_module(module):
        # The object code is owned by CompiledKernel and the imported launcher
        # module is retained by compile_module's cache, so there is no native
        # CPU module handle to release here.
        return None


class CPUDriver(DriverBase):
    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "obj"

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)

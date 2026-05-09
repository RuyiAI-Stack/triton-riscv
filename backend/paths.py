import os
import shutil
from pathlib import Path


def _get_triton_shared_opt_path() -> str:
    env_path = os.getenv("TRITON_SHARED_OPT_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return str(path)
        raise RuntimeError(f"TRITON_SHARED_OPT_PATH points to a missing file: {path}")

    bundled_path = Path(__file__).resolve().parent / "bin" / "triton-shared-opt"
    if bundled_path.is_file():
        return str(bundled_path)

    path_entry = shutil.which("triton-shared-opt")
    if path_entry:
        return path_entry

    raise RuntimeError(
        f"Unable to locate bundled 'triton-shared-opt' at {bundled_path}, "
        "resolve it via TRITON_SHARED_OPT_PATH, or find it in PATH."
    )


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path:
        return os.path.join(path, bin_name)

    path_entry = shutil.which(bin_name)
    if path_entry:
        return path_entry

    raise Exception(f"Unable to locate '{bin_name}' via LLVM_BINARY_DIR or PATH.")


def _get_buddy_bin_path(bin_name: str) -> str:
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path:
        return os.path.join(path, bin_name)

    path_entry = shutil.which(bin_name)
    if path_entry:
        return path_entry

    raise Exception(f"Unable to locate '{bin_name}' via BUDDY_MLIR_BINARY_DIR or PATH.")


def _get_buddy_opt_path() -> str:
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path:
        return os.path.join(path, "buddy-opt")

    path_entry = shutil.which("buddy-opt")
    if path_entry:
        return path_entry

    raise Exception("Unable to locate 'buddy-opt' via BUDDY_MLIR_BINARY_DIR or PATH.")

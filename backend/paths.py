"""Locate the external executables used by the backend.

Resolution order for every tool:

1. an explicit environment variable (local development / build trees),
2. a binary bundled inside this package under ``bin/``,
3. ``PATH`` (this is how the ``buddy`` pip package exposes
   ``buddy-opt`` / ``buddy-translate`` / ``buddy-llc`` console scripts).

Keeping the package-relative and ``PATH`` fallbacks means a plain
``pip install`` works without the user configuring any environment variable.
"""

import os
import shutil
from pathlib import Path

_PACKAGE_BIN_DIR = Path(__file__).resolve().parent / "bin"


def _bundled_bin_path(bin_name: str) -> str | None:
    bundled = _PACKAGE_BIN_DIR / bin_name
    if bundled.is_file():
        return str(bundled)
    return None


def _env_bin_path(env_var: str, bin_name: str) -> str | None:
    base = os.getenv(env_var, "").strip()
    if not base:
        return None
    candidate = Path(base) / bin_name
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(f"{env_var} is set but '{bin_name}' is missing under {base}")


def _get_triton_shared_opt_path() -> str:
    env_path = os.getenv("TRITON_SHARED_OPT_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return str(path)
        raise RuntimeError(f"TRITON_SHARED_OPT_PATH points to a missing file: {path}")

    if bundled := _bundled_bin_path("triton-shared-opt"):
        return bundled

    # No PATH lookup here: the only triton-shared-opt on PATH is this package's
    # own console script, which would resolve back to itself.
    raise RuntimeError(
        f"Unable to locate 'triton-shared-opt': no copy at "
        f"{_PACKAGE_BIN_DIR / 'triton-shared-opt'} and TRITON_SHARED_OPT_PATH is unset."
    )


def _get_llvm_bin_path(bin_name: str) -> str:
    if env_path := _env_bin_path("LLVM_BINARY_DIR", bin_name):
        return env_path

    if bundled := _bundled_bin_path(bin_name):
        return bundled

    path_entry = shutil.which(bin_name)
    if path_entry:
        return path_entry

    raise RuntimeError(
        f"Unable to locate LLVM tool '{bin_name}' via LLVM_BINARY_DIR, the "
        "bundled package bin/, or PATH."
    )


def _get_buddy_bin_path(bin_name: str) -> str:
    # The `buddy` pip package installs buddy-opt / buddy-translate / buddy-llc
    # as console scripts, so PATH covers the pip-installed case.
    if env_path := _env_bin_path("BUDDY_MLIR_BINARY_DIR", bin_name):
        return env_path

    if bundled := _bundled_bin_path(bin_name):
        return bundled

    path_entry = shutil.which(bin_name)
    if path_entry:
        return path_entry

    raise RuntimeError(
        f"Unable to locate Buddy tool '{bin_name}' via BUDDY_MLIR_BINARY_DIR, "
        "the bundled package bin/, or PATH (install the 'buddy' package)."
    )


def _get_buddy_opt_path() -> str:
    return _get_buddy_bin_path("buddy-opt")

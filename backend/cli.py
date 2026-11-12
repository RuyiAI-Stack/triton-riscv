from __future__ import annotations

import subprocess
import sys

from .paths import _get_triton_shared_opt_path


def triton_shared_opt() -> int:
    return subprocess.call([_get_triton_shared_opt_path(), *sys.argv[1:]])

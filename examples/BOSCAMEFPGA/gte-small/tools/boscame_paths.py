"""Make the shared example path helpers available to standalone ViT tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from paths import (  # noqa: E402, F401
    buddy_tools,
    llvm_lib_dir,
    source_identity,
)

# ruff: noqa: F821
"""Optional cross-repository tests, separate from the frontend core suite."""

import os
import sys
from pathlib import Path

import lit.formats

config.name = "TRITON-CONTRACT-INTEGRATION"
config.test_format = lit.formats.ShTest()
config.suffixes = [".py", ".mlir"]
config.excludes = ["Inputs", "lit.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = lit_config.params["integration_build_dir"]
buddy_bin = Path(lit_config.params["buddy_bin"])
llvm_bin = lit_config.params["llvm_bin"]
frontend = os.environ.get("TRITON_SHARED_OPT_PATH")
if not frontend or not Path(frontend).is_file():
    lit_config.fatal("set TRITON_SHARED_OPT_PATH to the external frontend tool")
config.environment.update(
    {
        "PATH": str(buddy_bin)
        + os.pathsep
        + llvm_bin
        + os.pathsep
        + os.environ["PATH"],
        "TRITON_SHARED_OPT_PATH": frontend,
        "BUDDY_ROOT": lit_config.params["buddy_root"],
        "BOSCAME_TEST_PIPELINE_OPT": str(
            Path(config.test_exec_root) / "contract-pipeline-opt"
        ),
        "BUDDY_OPT": str(buddy_bin / "buddy-opt"),
        "BUDDY_TRANSLATE": str(buddy_bin / "buddy-translate"),
    }
)
bridge = Path(config.test_exec_root) / "contract-pipeline-opt"
bridge.parent.mkdir(parents=True, exist_ok=True)
bridge.write_bytes(
    (Path(__file__).parent / "Inputs/pipeline-opt.py").read_bytes()
)
bridge.chmod(0o755)
config.substitutions.extend(
    [
        ("%PYTHON", os.environ.get("TRITON_CONTRACT_PYTHON", sys.executable)),
        ("contract-pipeline-opt", str(bridge)),
    ]
)

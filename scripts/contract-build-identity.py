#!/usr/bin/env python3
"""Generate the frontend's compiled identity without recording local paths."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def main():
    root, triton, llvm_build, output = map(Path, sys.argv[1:])
    sources = {}
    for directory in (
        "include",
        "lib",
        "tools",
        "backend",
        "patches",
        "scripts",
        "cmake",
    ):
        for path in sorted((root / directory).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and "bin" not in path.relative_to(root).parts
            ):
                sources[path.relative_to(root).as_posix()] = sha(path)
    for name in ("CMakeLists.txt", "triton-hash.txt", "triton_shared.cc"):
        sources[name] = sha(root / name)
    payload = {
        "frontend_revision": revision(root),
        "frontend_sources_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "frontend_files": sources,
        "triton_revision": revision(triton),
        "triton_patch_sha256": hashlib.sha256(
            subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"], cwd=triton
            )
        ).hexdigest(),
        "contract_schema_version": 1,
        "llvm_build_files": {
            name: sha(llvm_build / name)
            for name in (
                "include/llvm/Config/llvm-config.h",
                "include/llvm/Support/VCSRevision.h",
                "lib/libMLIRIR.a",
                "lib/libMLIRSupport.a",
            )
        },
    }
    text = (
        '// Generated; local paths are intentionally omitted.\nstatic constexpr const char contractBuildIdentity[] = R"identity('
        + json.dumps(payload, sort_keys=True)
        + ')identity";\n'
    )
    if not output.exists() or output.read_text() != text:
        output.write_text(text)


if __name__ == "__main__":
    main()

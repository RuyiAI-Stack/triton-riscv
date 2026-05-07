# Operator Migration Hints
1. The previous migration work was completed by AI. This document contains some prompts for using AI to perform operator migration.
2. The migration targets are the files under the `FlagGems/src/flag_gems/ops` directory. Using the files in this directory can minimize the impact of encapsulation on the migration work.
3. Each file often contains multiple kernels, from which representative kernels can be selected.
4. What AI needs to do is inline the functions called from other files and remove features unsupported by Triton-RISCV, such as autotuning, while keeping only the core implementation.
5. Most importantly, compare the migrated Triton implementation with the PyTorch interface to verify correctness.
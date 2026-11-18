# Overview

## 1. Stand-Alone
The middle layer can be used as a stand-alone component to convert Triton dialect to the middle layer dialects. This is intended for testing and validation purposes, but could potentially be used before sending the IR to another MLIR complier.

Stand-alone example:
```
triton-shared-opt --triton-to-linalg %file
```

## 2. Backend Component
The intended use of the Triton middle layer is to be used as a component in a Triton back-end. This can be accomplished by adding the cmake targets it produces and its headers files to that back-end. An example back-end will be published at a later date.

## 3. Reference CPU backend
We also include an experimental reference CPU backend that leverages all existing `mlir` passes. After building, the CPU backend can be used by setting `triton`'s active driver:

```python

import triton
from triton.backends.riscv.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
```

For more examples, please refer to `python/examples`.

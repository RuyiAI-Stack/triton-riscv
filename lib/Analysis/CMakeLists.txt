#
# CMake for TritonSharedAnalysis (KernelContract, masks, pointers).
#
# Inputs: analysis .cpp files in this directory.
#
# Outputs: the analysis library used by conversions and contract verify.
#
# When used: any pass that annotates or checks kernel buffer facts.
#

add_triton_library(TritonSharedAnalysis
  KernelContract.cpp
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  UseAnalysis.cpp

  DEPENDS
  TritonTableGen
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
)

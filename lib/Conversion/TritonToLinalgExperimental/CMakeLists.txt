#
# CMake for the experimental Triton to Linalg conversion library.
#
# Inputs: pass sources including contract tile specialization and
# collapse-shape.
#
# Outputs: a library linked into triton-shared-opt.
#
# When used: the midend build of emitContract support.
#

add_triton_library(TritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  ContractTileSpecialization.cpp
  ReconcilePtrCastsPass.cpp
  TritonToPtrPass.cpp
  CollapseShape.cpp

  DEPENDS
  TritonToLinalgExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTransforms
  MLIRSupport
  MLIRAffineDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPtrDialect
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorTransforms
  TPtrIR
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  TritonSharedUtils

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
)

add_triton_library(ExpandFloat8Conversions
  ExpandFloat8ConversionsPass.cpp

  DEPENDS
  ExpandFloat8ConversionsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
)

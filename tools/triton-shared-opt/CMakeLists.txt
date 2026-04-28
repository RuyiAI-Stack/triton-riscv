get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-shared-opt triton-shared-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-shared-opt)
target_link_libraries(triton-shared-opt PRIVATE
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

# Note: MLIRRegisterAllPasses and NVGPUIR are symbol libraries that may not exist
# MLIRRegisterAllPasses is provided by LLVM but may not be available in all builds
# NVGPUIR is only built when nvidia backend is enabled
# Allow undefined symbols for these optional libraries
target_link_options(triton-shared-opt PRIVATE -Wl,--allow-shlib-undefined)

if (DEFINED TRITON_WHEEL_DIR)
  set(TRITON_SHARED_OPT_WHEEL_DIR "${TRITON_WHEEL_DIR}/backends/triton_shared/bin")
  add_custom_command(TARGET triton-shared-opt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TRITON_SHARED_OPT_WHEEL_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:triton-shared-opt>
      "${TRITON_SHARED_OPT_WHEEL_DIR}/$<TARGET_FILE_NAME:triton-shared-opt>"
    VERBATIM)
endif()

mlir_check_all_link_libraries(triton-shared-opt)


llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCe_DIR}/lit.cfg.py
)

set(TRITON_SHARED_TEST_DEPENDS
  triton-shared-opt
)

if(TRITON_SHARED_BUILD_TRITON_SAN)
  list(APPEND TRITON_SHARED_TEST_DEPENDS SanitizerAttributes)
endif()

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")
add_lit_testsuite(check-triton-shared-lit-tests "Running the triton-shared regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${TRITON_SHARED_TEST_DEPENDS}
  )

set_target_properties(check-triton-shared-lit-tests PROPERTIES FOLDER "Tests")

# Optional example/simulator integration. Neither core build nor AOT export
# requires Buddy; explicitly select its checkout and installed build tools here.
set(BUDDY_ROOT "" CACHE PATH "External Buddy checkout for BOSCAME example tests")
set(BUDDY_MLIR_BINARY_DIR "" CACHE PATH "External Buddy binary directory")
if(BUDDY_ROOT AND BUDDY_MLIR_BINARY_DIR)
  add_lit_testsuite(check-triton-contract-integration
    "Running optional BOSCAME example integration tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/Integration/BOSCAMEFPGA
    ARGS "-Dbuddy_root=${BUDDY_ROOT}"
         "-Dbuddy_bin=${BUDDY_MLIR_BINARY_DIR}"
         "-Dllvm_bin=${LLVM_TOOLS_BINARY_DIR}"
         "-Dintegration_build_dir=${CMAKE_CURRENT_BINARY_DIR}/boscame-integration"
    DEPENDS triton-shared-opt)
endif()

add_lit_testsuites(TRITON-SHARED-LIT-TESTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TRITON_SHARED_TEST_DEPENDS})

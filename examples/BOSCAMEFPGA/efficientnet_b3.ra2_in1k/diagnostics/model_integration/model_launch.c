//
// model_launch.c for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k.
//

#include "model_board_support.h"
#ifndef MODEL_COMPARE_ATOL
#define MODEL_COMPARE_ATOL 0.05f
#endif
#ifndef MODEL_COMPARE_RTOL
#define MODEL_COMPARE_RTOL 0.01f
#endif
int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  efficientnet_install_trap();
  nh_e6_cache_writes();
  uint64_t begin = nr_cycles();
  float *logits = efficientnet_b3_execute_graph(
      (void *)model_params_blob, EFFICIENTNET_B3_PARAMETER_BYTES,
      (void *)model_input_blob, model_workspace_blob,
      EFFICIENTNET_B3_WORKSPACE_BYTES, &efficientnet_hooks);
  ame_fence();
  uint64_t elapsed = nr_cycles() - begin;
  const float *expected = (const float *)model_reference_blob;
  unsigned errors = 0, top = 0;
  float maximum = 0.0f, top_value;
  bosc_ame_ddr_read_begin();
  top_value = logits[0];
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  for (unsigned i = 0; i < EFFICIENTNET_B3_OUTPUT_ELEMENTS; ++i) {
    float a = logits[i], b = expected[i], d = a - b;
    if (d < 0)
      d = -d;
    if (a > top_value) {
      top_value = a;
      top = i;
    }
    if (d > maximum)
      maximum = d;
    union {
      float f;
      uint32_t u;
    } v = {a};
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (v.u >> (byte * 8)) & 255u;
      hash *= UINT64_C(0x100000001b3);
    }
    // Declared diagnostic threshold against the W8A8 semantics golden.
    // Host still downloads and reports the complete difference vector.
    float m = b < 0 ? -b : b;
    if ((v.u & 0x7f800000u) == 0x7f800000u ||
        d > MODEL_COMPARE_ATOL + MODEL_COMPARE_RTOL * m)
      ++errors;
  }
  bosc_ame_ddr_write_begin();
  union {
    float f;
    uint32_t u;
  } m = {maximum};
  efficientnet_export_results(logits);
  model_cycles = elapsed;
  model_logits_address = (uintptr_t)model_export_logits;
  model_logits_hash = hash;
  model_errors = errors;
  model_checked = EFFICIENTNET_B3_OUTPUT_ELEMENTS;
  model_top1 = top;
  model_max_abs_bits = m.u;
  model_returned = 1;
  nh_e6_cache_writes();
  __asm__ volatile("fence iorw, iorw" ::: "memory");
#ifdef MODEL_HOST_NUMERICAL_VALIDATION
  // Preserve every inline error in UART. Completion enables DDR collection;
  // independent host audit decides numerical success from the full logits.
  return 0;
#else
  return errors ? 1 : 0;
#endif
}

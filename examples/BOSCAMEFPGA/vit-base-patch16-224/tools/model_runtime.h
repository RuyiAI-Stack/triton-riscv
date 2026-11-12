//
// model_runtime.h for BOSCAME FPGA model 'vit-base-patch16-224'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/vit-base-patch16-224.
//

#ifndef VIT_MODEL_RUNTIME_H
#define VIT_MODEL_RUNTIME_H

// Shared ABI between the generated whole-model launcher and model_support.c.

#include <stddef.h>
#include <stdint.h>

#include "support.h"

// Private-DDR arena failure hook: republished through the platform DMA failure
// channel so NH reports reason/address/bytes instead of an opaque trap.
void vit_publish_private_ddr_failure(uint32_t reason, uint64_t address,
                                     uint64_t bytes);

// Embedded blobs (generated model_params.S, 64-byte aligned sections).
extern const unsigned char model_params_blob[];
extern const unsigned char model_input_blob[];
extern const unsigned char model_reference_blob[];
extern unsigned char model_workspace_blob[];

// Result cells read by the NH report path.  E6 writes them before returning;
// NH reads them only after the DONE publication.
extern volatile unsigned model_returned;
extern volatile unsigned model_validation_executed;
extern volatile unsigned model_checked;
extern volatile unsigned model_errors;
extern volatile uint64_t model_cycles;
extern volatile uint64_t model_logits_fnv64;
extern volatile uint32_t model_top1;
extern volatile uint32_t model_max_abs_bits;

static inline uint64_t model_fnv1a64(const void *data, size_t bytes) {
  const unsigned char *p = (const unsigned char *)data;
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= UINT64_C(0x100000001b3);
  }
  return hash;
}

#endif // VIT_MODEL_RUNTIME_H

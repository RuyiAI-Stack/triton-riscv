//
// board_support.h for BOSCAME FPGA model 'e5-small-v2'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/e5-small-v2.
//

#ifndef VIT_BOARD_SUPPORT_H
#define VIT_BOARD_SUPPORT_H

#include "nh_e6_platform.h"
#include "nr_runtime.h"
#include "scheduler_output_runtime.h"
#include "support.h"

unsigned vit_board_kernel_begin(void);
void vit_board_kernel_end(unsigned slot, uint64_t cycles);
void vit_board_validation_round(unsigned round, unsigned checked,
                                unsigned errors, unsigned guard_errors,
                                unsigned input_errors, uint64_t output_hash);
uint32_t vit_board_mmio_read32(uintptr_t address);
float fabsf(float value);

#endif

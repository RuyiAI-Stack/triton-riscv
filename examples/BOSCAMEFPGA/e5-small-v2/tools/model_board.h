//
// model_board.h for BOSCAME FPGA model 'e5-small-v2'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/e5-small-v2.
//

#ifndef E5_MODEL_BOARD_H
#define E5_MODEL_BOARD_H
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "support.h"
uint32_t e5_model_mmio_read32(uintptr_t address);
uint64_t e5_model_kernel_begin(unsigned case_id);
void e5_model_kernel_end(unsigned case_id, uint64_t begin);
void e5_model_checkpoint(unsigned layer, const float *hidden);
int e5_model_launch(void);
extern float e5_sentence_output[384];
extern float e5_hidden_output[128 * 384];
extern float e5_layer_outputs[13 * 128 * 384];
extern volatile uint64_t e5_model_cycles, e5_kernel_cycles, e5_kernel_calls;
extern volatile uint64_t e5_kernel_case, e5_checkpoints, e5_launch_result;
#endif

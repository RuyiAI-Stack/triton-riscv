//
// model_support.h for BOSCAME FPGA model 'mobilenetv3_small_100.lamb_in1k'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k.
//

#ifndef MOBILENET_MODEL_SUPPORT_H
#define MOBILENET_MODEL_SUPPORT_H
#include <stdint.h>
void model_start(unsigned total);
void model_op_begin(unsigned index, const char *name, const char *case_name,
                    const char *symbol);
void model_op_end(uint64_t elapsed);
// Optional NH runtime trap hook; reads only this core's execution state.
void model_report_failure(void);
int model_verify(const float *actual, const float *golden, unsigned count,
                 unsigned expected_top1, uint64_t elapsed);
#endif

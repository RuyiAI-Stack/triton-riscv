//
// model.h for BOSCAME FPGA model 'xlnet-base-cased'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/xlnet-base-cased.
//

#ifndef XLNET_FIXED_MODEL_H
#define XLNET_FIXED_MODEL_H
#include "support.h"
// All pointers are nonalias, 64-byte aligned. Shapes are fixed and contiguous.
// Matrices contain offline signed-i8 physical [N,K] rows and FP32 scales[N].
typedef struct {
  int8_t *q;
  float *scale;
} XLNetMatrix;
typedef struct {
  XLNetMatrix q, k, v, r, o, segment[12], ff1, ff2;
  float *rw_bias, *rr_bias, *rs_bias; // each [12,64]
  float *attn_norm_weight, *attn_norm_bias, *ffn_norm_weight, *ffn_norm_bias;
  float *ff1_bias, *ff2_bias;
} XLNetLayer;
typedef struct {
  float *embedding;          // original checkpoint FP32 [32000,768]
  float *relative_positions; // offline [256,768], sin then cos
  XLNetLayer layers[12];
  XLNetMatrix lm_tiles[32];
  float *lm_bias[32]; // [1024] padded last tile; copy first256 of final tile
} XLNetWeights;
#define XLNET_MODEL_SCRATCH_BYTES (24u * 1024u * 1024u)
// Batch1,seq128,eval,bi attention. mask:1 real/0 pad. segments may be NULL;
// supplied token ids may include XLNet special segment 2. No mems/query stream.
// hidden[128,768], optional logits[128,32000]. Return 0 on valid completion.
// This composes real typed entries. Validate scheduler-returned outputs
// locally against execution of the emitted model.
int xlnet_forward(const XLNetWeights *, int32_t *ids, int32_t *mask,
                  int32_t *segments, void *scratch, size_t scratch_bytes,
                  float *hidden, float *logits);
#endif

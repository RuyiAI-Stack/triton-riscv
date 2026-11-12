//
// model.h for BOSCAME FPGA model 'e5-small-v2'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/e5-small-v2.
//

#ifndef E5_MODEL_RUNTIME_H
#define E5_MODEL_RUNTIME_H
#include <stddef.h>
#include <stdint.h>

enum {
  E5_SEQ = 128,
  E5_HIDDEN = 384,
  E5_INTERMEDIATE = 1536,
  E5_LAYERS = 12,
  E5_HEADS = 12,
  E5_HEAD_DIM = 32
};

// Every buffer is contiguous and 64-byte-aligned. Linear weights use the
// checkpoint's [output,input] layout; no checkpoint transposition is needed.
// Caller retains read-only ownership throughout inference.
typedef struct {
  const float *weight, *bias;
} E5Linear;
typedef struct {
  const float *weight, *bias;
} E5LayerNorm;
typedef struct {
  E5Linear query, key, value, attention_output, intermediate, output;
  E5LayerNorm attention_norm, output_norm;
} E5Layer;
typedef struct {
  const float *word_embeddings;     // [30522,384]
  const float *position_embeddings; // [512,384]
  const float *type_embeddings;     // [2,384]
  E5LayerNorm embedding_norm;
  E5Layer layers[E5_LAYERS];
} E5Weights;

// Weights, IDs, workspace and output buffers must not overlap. The workspace
// is caller-owned and can be reused after the call returns. No heap allocation
// or implicit global state is used. Invalid IDs/masks or workspace return -1
// before any kernel call. Output is the unit-normalized mean sentence vector;
// last_hidden may be NULL, otherwise it receives [128,384] FP32 states.
size_t e5_workspace_bytes(void);
int e5_infer(const E5Weights *weights, const int32_t *input_ids,
             const int32_t *attention_mask, const int32_t *token_type_ids,
             const int32_t *position_ids, void *workspace,
             size_t workspace_bytes, float *sentence_embedding,
             float *last_hidden);
#endif

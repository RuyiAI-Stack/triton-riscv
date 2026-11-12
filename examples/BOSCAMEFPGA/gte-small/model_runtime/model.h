//
// model.h for BOSCAME FPGA model 'gte-small'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/gte-small.
//

#ifndef GTE_MODEL_RUNTIME_H
#define GTE_MODEL_RUNTIME_H
#include <stddef.h>
#include <stdint.h>

enum {
  GTE_SEQ = 128,
  GTE_HIDDEN = 384,
  GTE_INTERMEDIATE = 1536,
  GTE_LAYERS = 12,
  GTE_HEADS = 12,
  GTE_HEAD_DIM = 32
};

// Every buffer is contiguous and 64-byte-aligned. All weight tensors are FP32;
// convert the upstream FP16 checkpoint before binding these pointers.
// Linear weights use the
// checkpoint's [output,input] layout; no checkpoint transposition is needed.
// Caller retains read-only ownership throughout inference.
typedef struct {
  const float *weight, *bias;
} GTELinear;
typedef struct {
  const float *weight, *bias;
} GTELayerNorm;
typedef struct {
  GTELinear query, key, value, attention_output, intermediate, output;
  GTELayerNorm attention_norm, output_norm;
} GTELayer;
typedef struct {
  const float *word_embeddings;     // [30522,384]
  const float *position_embeddings; // [512,384]
  const float *type_embeddings;     // [2,384]
  GTELayerNorm embedding_norm;
  GTELayer layers[GTE_LAYERS];
} GTEWeights;

// Weights, IDs, workspace and output buffers must not overlap. The workspace
// is caller-owned and can be reused after the call returns. No heap allocation
// or implicit global state is used. Invalid IDs/masks or workspace return -1
// before any kernel call. Output is the unit-normalized mean sentence vector;
// last_hidden may be NULL, otherwise it receives [128,384] FP32 states.
size_t gte_workspace_bytes(void);
int gte_infer(const GTEWeights *weights, const int32_t *input_ids,
              const int32_t *attention_mask, const int32_t *token_type_ids,
              const int32_t *position_ids, void *workspace,
              size_t workspace_bytes, float *sentence_embedding,
              float *last_hidden);
#endif

//
// model.h for BOSCAME FPGA model 'electra-base-discriminator'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/electra-base-discriminator.
//

#ifndef ELECTRA_MODEL_H
#define ELECTRA_MODEL_H
#include <stddef.h>
#include <stdint.h>

// Every tensor starts at a distinct 64-byte-aligned address. Dense matrices
// use physical [out_features,in_features], identical to HF checkpoint storage.
// The caller owns weights, inputs, output and scratch for the entire call.
typedef struct {
  const float *weight, *bias;
#ifdef ELECTRA_PREQUANTIZED_WEIGHTS
  // Identical row quantization to the runtime kernel, prepared on the host.
  const int8_t *quantized;
  const float *scale;
#endif
} ElectraLinear;
typedef struct {
  const float *weight, *bias;
} ElectraNorm;
typedef struct {
  ElectraLinear query, key, value, attention_output, intermediate, output;
  ElectraNorm attention_norm, output_norm;
} ElectraLayer;
typedef struct {
  const float *word_embeddings;       // [30522,768]
  const float *position_embeddings;   // [512,768]
  const float *token_type_embeddings; // [2,768]
  ElectraNorm embedding_norm;
  ElectraLayer layers[12];
  ElectraLinear discriminator_dense, discriminator_prediction;
} ElectraWeights;

#define ELECTRA_MODEL_WORKSPACE_BYTES (16u * 1024u * 1024u)
// Fixed batch=1, seq=128; IDs [0,30522), Types 0/1, Mask binary and nonempty.
// Returns 128 raw replaced-token logits. Dropout is disabled. Input descriptors
// and all primitive calls enforce the same nonalias/alignment ABI as ViT.
// Actual compiled-operator scheduling. The prequantized board specialization
// has been validated on fpga112; see MODEL_BOARD_RESULTS.md for the scope.
int electra_infer(const ElectraWeights *weights, const int32_t *ids,
                  const int32_t *types, const int32_t *mask, float *logits,
                  void *scratch, size_t scratch_bytes);
#endif

//
// model.c for BOSCAME FPGA model 'electra-base-discriminator'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/electra-base-discriminator.
//

#include "model.h"
#include "kernels.h"

#ifdef ELECTRA_MODEL_BOARD
extern void electra_model_trace(unsigned stage, const float *data,
                                size_t count);
#define TRACE(stage, data, count) electra_model_trace(stage, data, count)
#else
#define TRACE(stage, data, count) ((void)0)
#endif

typedef struct {
  unsigned char *base;
  size_t used, size;
} Arena;
static void *take(Arena *a, size_t bytes) {
  size_t start = (a->used + 63u) & ~(size_t)63;
  if (start > a->size || bytes > a->size - start)
    __builtin_trap();
  a->used = start + bytes;
  return a->base + start;
}
static void normalize(const float *x, const ElectraNorm *w, float *out) {
  MemRef2 X = make_2((void *)x, 128, 768), O = make_2(out, 128, 768);
  MemRef1 W = make_1((void *)w->weight, 768), B = make_1((void *)w->bias, 768);
  _mlir_ciface_kernel_layernorm_128x768(&X, &W, &B, &O);
}
static void add(const float *x, const float *y, float *out) {
  MemRef2 X = make_2((void *)x, 128, 768), Y = make_2((void *)y, 128, 768),
          O = make_2(out, 128, 768);
  _mlir_ciface_kernel_add_128x768(&X, &Y, &O);
}
static void pack(const float *x, float *out) {
  MemRef2 X = make_2((void *)x, 128, 768);
  MemRef3 O = make_3(out, 12, 128, 64);
  _mlir_ciface_kernel_pack_heads_12x128x64(&X, &O);
}
// All products overwrite INT32 C. QK's 1/8 occurs only in dequantize_128x128;
// PV's logical right [128,64] is quantized with a compiled transpose.
static void product(Arena *arena, const float *left, const float *right, int n,
                    int k, int transpose, float *out, const int8_t *cached_q,
                    const float *cached_scale) {
  size_t saved = arena->used;
  int8_t *aq = take(arena, (size_t)128 * k);
  int8_t *bq = cached_q ? (int8_t *)cached_q : take(arena, (size_t)n * k);
  int32_t *acc = take(arena, (size_t)128 * n * 4);
  float *as = take(arena, 128 * 4);
  float *bs = cached_scale ? (float *)cached_scale : take(arena, (size_t)n * 4);
  MemRef2 A = make_2((void *)left, 128, k), B = make_2((void *)right, n, k);
  MemRef2 AQ = make_2(aq, 128, k), BQ = make_2(bq, n, k),
          C = make_2(acc, 128, n), O = make_2(out, 128, n);
  MemRef1 AS = make_1(as, 128), BS = make_1(bs, n);
  if (k == 768)
    _mlir_ciface_kernel_quantize_128x768(&A, &AQ, &AS);
  else if (k == 3072)
    _mlir_ciface_kernel_quantize_128x3072(&A, &AQ, &AS);
  else if (k == 64)
    _mlir_ciface_kernel_quantize_128x64(&A, &AQ, &AS);
  else if (k == 128)
    _mlir_ciface_kernel_quantize_128x128(&A, &AQ, &AS);
  else
    __builtin_trap();
  if (cached_q) {
    if (!cached_scale || transpose)
      __builtin_trap();
  } else if (transpose) {
    B = make_2((void *)right, 128, 64);
    _mlir_ciface_kernel_quantize_transpose_128x64(&B, &BQ, &BS);
  } else if (n == 768 && k == 768)
    _mlir_ciface_kernel_quantize_768x768(&B, &BQ, &BS);
  else if (n == 3072 && k == 768)
    _mlir_ciface_kernel_quantize_3072x768(&B, &BQ, &BS);
  else if (n == 768 && k == 3072)
    _mlir_ciface_kernel_quantize_768x3072(&B, &BQ, &BS);
  else if (n == 128 && k == 64)
    _mlir_ciface_kernel_quantize_128x64(&B, &BQ, &BS);
  else if (n == 1 && k == 768)
    _mlir_ciface_kernel_quantize_1x768(&B, &BQ, &BS);
  else
    __builtin_trap();
  if (n == 768 && k == 768)
    _mlir_ciface_kernel_matmul_i8_128x768x768(&AQ, &BQ, &C);
  else if (n == 3072 && k == 768)
    _mlir_ciface_kernel_matmul_i8_128x3072x768(&AQ, &BQ, &C);
  else if (n == 768 && k == 3072)
    _mlir_ciface_kernel_matmul_i8_128x768x3072(&AQ, &BQ, &C);
  else if (n == 128 && k == 64)
    _mlir_ciface_kernel_matmul_i8_128x128x64(&AQ, &BQ, &C);
  else if (n == 64 && k == 128)
    _mlir_ciface_kernel_matmul_i8_128x64x128(&AQ, &BQ, &C);
  else if (n == 1 && k == 768)
    _mlir_ciface_kernel_matmul_i8_128x1x768(&AQ, &BQ, &C);
  else
    __builtin_trap();
  if (n == 768)
    _mlir_ciface_kernel_dequantize_128x768(&C, &AS, &BS, &O);
  else if (n == 3072)
    _mlir_ciface_kernel_dequantize_128x3072(&C, &AS, &BS, &O);
  else if (n == 128)
    _mlir_ciface_kernel_dequantize_128x128(&C, &AS, &BS, &O);
  else if (n == 64)
    _mlir_ciface_kernel_dequantize_128x64(&C, &AS, &BS, &O);
  else if (n == 1)
    _mlir_ciface_kernel_dequantize_128x1(&C, &AS, &BS, &O);
  arena->used = saved;
}
static void linear(Arena *arena, const float *x, const ElectraLinear *w, int n,
                   int k, float *temp, float *out) {
#ifdef ELECTRA_PREQUANTIZED_WEIGHTS
  product(arena, x, w->weight, n, k, 0, temp, w->quantized, w->scale);
#else
  product(arena, x, w->weight, n, k, 0, temp, 0, 0);
#endif
  MemRef2 X = make_2(temp, 128, n), O = make_2(out, 128, n);
  MemRef1 B = make_1((void *)w->bias, n);
  if (n == 768)
    _mlir_ciface_kernel_bias_add_128x768(&X, &B, &O);
  else if (n == 3072)
    _mlir_ciface_kernel_bias_add_128x3072(&X, &B, &O);
  else if (n == 1)
    _mlir_ciface_kernel_bias_add_128x1(&X, &B, &O);
  else
    __builtin_trap();
}
static int aligned(const void *p) { return p && !((uintptr_t)p & 63u); }
int electra_infer(const ElectraWeights *w, const int32_t *ids,
                  const int32_t *types, const int32_t *mask, float *logits,
                  void *scratch, size_t bytes) {
  if (!w || !aligned(ids) || !aligned(types) || !aligned(mask) ||
      !aligned(logits) || !aligned(scratch) ||
      bytes < ELECTRA_MODEL_WORKSPACE_BYTES)
    return -1;
  int any = 0;
  for (int i = 0; i < 128; i++) {
    if (ids[i] < 0 || ids[i] >= 30522 || types[i] < 0 || types[i] > 1 ||
        mask[i] < 0 || mask[i] > 1)
      return -2;
    any |= mask[i];
  }
  if (!any)
    return -2;
  Arena arena = {(unsigned char *)scratch, 0, bytes};
  size_t hidden = (size_t)128 * 768 * 4, wide = (size_t)128 * 3072 * 4;
  float *x = take(&arena, hidden), *y = take(&arena, hidden),
        *temp = take(&arena, wide);
  float *q = take(&arena, hidden), *k = take(&arena, hidden),
        *v = take(&arena, hidden);
  float *qheads = take(&arena, hidden), *kheads = take(&arena, hidden),
        *vheads = take(&arena, hidden);
  float *scores = take(&arena, 12 * 128 * 128 * 4),
        *probs = take(&arena, 12 * 128 * 128 * 4);
  float *heads = take(&arena, hidden), *context = take(&arena, hidden);
  float *ff = take(&arena, wide), *activated = take(&arena, wide);
  MemRef1 IDs = make_1((void *)ids, 128), Types = make_1((void *)types, 128),
          Mask = make_1((void *)mask, 128);
  MemRef2 Word = make_2((void *)w->word_embeddings, 30522, 768),
          Pos = make_2((void *)w->position_embeddings, 512, 768);
  MemRef2 Seg = make_2((void *)w->token_type_embeddings, 2, 768),
          Y = make_2(y, 128, 768);
  _mlir_ciface_kernel_embedding_128x768(&IDs, &Types, &Word, &Pos, &Seg, &Y);
  normalize(y, &w->embedding_norm, x);
  TRACE(0, x, 128u * 768u);
  for (int layer = 0; layer < 12; layer++) {
    const ElectraLayer *l = &w->layers[layer];
    linear(&arena, x, &l->query, 768, 768, temp, q);
    pack(q, qheads);
    linear(&arena, x, &l->key, 768, 768, temp, k);
    pack(k, kheads);
    linear(&arena, x, &l->value, 768, 768, temp, v);
    pack(v, vheads);
    for (int h = 0; h < 12; h++)
      product(&arena, qheads + h * 128 * 64, kheads + h * 128 * 64, 128, 64, 0,
              scores + h * 128 * 128, 0, 0);
    MemRef3 S = make_3(scores, 12, 128, 128), P = make_3(probs, 12, 128, 128);
    _mlir_ciface_kernel_masked_softmax_12x128x128(&S, &Mask, &P);
    for (int h = 0; h < 12; h++)
      product(&arena, probs + h * 128 * 128, vheads + h * 128 * 64, 64, 128, 1,
              heads + h * 128 * 64, 0, 0);
    MemRef3 H = make_3(heads, 12, 128, 64);
    MemRef2 Context = make_2(context, 128, 768);
    _mlir_ciface_kernel_unpack_heads_12x128x64(&H, &Context);
    linear(&arena, context, &l->attention_output, 768, 768, temp, y);
    add(x, y, context);
    normalize(context, &l->attention_norm, x);
    linear(&arena, x, &l->intermediate, 3072, 768, temp, ff);
    MemRef2 FF = make_2(ff, 128, 3072), Act = make_2(activated, 128, 3072);
    _mlir_ciface_kernel_gelu_128x3072(&FF, &Act);
    linear(&arena, activated, &l->output, 768, 3072, temp, y);
    add(x, y, context);
    normalize(context, &l->output_norm, x);
    TRACE((unsigned)layer + 1, x, 128u * 768u);
  }
  linear(&arena, x, &w->discriminator_dense, 768, 768, temp, y);
  MemRef2 Pred = make_2(y, 128, 768), ActPred = make_2(context, 128, 768);
  _mlir_ciface_kernel_gelu_128x768(&Pred, &ActPred);
  linear(&arena, context, &w->discriminator_prediction, 1, 768, temp, logits);
  TRACE(13, logits, 128);
  return 0;
}

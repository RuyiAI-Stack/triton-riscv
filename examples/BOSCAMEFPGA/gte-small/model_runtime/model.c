// Complete fixed-shape W8A8 model routing through the generated typed adapters.
// This file does not emulate any tensor operator on the target CPU.
#include "model.h"
#include "../support.h"
#ifndef GTE_MODEL_LAYERS
#define GTE_MODEL_LAYERS 12
#endif

#define EXTERN(name, ...) extern void _mlir_ciface_kernel_##name(__VA_ARGS__)
EXTERN(embedding_lookup_128x384, MemRef1 *, MemRef2 *, MemRef2 *);
EXTERN(embedding_add_128x384, MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *,
       MemRef2 *, MemRef2 *);
EXTERN(layernorm_128x384, MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);
EXTERN(add_128x384, MemRef2 *, MemRef2 *, MemRef2 *);
EXTERN(bias_add_128x384, MemRef2 *, MemRef1 *, MemRef2 *);
EXTERN(bias_add_128x1536, MemRef2 *, MemRef1 *, MemRef2 *);
EXTERN(gelu_128x1536, MemRef2 *, MemRef2 *);
EXTERN(split_heads_128x384, MemRef2 *, MemRef3 *);
EXTERN(merge_heads_12x128x32, MemRef3 *, MemRef2 *);
EXTERN(attention_mask_12x128x128, MemRef3 *, MemRef1 *, MemRef3 *);
EXTERN(softmax_12x128x128, MemRef3 *, MemRef3 *);
EXTERN(mean_pool_128x384, MemRef2 *, MemRef1 *, MemRef1 *);
EXTERN(l2_normalize_1x384, MemRef1 *, MemRef1 *);
#define QUANT(name) EXTERN(name, MemRef2 *, MemRef2 *, MemRef1 *)
QUANT(quantize_128x384);
QUANT(quantize_384x384);
QUANT(quantize_1536x384);
QUANT(quantize_128x1536);
QUANT(quantize_384x1536);
QUANT(quantize_128x32);
QUANT(quantize_128x128);
QUANT(quantize_transpose_128x32);
#define MAT(name) EXTERN(name, MemRef2 *, MemRef2 *, MemRef2 *)
MAT(matmul_i8_128x384x384);
MAT(matmul_i8_128x1536x384);
MAT(matmul_i8_128x384x1536);
MAT(matmul_i8_128x128x32);
MAT(matmul_i8_128x32x128);
#define DEQUANT(name) EXTERN(name, MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *)
DEQUANT(dequantize_128x384);
DEQUANT(dequantize_128x1536);
DEQUANT(dequantize_128x128);
DEQUANT(dequantize_128x32);
#define CALL(name, ...) _mlir_ciface_kernel_##name(__VA_ARGS__)
#ifdef GTE_MODEL_CHECKPOINTS
extern void gte_model_checkpoint(unsigned stage, const float *hidden);
#define CHECKPOINT(stage, hidden) gte_model_checkpoint(stage, hidden)
#else
#define CHECKPOINT(stage, hidden) ((void)0)
#endif

typedef struct {
  float *state[3], *qkv[3], *heads[4], *scores[2], *ffn[2], *linear;
  int8_t *left_q, *right_q;
  float *left_scale, *right_scale, *pooled;
  int32_t *acc;
} Buffers;

static size_t reserve(unsigned char *base, size_t *offset, size_t count,
                      size_t element_size, void **pointer) {
  *offset = (*offset + 63u) & ~(size_t)63u;
  if (pointer)
    *pointer = base ? base + *offset : NULL;
  *offset += count * element_size;
  return *offset;
}
static size_t bind_buffers(void *arena, Buffers *b) {
  size_t used = 0;
#define ALLOC(field, count, type)                                              \
  do {                                                                         \
    void *p;                                                                   \
    reserve(arena, &used, (count), sizeof(type), &p);                          \
    if (b)                                                                     \
      b->field = p;                                                            \
  } while (0)
  for (int i = 0; i < 3; i++) {
    ALLOC(state[i], 128u * 384u, float);
    ALLOC(qkv[i], 128u * 384u, float);
  }
  for (int i = 0; i < 4; i++) {
    ALLOC(heads[i], 12u * 128u * 32u, float);
  }
  for (int i = 0; i < 2; i++) {
    ALLOC(scores[i], 12u * 128u * 128u, float);
    ALLOC(ffn[i], 128u * 1536u, float);
  }
  ALLOC(linear, 128u * 1536u, float);
  ALLOC(left_q, 128u * 1536u, int8_t);
  ALLOC(right_q, 1536u * 384u, int8_t);
  ALLOC(left_scale, 128u, float);
  ALLOC(right_scale, 1536u, float);
  ALLOC(acc, 128u * 1536u, int32_t);
  ALLOC(pooled, 384u, float);
#undef ALLOC
  return (used + 63u) & ~(size_t)63u;
}
size_t gte_workspace_bytes(void) { return bind_buffers(NULL, NULL); }

static int aligned(const void *p) { return p && !((uintptr_t)p & 63u); }
static int affine_valid(const GTELinear *p) {
  return aligned(p->weight) && aligned(p->bias);
}
static int norm_valid(const GTELayerNorm *p) {
  return aligned(p->weight) && aligned(p->bias);
}
static int weights_valid(const GTEWeights *w) {
  if (!w || !aligned(w->word_embeddings) || !aligned(w->position_embeddings) ||
      !aligned(w->type_embeddings) || !norm_valid(&w->embedding_norm))
    return 0;
  for (int i = 0; i < 12; i++) {
    const GTELayer *l = &w->layers[i];
    if (!affine_valid(&l->query) || !affine_valid(&l->key) ||
        !affine_valid(&l->value) || !affine_valid(&l->attention_output) ||
        !affine_valid(&l->intermediate) || !affine_valid(&l->output) ||
        !norm_valid(&l->attention_norm) || !norm_valid(&l->output_norm))
      return 0;
  }
  return 1;
}
static void norm(float *input, const GTELayerNorm *n, float *output) {
  MemRef2 x = make_2(input, 128, 384), y = make_2(output, 128, 384);
  MemRef1 w = make_1((void *)n->weight, 384), b = make_1((void *)n->bias, 384);
  CALL(layernorm_128x384, &x, &w, &b, &y);
}
static void add(float *left, float *right, float *out) {
  MemRef2 a = make_2(left, 128, 384), b = make_2(right, 128, 384),
          c = make_2(out, 128, 384);
  CALL(add_128x384, &a, &b, &c);
}
// Dense dimensions choose one of three shape-specialized primitive chains.
static void linear(Buffers *b, float *input, const GTELinear *l, float *output,
                   int kind) {
  int n = kind == 1 ? 1536 : 384, k = kind == 2 ? 1536 : 384;
  MemRef2 x = make_2(input, 128, k), w = make_2((void *)l->weight, n, k);
  MemRef2 xq = make_2(b->left_q, 128, k), wq = make_2(b->right_q, n, k);
  MemRef1 xs = make_1(b->left_scale, 128), ws = make_1(b->right_scale, n);
  MemRef2 acc = make_2(b->acc, 128, n), raw = make_2(b->linear, 128, n),
          out = make_2(output, 128, n);
  MemRef1 bias = make_1((void *)l->bias, n);
  if (kind == 0) {
    CALL(quantize_128x384, &x, &xq, &xs);
    CALL(quantize_384x384, &w, &wq, &ws);
    CALL(matmul_i8_128x384x384, &xq, &wq, &acc);
    CALL(dequantize_128x384, &acc, &xs, &ws, &raw);
    CALL(bias_add_128x384, &raw, &bias, &out);
  } else if (kind == 1) {
    CALL(quantize_128x384, &x, &xq, &xs);
    CALL(quantize_1536x384, &w, &wq, &ws);
    CALL(matmul_i8_128x1536x384, &xq, &wq, &acc);
    CALL(dequantize_128x1536, &acc, &xs, &ws, &raw);
    CALL(bias_add_128x1536, &raw, &bias, &out);
  } else {
    CALL(quantize_128x1536, &x, &xq, &xs);
    CALL(quantize_384x1536, &w, &wq, &ws);
    CALL(matmul_i8_128x384x1536, &xq, &wq, &acc);
    CALL(dequantize_128x384, &acc, &xs, &ws, &raw);
    CALL(bias_add_128x384, &raw, &bias, &out);
  }
}
static void attention(Buffers *b, MemRef1 *mask) {
  for (int i = 0; i < 3; i++) {
    MemRef2 input = make_2(b->qkv[i], 128, 384);
    MemRef3 output = make_3(b->heads[i], 12, 128, 32);
    CALL(split_heads_128x384, &input, &output);
  }
  MemRef1 left_scale = make_1(b->left_scale, 128),
          right_scale = make_1(b->right_scale, 128);
  for (int h = 0; h < 12; h++) {
    MemRef2 q = make_2(b->heads[0] + h * 128 * 32, 128, 32),
            k = make_2(b->heads[1] + h * 128 * 32, 128, 32);
    MemRef2 qq = make_2(b->left_q, 128, 32), kq = make_2(b->right_q, 128, 32);
    MemRef2 acc = make_2(b->acc, 128, 128),
            score = make_2(b->scores[0] + h * 128 * 128, 128, 128);
    CALL(quantize_128x32, &q, &qq, &left_scale);
    CALL(quantize_128x32, &k, &kq, &right_scale);
    CALL(matmul_i8_128x128x32, &qq, &kq, &acc);
    CALL(dequantize_128x128, &acc, &left_scale, &right_scale, &score);
  }
  MemRef3 scores = make_3(b->scores[0], 12, 128, 128),
          masked = make_3(b->scores[1], 12, 128, 128);
  CALL(attention_mask_12x128x128, &scores, mask, &masked);
  CALL(softmax_12x128x128, &masked, &scores);
  for (int h = 0; h < 12; h++) {
    MemRef2 p = make_2(b->scores[0] + h * 128 * 128, 128, 128),
            v = make_2(b->heads[2] + h * 128 * 32, 128, 32);
    MemRef2 pq = make_2(b->left_q, 128, 128), vq = make_2(b->right_q, 32, 128);
    MemRef2 acc = make_2(b->acc, 128, 32),
            ctx = make_2(b->heads[3] + h * 128 * 32, 128, 32);
    right_scale = make_1(b->right_scale, 32);
    CALL(quantize_128x128, &p, &pq, &left_scale);
    CALL(quantize_transpose_128x32, &v, &vq, &right_scale);
    CALL(matmul_i8_128x32x128, &pq, &vq, &acc);
    CALL(dequantize_128x32, &acc, &left_scale, &right_scale, &ctx);
  }
  MemRef3 context = make_3(b->heads[3], 12, 128, 32);
  MemRef2 merged = make_2(b->qkv[0], 128, 384);
  CALL(merge_heads_12x128x32, &context, &merged);
}
int gte_infer(const GTEWeights *w, const int32_t *ids, const int32_t *mask,
              const int32_t *types, const int32_t *positions, void *arena,
              size_t bytes, float *embedding, float *last_hidden) {
  if (!weights_valid(w) || !aligned(ids) || !aligned(mask) || !aligned(types) ||
      !aligned(positions) || !aligned(arena) || !aligned(embedding) ||
      (last_hidden && !aligned(last_hidden)) || bytes < gte_workspace_bytes())
    return -1;
  int any = 0;
  for (int i = 0; i < 128; i++) {
    if (ids[i] < 0 || ids[i] >= 30522 || mask[i] < 0 || mask[i] > 1 ||
        types[i] < 0 || types[i] > 1 || positions[i] < 0 || positions[i] >= 512)
      return -1;
    any |= mask[i];
  }
  if (!any)
    return -1;
  Buffers b;
  bind_buffers(arena, &b);
  MemRef1 id = make_1((void *)ids, 128), m = make_1((void *)mask, 128),
          type = make_1((void *)types, 128),
          pos = make_1((void *)positions, 128);
  MemRef2 table = make_2((void *)w->word_embeddings, 30522, 384),
          word = make_2(b.state[0], 128, 384);
  CALL(embedding_lookup_128x384, &id, &table, &word);
  MemRef2 position = make_2((void *)w->position_embeddings, 512, 384),
          segment = make_2((void *)w->type_embeddings, 2, 384);
  MemRef2 sum = make_2(b.state[1], 128, 384);
  CALL(embedding_add_128x384, &word, &pos, &type, &position, &segment, &sum);
  norm(b.state[1], &w->embedding_norm, b.state[0]);
  CHECKPOINT(0, b.state[0]);
  for (int layer = 0; layer < GTE_MODEL_LAYERS; layer++) {
    const GTELayer *l = &w->layers[layer];
    linear(&b, b.state[0], &l->query, b.qkv[0], 0);
    linear(&b, b.state[0], &l->key, b.qkv[1], 0);
    linear(&b, b.state[0], &l->value, b.qkv[2], 0);
    attention(&b, &m);
    linear(&b, b.qkv[0], &l->attention_output, b.state[1], 0);
    add(b.state[0], b.state[1], b.state[2]);
    norm(b.state[2], &l->attention_norm, b.state[0]);
    linear(&b, b.state[0], &l->intermediate, b.ffn[0], 1);
    MemRef2 fi = make_2(b.ffn[0], 128, 1536), fo = make_2(b.ffn[1], 128, 1536);
    CALL(gelu_128x1536, &fi, &fo);
    linear(&b, b.ffn[1], &l->output, b.state[1], 2);
    add(b.state[0], b.state[1], b.state[2]);
    norm(b.state[2], &l->output_norm, b.state[0]);
    CHECKPOINT((unsigned)layer + 1, b.state[0]);
  }
  MemRef2 hidden = make_2(b.state[0], 128, 384);
  MemRef1 pooled = make_1(b.pooled, 384), result = make_1(embedding, 384);
  CALL(mean_pool_128x384, &hidden, &m, &pooled);
  CALL(l2_normalize_1x384, &pooled, &result);
  if (last_hidden)
    nr_copy_bytes(last_hidden, b.state[0], 128u * 384u * sizeof(float));
  return 0;
}

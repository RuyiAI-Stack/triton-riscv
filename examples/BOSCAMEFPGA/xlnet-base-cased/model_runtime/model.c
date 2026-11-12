// Fixed graph composition; no scalar replacement for any model operator.
#include "model.h"
extern void _mlir_ciface_kernel_add_128x768(MemRef2 *, MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_attention_score_12x128x128(MemRef3 *, MemRef3 *,
                                                           MemRef3 *, MemRef2 *,
                                                           MemRef3 *);
extern void _mlir_ciface_kernel_bias_add_128x1024(MemRef2 *, MemRef1 *,
                                                  MemRef2 *);
extern void _mlir_ciface_kernel_bias_add_128x3072(MemRef2 *, MemRef1 *,
                                                  MemRef2 *);
extern void _mlir_ciface_kernel_bias_add_128x768(MemRef2 *, MemRef1 *,
                                                 MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x1024(MemRef2 *, MemRef1 *,
                                                    MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x128(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x2(MemRef2 *, MemRef1 *,
                                                 MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x256(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x3072(MemRef2 *, MemRef1 *,
                                                    MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x64(MemRef2 *, MemRef1 *,
                                                  MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x768(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_256x768(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_embedding_128x768(MemRef1 *, MemRef2 *,
                                                  MemRef2 *);
extern void _mlir_ciface_kernel_gelu_128x3072(MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_head_bias_add_12x128x64(MemRef3 *, MemRef2 *,
                                                        MemRef3 *);
extern void _mlir_ciface_kernel_layernorm_128x768(MemRef2 *, MemRef1 *,
                                                  MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x1024x768(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x128x64(MemRef2 *, MemRef2 *,
                                                     MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x256x64(MemRef2 *, MemRef2 *,
                                                     MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x2x64(MemRef2 *, MemRef2 *,
                                                   MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x3072x768(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x64x128(MemRef2 *, MemRef2 *,
                                                     MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x768x3072(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x768x768(MemRef2 *, MemRef2 *,
                                                      MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_256x768x768(MemRef2 *, MemRef2 *,
                                                      MemRef2 *);
extern void _mlir_ciface_kernel_merge_heads_128x768(MemRef3 *, MemRef2 *);
extern void _mlir_ciface_kernel_quantize_1024x768(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x128(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x3072(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x64(MemRef2 *, MemRef2 *,
                                                MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x768(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_256x64(MemRef2 *, MemRef2 *,
                                                MemRef1 *);
extern void _mlir_ciface_kernel_quantize_256x768(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_2x64(MemRef2 *, MemRef2 *, MemRef1 *);
extern void _mlir_ciface_kernel_quantize_3072x768(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_768x3072(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_768x768(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_transpose_128x64(MemRef2 *, MemRef2 *,
                                                          MemRef1 *);
extern void _mlir_ciface_kernel_relative_shift_12x128x128(MemRef3 *, MemRef3 *);
extern void _mlir_ciface_kernel_segment_gather_12x128x128(MemRef3 *, MemRef3 *,
                                                          MemRef3 *);
extern void _mlir_ciface_kernel_softmax_12x128x128(MemRef3 *, MemRef3 *);
extern void _mlir_ciface_kernel_split_heads_12x128x64(MemRef2 *, MemRef3 *);
extern void _mlir_ciface_kernel_split_heads_12x256x64(MemRef2 *, MemRef3 *);
#define CALL(n, ...) _mlir_ciface_kernel_##n(__VA_ARGS__)
// Optional snapshot/progress hook. No numerical validation runs on E6.
__attribute__((weak)) void xlnet_layer_checkpoint(unsigned layer,
                                                  const float *hidden) {
  (void)layer;
  (void)hidden;
}
static void dense(int m, int n, int k, float *x, const XLNetMatrix *w,
                  float *out, int8_t *aq, float *as, int32_t *acc) {
  MemRef2 X = make_2(x, m, k), A = make_2(aq, m, k), B = make_2(w->q, n, k),
          C = make_2(acc, m, n), O = make_2(out, m, n);
  MemRef1 AS = make_1(as, m), BS = make_1(w->scale, n);
  if (k == 768 && m == 256)
    CALL(quantize_256x768, &X, &A, &AS);
  else if (k == 768)
    CALL(quantize_128x768, &X, &A, &AS);
  else if (k == 3072)
    CALL(quantize_128x3072, &X, &A, &AS);
  else if (k == 128)
    CALL(quantize_128x128, &X, &A, &AS);
  else
    CALL(quantize_128x64, &X, &A, &AS);
  if (m == 256) {
    CALL(matmul_i8_256x768x768, &A, &B, &C);
    CALL(dequantize_256x768, &C, &AS, &BS, &O);
  } else if (n == 3072) {
    CALL(matmul_i8_128x3072x768, &A, &B, &C);
    CALL(dequantize_128x3072, &C, &AS, &BS, &O);
  } else if (n == 1024) {
    CALL(matmul_i8_128x1024x768, &A, &B, &C);
    CALL(dequantize_128x1024, &C, &AS, &BS, &O);
  } else if (k == 3072) {
    CALL(matmul_i8_128x768x3072, &A, &B, &C);
    CALL(dequantize_128x768, &C, &AS, &BS, &O);
  } else if (n == 768) {
    CALL(matmul_i8_128x768x768, &A, &B, &C);
    CALL(dequantize_128x768, &C, &AS, &BS, &O);
  } else if (n == 256) {
    CALL(matmul_i8_128x256x64, &A, &B, &C);
    CALL(dequantize_128x256, &C, &AS, &BS, &O);
  } else if (n == 128) {
    CALL(matmul_i8_128x128x64, &A, &B, &C);
    CALL(dequantize_128x128, &C, &AS, &BS, &O);
  } else if (n == 2) {
    CALL(matmul_i8_128x2x64, &A, &B, &C);
    CALL(dequantize_128x2, &C, &AS, &BS, &O);
  } else {
    CALL(matmul_i8_128x64x128, &A, &B, &C);
    CALL(dequantize_128x64, &C, &AS, &BS, &O);
  }
}
static void addnorm(float *x, float *res, float *tmp, float *out, float *weight,
                    float *bias) {
  MemRef2 X = make_2(x, 128, 768), R = make_2(res, 128, 768),
          T = make_2(tmp, 128, 768), O = make_2(out, 128, 768);
  MemRef1 W = make_1(weight, 768), B = make_1(bias, 768);
  CALL(add_128x768, &X, &R, &T);
  CALL(layernorm_128x768, &T, &W, &B, &O);
}
static void headbias(float *q, float *bias, float *out) {
  MemRef3 Q = make_3(q, 12, 128, 64), O = make_3(out, 12, 128, 64);
  MemRef2 B = make_2(bias, 12, 64);
  CALL(head_bias_add_12x128x64, &Q, &B, &O);
}
static void split(float *x, float *out, int len) {
  MemRef2 X = make_2(x, len, 768);
  MemRef3 O = make_3(out, 12, len, 64);
  if (len == 256)
    CALL(split_heads_12x256x64, &X, &O);
  else
    CALL(split_heads_12x128x64, &X, &O);
}
int xlnet_forward(const XLNetWeights *weights, int32_t *ids, int32_t *mask,
                  int32_t *segments, void *scratch, size_t bytes, float *hidden,
                  float *logits) {
  if (!weights || !ids || !mask || !scratch || !hidden ||
      bytes < XLNET_MODEL_SCRATCH_BYTES || ((uintptr_t)scratch & 63u))
    return -1;
  for (int i = 0; i < 128; ++i)
    if (ids[i] < 0 || ids[i] >= 32000 || (mask[i] != 0 && mask[i] != 1) ||
        (segments && (segments[i] < 0 || segments[i] > 2)))
      return -2;
  size_t offset = 0;
#define ALLOC(type, name, count)                                               \
  type *name = (type *)((unsigned char *)scratch + offset);                    \
  offset = vit_align64(offset + (size_t)(count) * sizeof(type))
  ALLOC(float, h, 128 * 768);
  ALLOC(float, qn, 128 * 768);
  ALLOC(float, kn, 128 * 768);
  ALLOC(float, vn, 128 * 768);
  ALLOC(float, rn, 256 * 768);
  ALLOC(float, qh, 12 * 128 * 64);
  ALLOC(float, kh, 12 * 128 * 64);
  ALLOC(float, vh, 12 * 128 * 64);
  ALLOC(float, rh, 12 * 256 * 64);
  ALLOC(float, qbias, 12 * 128 * 64);
  ALLOC(float, ac, 12 * 128 * 128);
  ALLOC(float, bd_raw, 12 * 128 * 256);
  ALLOC(float, bd, 12 * 128 * 128);
  ALLOC(float, ef2, 12 * 128 * 2);
  ALLOC(float, ef, 12 * 128 * 128);
  ALLOC(float, score, 12 * 128 * 128);
  ALLOC(float, prob, 12 * 128 * 128);
  ALLOC(float, ctx, 12 * 128 * 64);
  ALLOC(float, merged, 128 * 768);
  ALLOC(float, proj, 128 * 768);
  ALLOC(float, tmp, 128 * 768);
  ALLOC(float, attn, 128 * 768);
  ALLOC(float, ff1, 128 * 3072);
  ALLOC(float, ffbias, 128 * 3072);
  ALLOC(float, ffgelu, 128 * 3072);
  ALLOC(float, ff2, 128 * 768);
  ALLOC(float, ff2bias, 128 * 768);
  ALLOC(int8_t, aq, 128 * 3072);
  ALLOC(float, as, 256);
  ALLOC(int32_t, acc, 128 * 3072);
  ALLOC(int8_t, bq, 256 * 64);
  ALLOC(float, bs, 256);
  ALLOC(int32_t, non_target, 128 * 128);
  ALLOC(float, seg_copy, 2 * 128 * 128);
  ALLOC(float, lm, 128 * 1024);
  ALLOC(float, lmb, 128 * 1024);
  if (offset > bytes)
    return -3;
  for (int i = 0; i < 128; ++i) {
    for (int j = 0; j < 128; ++j) {
      non_target[i * 128 + j] = mask[j] == 0 && i != j;
      if (segments) {
        seg_copy[i * 128 + j] = segments[i] == segments[j];
        seg_copy[16384 + i * 128 + j] = segments[i] != segments[j];
      }
    }
  }
  MemRef1 Ids = make_1(ids, 128);
  MemRef2 Emb = make_2(weights->embedding, 32000, 768), H = make_2(h, 128, 768);
  CALL(embedding_128x768, &Ids, &Emb, &H);
  xlnet_layer_checkpoint(0, h);
  for (int layer = 0; layer < 12; ++layer) {
    const XLNetLayer *w = &weights->layers[layer];
    dense(128, 768, 768, h, &w->q, qn, aq, as, acc);
    dense(128, 768, 768, h, &w->k, kn, aq, as, acc);
    dense(128, 768, 768, h, &w->v, vn, aq, as, acc);
    dense(256, 768, 768, weights->relative_positions, &w->r, rn, aq, as, acc);
    split(qn, qh, 128);
    split(kn, kh, 128);
    split(vn, vh, 128);
    split(rn, rh, 256);
    headbias(qh, w->rw_bias, qbias);
    for (int hd = 0; hd < 12; ++hd) {
      MemRef2 K = make_2(kh + hd * 128 * 64, 128, 64), QK = make_2(bq, 128, 64);
      MemRef1 S = make_1(bs, 128);
      CALL(quantize_128x64, &K, &QK, &S);
      XLNetMatrix right = {bq, bs};
      dense(128, 128, 64, qbias + hd * 128 * 64, &right, ac + hd * 128 * 128,
            aq, as, acc);
    }
    headbias(qh, w->rr_bias, qbias);
    for (int hd = 0; hd < 12; ++hd) {
      MemRef2 R = make_2(rh + hd * 256 * 64, 256, 64), QR = make_2(bq, 256, 64);
      MemRef1 S = make_1(bs, 256);
      CALL(quantize_256x64, &R, &QR, &S);
      XLNetMatrix right = {bq, bs};
      dense(128, 256, 64, qbias + hd * 128 * 64, &right,
            bd_raw + hd * 128 * 256, aq, as, acc);
    }
    MemRef3 BD = make_3(bd_raw, 12, 128, 256), Shift = make_3(bd, 12, 128, 128);
    CALL(relative_shift_12x128x128, &BD, &Shift);
    if (segments) {
      headbias(qh, w->rs_bias, qbias);
      for (int hd = 0; hd < 12; ++hd)
        dense(128, 2, 64, qbias + hd * 128 * 64, &w->segment[hd],
              ef2 + hd * 128 * 2, aq, as, acc);
      MemRef3 E2 = make_3(ef2, 12, 128, 2), E = make_3(ef, 12, 128, 128);
      MemRef3 Seg = make_3(seg_copy, 2, 128, 128);
      CALL(segment_gather_12x128x128, &E2, &Seg, &E);
    } else
      for (int i = 0; i < 12 * 128 * 128; ++i)
        ef[i] = 0;
    MemRef3 AC = make_3(ac, 12, 128, 128), EF = make_3(ef, 12, 128, 128),
            Score = make_3(score, 12, 128, 128), P = make_3(prob, 12, 128, 128);
    MemRef2 Mask = make_2(non_target, 128, 128);
    CALL(attention_score_12x128x128, &AC, &Shift, &EF, &Mask, &Score);
    CALL(softmax_12x128x128, &Score, &P);
    for (int hd = 0; hd < 12; ++hd) {
      MemRef2 V = make_2(vh + hd * 128 * 64, 128, 64), QV = make_2(bq, 64, 128);
      MemRef1 S = make_1(bs, 64);
      CALL(quantize_transpose_128x64, &V, &QV, &S);
      XLNetMatrix right = {bq, bs};
      dense(128, 64, 128, prob + hd * 128 * 128, &right, ctx + hd * 128 * 64,
            aq, as, acc);
    }
    MemRef3 Context = make_3(ctx, 12, 128, 64);
    MemRef2 Merged = make_2(merged, 128, 768);
    CALL(merge_heads_128x768, &Context, &Merged);
    dense(128, 768, 768, merged, &w->o, proj, aq, as, acc);
    addnorm(proj, h, tmp, attn, w->attn_norm_weight, w->attn_norm_bias);
    dense(128, 3072, 768, attn, &w->ff1, ff1, aq, as, acc);
    MemRef2 FF = make_2(ff1, 128, 3072), FB = make_2(ffbias, 128, 3072),
            FG = make_2(ffgelu, 128, 3072);
    MemRef1 B1 = make_1(w->ff1_bias, 3072);
    CALL(bias_add_128x3072, &FF, &B1, &FB);
    CALL(gelu_128x3072, &FB, &FG);
    dense(128, 768, 3072, ffgelu, &w->ff2, ff2, aq, as, acc);
    MemRef2 F2 = make_2(ff2, 128, 768), F2B = make_2(ff2bias, 128, 768);
    MemRef1 B2 = make_1(w->ff2_bias, 768);
    CALL(bias_add_128x768, &F2, &B2, &F2B);
    addnorm(ff2bias, attn, tmp, h, w->ffn_norm_weight, w->ffn_norm_bias);
    xlnet_layer_checkpoint((unsigned)layer + 1, h);
  }
  nr_copy_bytes(hidden, h, 128u * 768u * sizeof(float));
  if (logits)
    for (int tile = 0; tile < 32; ++tile) {
      dense(128, 1024, 768, h, &weights->lm_tiles[tile], lm, aq, as, acc);
      MemRef2 L = make_2(lm, 128, 1024), LB = make_2(lmb, 128, 1024);
      MemRef1 Bias = make_1(weights->lm_bias[tile], 1024);
      CALL(bias_add_128x1024, &L, &Bias, &LB);
      int valid = tile == 31 ? 256 : 1024;
      for (int row = 0; row < 128; ++row)
        nr_copy_bytes(logits + row * 32000 + tile * 1024, lmb + row * 1024,
                      (size_t)valid * sizeof(float));
    }
  return 0;
}

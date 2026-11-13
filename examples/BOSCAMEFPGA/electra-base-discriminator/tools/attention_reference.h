// Independent validation only: never linked into a typed kernel adapter.
// The FP32 baseline is host-only and cannot be a production model fallback.
//
#ifndef VIT_ATTENTION_REFERENCE_H
#define VIT_ATTENTION_REFERENCE_H
#include "support.h"
static inline float vit_reference_abs(float v) { return v < 0.0f ? -v : v; }
static inline int8_t vit_reference_i8(float x, float scale) {
  float v = x / scale;
  if (v > 127.0f)
    v = 127.0f;
  if (v < -127.0f)
    v = -127.0f;
  int whole = (int)v;
  float fraction = v - (float)whole;
  int rounded =
      whole + (vit_reference_abs(fraction) >= 0.5f ? (v >= 0.0f ? 1 : -1) : 0);
  return (int8_t)rounded;
}
static void vit_reference_rows(const float *x, int rows, int reduction,
                               int8_t *q, float *scales) {
  for (int i = 0; i < rows; ++i) {
    float maximum = 0.0f;
    for (int k = 0; k < reduction; ++k) {
      float v = vit_reference_abs(x[(size_t)i * reduction + k]);
      if (v > maximum)
        maximum = v;
    }
    float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    scales[i] = scale;
    for (int k = 0; k < reduction; ++k)
      q[(size_t)i * reduction + k] =
          vit_reference_i8(x[(size_t)i * reduction + k], scale);
  }
}
// Runtime B has the external [K,N] layout; AME requires packed [N,K].
static void vit_reference_columns(const float *x, int reduction, int columns,
                                  int8_t *q, float *scales) {
  for (int j = 0; j < columns; ++j) {
    float maximum = 0.0f;
    for (int k = 0; k < reduction; ++k) {
      float v = vit_reference_abs(x[(size_t)k * columns + j]);
      if (v > maximum)
        maximum = v;
    }
    float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    scales[j] = scale;
    for (int k = 0; k < reduction; ++k)
      q[(size_t)j * reduction + k] =
          vit_reference_i8(x[(size_t)k * columns + j], scale);
  }
}
#ifdef HOST_TEST
// Original scalar FP32 oracle; host accuracy baseline, never an FPGA kernel.
static float vit_reference_fp32_dot(const float *a, const float *b, int row,
                                    int col, int reduction, int columns) {
  float value = 0.0f;
  for (int k = 0; k < reduction; ++k)
    value += a[(size_t)row * reduction + k] * b[(size_t)k * columns + col];
  return value;
}
#endif
#endif

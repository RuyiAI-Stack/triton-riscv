// Isolated depthwise im2col candidate; original tile/production sources stay.
// Copy each tile's valid input bounding rectangle once into CPU-owned scalar
// stack storage under the uncached read policy. Under the write policy, reuse
// that snapshot for scalar max, exact f32 quantization, and QPatch/scales.
// No RVV scratch stores/reductions or arithmetic approximations are used.
// A wrapped tile uses the whole input width. If that rectangle would read at
// least as many floats as the original gather, retain the gather snapshot.
// Ordinary convolution keeps the original CPU-owned gather algorithm.
//
#include <stddef.h>
#include <stdint.h>

#ifndef IM2COL_ENTRY
#define IM2COL_ENTRY efficientnet_im2col_tile_cached_candidate
#endif
#define IM2COL_K (DEPTHWISE ? KH * KW : C * KH * KW)
// An arbitrary 64-position interval crosses at most ceil(63/OW) row edges.
#define TILE_INPUT_ROWS (((63 + OW - 1) / OW) * SH + KH)
typedef struct {
  void *allocated, *aligned;
  int64_t offset, size, stride;
} Descriptor1D;

static inline void read_phase(void) {
#ifdef __riscv
  uintptr_t policy = UINT64_C(0x0f0f000f000f000f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
#endif
}
static inline void write_phase(void) {
#ifdef __riscv
  uintptr_t policy = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
#endif
}
static inline int8_t quantize(float value, float scale) {
  float ratio = value / scale;
  if (ratio < -127.0f)
    ratio = -127.0f;
  if (ratio > 127.0f)
    ratio = 127.0f;
  int32_t base = (int32_t)ratio;
  float fraction = ratio - (float)base;
  if (__builtin_fabsf(fraction) >= 0.5f)
    base += ratio >= 0.0f ? 1 : -1;
  return (int8_t)base;
}

static void gather_snapshot(const float *input, int8_t *quantized,
                            float *output_scale, int32_t tile, int32_t group) {
  float values[64][IM2COL_K], scales[64];
  read_phase();
  for (size_t lane = 0; lane < 64; ++lane) {
    int32_t position = tile * 64 + (int32_t)lane;
    int32_t row = position / OW * SH - PH;
    int32_t col = position % OW * SW - PW;
    float maximum = 0.0f;
    for (size_t k = 0; k < IM2COL_K; ++k) {
      int32_t channel = DEPTHWISE ? group : (int32_t)k / (KH * KW);
      int32_t ih = row + (int32_t)k / KW % KH;
      int32_t iw = col + (int32_t)k % KW;
      float value = 0.0f;
      if (position < OH * OW && ih >= 0 && ih < H && iw >= 0 && iw < W)
        value = input[channel * H * W + ih * W + iw];
      values[lane][k] = value;
      float magnitude = __builtin_fabsf(value);
      if (magnitude > maximum)
        maximum = magnitude;
    }
    scales[lane] = maximum == 0.0f ? 1.0f : maximum / 127.0f;
  }
  write_phase();
  for (size_t lane = 0; lane < 64; ++lane) {
    for (size_t k = 0; k < IM2COL_K; ++k)
      quantized[lane * 64 + k] = quantize(values[lane][k], scales[lane]);
    output_scale[lane] = scales[lane];
  }
}

void IM2COL_ENTRY(int64_t rank_x, const Descriptor1D *x_desc, int64_t rank_q,
                  const Descriptor1D *q_desc, int64_t rank_s,
                  const Descriptor1D *s_desc, int32_t grid_x, int32_t grid_y,
                  int32_t grid_z, int32_t tile, int32_t unused_y,
                  int32_t group) {
  (void)rank_x;
  (void)rank_q;
  (void)rank_s;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  (void)unused_y;
  const float *input = (const float *)x_desc->aligned + x_desc->offset;
  int8_t *quantized = (int8_t *)q_desc->aligned + q_desc->offset;
  float *output_scale = (float *)s_desc->aligned + s_desc->offset;
#if DEPTHWISE
  int32_t first = tile * 64;
  int32_t last = first + 63;
  if (last >= OH * OW)
    last = OH * OW - 1;
  int32_t first_row = first / OW, last_row = last / OW;
  int32_t top = first_row * SH - PH;
  int32_t bottom = last_row * SH - PH + KH;
  int32_t left = first_row == last_row ? first % OW * SW - PW : 0;
  int32_t right = first_row == last_row ? last % OW * SW - PW + KW : W;
  if (top < 0)
    top = 0;
  if (bottom > H)
    bottom = H;
  if (left < 0)
    left = 0;
  if (right > W)
    right = W;
  int32_t height = bottom - top, width = right - left;
  if (first < OH * OW && height > 0 && width > 0 &&
      height * width < 64 * IM2COL_K) {
    float cached[TILE_INPUT_ROWS * W];
    const volatile float *source = input + group * H * W;
    read_phase();
    for (int32_t row = 0; row < height; ++row)
      for (int32_t col = 0; col < width; ++col)
        cached[row * width + col] = source[(top + row) * W + left + col];
    write_phase();
    for (int32_t lane = 0; lane < 64; ++lane) {
      int32_t position = first + lane;
      int32_t row = position / OW * SH - PH;
      int32_t col = position % OW * SW - PW;
      float maximum = 0.0f;
      for (int32_t k = 0; k < IM2COL_K; ++k) {
        int32_t ih = row + k / KW, iw = col + k % KW;
        float value = 0.0f;
        if (position < OH * OW && ih >= 0 && ih < H && iw >= 0 && iw < W)
          value = cached[(ih - top) * width + iw - left];
        float magnitude = __builtin_fabsf(value);
        if (magnitude > maximum)
          maximum = magnitude;
      }
      float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
      for (int32_t k = 0; k < IM2COL_K; ++k) {
        int32_t ih = row + k / KW, iw = col + k % KW;
        float value = 0.0f;
        if (position < OH * OW && ih >= 0 && ih < H && iw >= 0 && iw < W)
          value = cached[(ih - top) * width + iw - left];
        quantized[lane * 64 + k] = quantize(value, scale);
      }
      output_scale[lane] = scale;
    }
  } else {
    gather_snapshot(input, quantized, output_scale, tile, group);
  }
#else
  gather_snapshot(input, quantized, output_scale, tile, group);
#endif
#ifdef __riscv
  __asm__ volatile("fence iorw,iorw" ::: "memory");
#endif
}

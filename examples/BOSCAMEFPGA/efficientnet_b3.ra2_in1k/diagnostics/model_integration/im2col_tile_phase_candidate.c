// Isolated im2col preparation: CPU owns all temporary storage.
// Read all taps under the uncached DDR read policy, then quantize only the
// CPU-owned snapshot and publish QPatch/scales under the write policy.
// Max and all arithmetic remain scalar. Integer MMA is supplied unchanged
// by the original convolution object. Finite FP32 input contract only.
//
#include <stddef.h>
#include <stdint.h>

#ifndef IM2COL_ENTRY
#define IM2COL_ENTRY efficientnet_im2col_tile_phase_candidate
#endif
#define IM2COL_K (DEPTHWISE ? KH * KW : C * KH * KW)
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
    float scale = scales[lane];
    for (size_t k = 0; k < IM2COL_K; ++k) {
      float ratio = values[lane][k] / scale;
      if (ratio < -127.0f)
        ratio = -127.0f;
      if (ratio > 127.0f)
        ratio = 127.0f;
      int32_t base = (int32_t)ratio;
      float fraction = ratio - (float)base;
      if (__builtin_fabsf(fraction) >= 0.5f)
        base += ratio >= 0.0f ? 1 : -1;
      quantized[lane * 64 + k] = (int8_t)base;
    }
    output_scale[lane] = scale;
  }
#ifdef __riscv
  __asm__ volatile("fence iorw,iorw" ::: "memory");
#endif
}

// Isolated K3/S1 depthwise im2col experiment. No production binding.
// Four spatial patches occupy four e32,m1 lanes; nine taps stay in registers.
// No vector reductions, strided memory or vector-written stack scratch.
// QPatch[0][16:52] is temporary tap-major storage, restored to zero before
// return. CPU reads it only under the uncached read policy into CPU-owned
// storage, then scatters final patch-major bytes under the write policy.
//
#include <stddef.h>
#include <stdint.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

#ifndef IM2COL_ENTRY
#define IM2COL_ENTRY efficientnet_im2col_spatial_vl4_candidate
#endif
#if !DEPTHWISE || KH != 3 || KW != 3 || SH != 1 || SW != 1
#error "This isolated candidate supports only depthwise K3/S1"
#endif

typedef struct {
  void *allocated, *aligned;
  int64_t offset, size, stride;
} Descriptor1D;
typedef uint32_t AliasU32 __attribute__((__may_alias__));

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
static inline float scalar_max(float a, float b) {
#ifdef __riscv
  float r;
  __asm__("fmax.s %0,%1,%2" : "=f"(r) : "f"(a), "f"(b));
  return r;
#else
  return a > b ? a : b;
#endif
}
static inline float scalar_min(float a, float b) {
#ifdef __riscv
  float r;
  __asm__("fmin.s %0,%1,%2" : "=f"(r) : "f"(a), "f"(b));
  return r;
#else
  return a < b ? a : b;
#endif
}
static inline int8_t quantize_scalar(float value, float scale) {
  float ratio = value / scale;
  ratio = scalar_max(ratio, -127.0f);
  ratio = scalar_min(ratio, 127.0f);
  int32_t base = (int32_t)ratio;
  float fraction = ratio - (float)base;
  if (__builtin_fabsf(fraction) >= 0.5f)
    base += ratio >= 0.0f ? 1 : -1;
  return (int8_t)base;
}

static void scalar_patch(const float *input, int8_t *q, float *ps,
                         int32_t position, int32_t lane, int32_t group) {
  float values[9], maximum = 0.0f;
  int32_t row = position / OW - PH, col = position % OW - PW;
  read_phase();
#pragma clang loop unroll(full)
  for (int32_t k = 0; k < 9; ++k) {
    int32_t ih = row + k / 3, iw = col + k % 3;
    float value = 0.0f;
    if (position < OH * OW && ih >= 0 && ih < H && iw >= 0 && iw < W)
      value = input[group * H * W + ih * W + iw];
    values[k] = value;
    maximum = scalar_max(maximum, __builtin_fabsf(value));
  }
  float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
  write_phase();
#pragma clang loop unroll(full)
  for (int32_t k = 0; k < 9; ++k)
    q[lane * 64 + k] = quantize_scalar(values[k], scale);
  ps[lane] = scale;
}

#ifdef __riscv_vector
__attribute__((always_inline)) static inline vint8mf4_t
quantize_vector(vfloat32m1_t value, vfloat32m1_t scale) {
  const size_t vl = 4;
  vfloat32m1_t ratio = __riscv_vfdiv_vv_f32m1(value, scale, vl);
  ratio = __riscv_vfmax_vf_f32m1(ratio, -127.0f, vl);
  ratio = __riscv_vfmin_vf_f32m1(ratio, 127.0f, vl);
  vint32m1_t base = __riscv_vfcvt_rtz_x_f_v_i32m1(ratio, vl);
  vfloat32m1_t fraction =
      __riscv_vfsub_vv_f32m1(ratio, __riscv_vfcvt_f_x_v_f32m1(base, vl), vl);
  vbool32_t positive = __riscv_vmfge_vf_f32m1_b32(ratio, 0.0f, vl);
  vint32m1_t direction =
      __riscv_vmerge_vxm_i32m1(__riscv_vmv_v_x_i32m1(-1, vl), 1, positive, vl);
  vbool32_t rounded =
      __riscv_vmfge_vf_f32m1_b32(__riscv_vfabs_v_f32m1(fraction, vl), 0.5f, vl);
  base = __riscv_vadd_vv_i32m1_mu(rounded, base, base, direction, vl);
  vint16mf2_t halves = __riscv_vnsra_wx_i16mf2(base, 0, vl);
  return __riscv_vnsra_wx_i8mf4(halves, 0, vl);
}
#endif

__attribute__((always_inline)) static inline void
spatial_batch(const float *input, int8_t *q, float *ps, int32_t position,
              int32_t lane, int32_t group) {
  uint8_t *temporary = (uint8_t *)q + 16;
  int32_t row = position / OW - PH, col = position % OW - PW;
  const float *start = input + group * H * W + row * W + col;
  read_phase();
#ifdef __riscv_vector
  const size_t vl = 4;
  vfloat32m1_t a0 = __riscv_vle32_v_f32m1(start + 0 * W + 0, vl);
  vfloat32m1_t a1 = __riscv_vle32_v_f32m1(start + 0 * W + 1, vl);
  vfloat32m1_t a2 = __riscv_vle32_v_f32m1(start + 0 * W + 2, vl);
  vfloat32m1_t a3 = __riscv_vle32_v_f32m1(start + 1 * W + 0, vl);
  vfloat32m1_t a4 = __riscv_vle32_v_f32m1(start + 1 * W + 1, vl);
  vfloat32m1_t a5 = __riscv_vle32_v_f32m1(start + 1 * W + 2, vl);
  vfloat32m1_t a6 = __riscv_vle32_v_f32m1(start + 2 * W + 0, vl);
  vfloat32m1_t a7 = __riscv_vle32_v_f32m1(start + 2 * W + 1, vl);
  vfloat32m1_t a8 = __riscv_vle32_v_f32m1(start + 2 * W + 2, vl);
  vfloat32m1_t maximum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a0, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a1, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a2, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a3, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a4, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a5, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a6, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a7, vl), vl);
  maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a8, vl), vl);
  vfloat32m1_t scale = __riscv_vfdiv_vf_f32m1(maximum, 127.0f, vl);
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vbool32_t is_zero = __riscv_vmfge_vf_f32m1_b32(
      __riscv_vfsub_vv_f32m1(zero, maximum, vl), 0.0f, vl);
  vint32m1_t scale_bits = __riscv_vreinterpret_v_f32m1_i32m1(scale);
  scale_bits = __riscv_vmerge_vxm_i32m1(scale_bits, 0x3f800000, is_zero, vl);
  scale = __riscv_vreinterpret_v_i32m1_f32m1(scale_bits);
  write_phase();
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 0 * 4, quantize_vector(a0, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 1 * 4, quantize_vector(a1, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 2 * 4, quantize_vector(a2, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 3 * 4, quantize_vector(a3, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 4 * 4, quantize_vector(a4, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 5 * 4, quantize_vector(a5, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 6 * 4, quantize_vector(a6, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 7 * 4, quantize_vector(a7, scale),
                       vl);
  __riscv_vse8_v_i8mf4((int8_t *)temporary + 8 * 4, quantize_vector(a8, scale),
                       vl);
  __riscv_vse32_v_f32m1(ps + lane, scale, vl);
#else
  // Host emulation exercises the same temporary alias/transpose protocol.
  float values[9][4], scales[4];
  for (int32_t j = 0; j < 4; ++j) {
    float maximum = 0.0f;
    for (int32_t k = 0; k < 9; ++k) {
      float value = start[(k / 3) * W + k % 3 + j];
      values[k][j] = value;
      maximum = scalar_max(maximum, __builtin_fabsf(value));
    }
    scales[j] = maximum == 0.0f ? 1.0f : maximum / 127.0f;
  }
  write_phase();
  for (int32_t k = 0; k < 9; ++k)
    for (int32_t j = 0; j < 4; ++j)
      temporary[k * 4 + j] = (uint8_t)quantize_scalar(values[k][j], scales[j]);
  for (int32_t j = 0; j < 4; ++j)
    ps[lane + j] = scales[j];
#endif
  union {
    uint32_t words[9];
    uint8_t bytes[36];
  } snapshot;
  const volatile AliasU32 *source = (const volatile AliasU32 *)temporary;
  read_phase();
#pragma clang loop unroll(full)
  for (int32_t k = 0; k < 9; ++k)
    snapshot.words[k] = source[k];
  write_phase();
  volatile AliasU32 *clear = (volatile AliasU32 *)temporary;
#pragma clang loop unroll(full)
  for (int32_t k = 0; k < 9; ++k)
    clear[k] = 0;
#pragma clang loop unroll(full)
  for (int32_t j = 0; j < 4; ++j) {
#pragma clang loop unroll(full)
    for (int32_t k = 0; k < 9; ++k)
      q[(lane + j) * 64 + k] = (int8_t)snapshot.bytes[k * 4 + j];
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
  int8_t *q = (int8_t *)q_desc->aligned + q_desc->offset;
  float *ps = (float *)s_desc->aligned + s_desc->offset;
  int32_t lane = 0;
  while (lane < 64) {
    int32_t position = tile * 64 + lane;
    int32_t row = position / OW - PH, col = position % OW - PW;
    int eligible = lane <= 60 && position + 3 < OH * OW &&
                   position / OW == (position + 3) / OW && row >= 0 &&
                   row + 2 < H && col >= 0 && col + 5 < W;
#ifdef __riscv_vector
    if (eligible)
      eligible = __riscv_vsetvl_e32m1(4) == 4;
#endif
    if (eligible) {
      spatial_batch(input, q, ps, position, lane, group);
      lane += 4;
    } else {
      scalar_patch(input, q, ps, position, lane, group);
      ++lane;
    }
  }
#ifdef __riscv
  __asm__ volatile("fence iorw,iorw" ::: "memory");
#endif
}

// Final isolated K3/S1 depthwise tile-snapshot + spatial VL4 experiment.
// CPU gathers all 64x9 taps into CPU-owned tap-major float stack storage.
// RVV reads that stack (never writes it), and publishes tap-major Q to public
// QPatch[0:576] plus contiguous scales. CPU copies every temporary byte under
// READ policy into CPU-owned byte storage before clearing and transposing.
// Four policy phases per tile; all padding restored before returning. No
// production binding, vector reduction, strided vector memory or ISA widening.
//
#include <stddef.h>
#include <stdint.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

#ifndef IM2COL_ENTRY
#define IM2COL_ENTRY efficientnet_im2col_tile_spatial_vl4_candidate
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
  float values[9][64];
  union {
    uint32_t words[144];
    uint8_t bytes[576];
  } snapshot;
  read_phase();
  for (int32_t lane = 0; lane < 64; ++lane) {
    int32_t position = tile * 64 + lane;
    int32_t row = position / OW - PH, col = position % OW - PW;
#pragma clang loop unroll(full)
    for (int32_t k = 0; k < 9; ++k) {
      int32_t ih = row + k / 3, iw = col + k % 3;
      float value = 0.0f;
      if (position < OH * OW && ih >= 0 && ih < H && iw >= 0 && iw < W)
        value = input[group * H * W + ih * W + iw];
      values[k][lane] = value;
    }
  }
  // Publishes CPU-owned stack writes before RVV reads. Stack entry6 has the
  // same 0x0f policy in READ and WRITE; only RVV reads this float snapshot.
  write_phase();
#ifdef __riscv_vector
  for (int32_t lane = 0; lane < 64; lane += 4) {
    const size_t vl = 4;
    vfloat32m1_t a0 = __riscv_vle32_v_f32m1(values[0] + lane, vl);
    vfloat32m1_t a1 = __riscv_vle32_v_f32m1(values[1] + lane, vl);
    vfloat32m1_t a2 = __riscv_vle32_v_f32m1(values[2] + lane, vl);
    vfloat32m1_t a3 = __riscv_vle32_v_f32m1(values[3] + lane, vl);
    vfloat32m1_t a4 = __riscv_vle32_v_f32m1(values[4] + lane, vl);
    vfloat32m1_t a5 = __riscv_vle32_v_f32m1(values[5] + lane, vl);
    vfloat32m1_t a6 = __riscv_vle32_v_f32m1(values[6] + lane, vl);
    vfloat32m1_t a7 = __riscv_vle32_v_f32m1(values[7] + lane, vl);
    vfloat32m1_t a8 = __riscv_vle32_v_f32m1(values[8] + lane, vl);
    vfloat32m1_t maximum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a0, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a1, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a2, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a3, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a4, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a5, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a6, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a7, vl), vl);
    maximum =
        __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(a8, vl), vl);
    vfloat32m1_t scale = __riscv_vfdiv_vf_f32m1(maximum, 127.0f, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vbool32_t is_zero = __riscv_vmfge_vf_f32m1_b32(
        __riscv_vfsub_vv_f32m1(zero, maximum, vl), 0.0f, vl);
    vint32m1_t scale_bits = __riscv_vreinterpret_v_f32m1_i32m1(scale);
    scale_bits = __riscv_vmerge_vxm_i32m1(scale_bits, 0x3f800000, is_zero, vl);
    scale = __riscv_vreinterpret_v_i32m1_f32m1(scale_bits);
    __riscv_vse8_v_i8mf4((int8_t *)q + 0 * 64 + lane,
                         quantize_vector(a0, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 1 * 64 + lane,
                         quantize_vector(a1, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 2 * 64 + lane,
                         quantize_vector(a2, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 3 * 64 + lane,
                         quantize_vector(a3, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 4 * 64 + lane,
                         quantize_vector(a4, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 5 * 64 + lane,
                         quantize_vector(a5, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 6 * 64 + lane,
                         quantize_vector(a6, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 7 * 64 + lane,
                         quantize_vector(a7, scale), vl);
    __riscv_vse8_v_i8mf4((int8_t *)q + 8 * 64 + lane,
                         quantize_vector(a8, scale), vl);
    __riscv_vse32_v_f32m1(ps + lane, scale, vl);
  }
#else
  // Arithmetic/alias host emulation; separate board proof executes RVV.
  for (int32_t lane = 0; lane < 64; ++lane) {
    float maximum = 0.0f;
    for (int32_t k = 0; k < 9; ++k)
      maximum = scalar_max(maximum, __builtin_fabsf(values[k][lane]));
    float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    for (int32_t k = 0; k < 9; ++k)
      q[k * 64 + lane] = quantize_scalar(values[k][lane], scale);
    ps[lane] = scale;
  }
#endif
  // Drain vector Q stores, then copy the COMPLETE alias before any clear or
  // final scatter; CPU never reads vector-written stack data.
  read_phase();
  const volatile AliasU32 *temporary = (const volatile AliasU32 *)q;
  for (int32_t k = 0; k < 144; ++k)
    snapshot.words[k] = temporary[k];
  write_phase();
  volatile AliasU32 *clear = (volatile AliasU32 *)q;
  for (int32_t k = 0; k < 144; ++k)
    clear[k] = 0;
  for (int32_t lane = 0; lane < 64; ++lane) {
#pragma clang loop unroll(full)
    for (int32_t k = 0; k < 9; ++k)
      q[lane * 64 + k] = (int8_t)snapshot.bytes[k * 64 + lane];
  }
#ifdef __riscv
  __asm__ volatile("fence iorw,iorw" ::: "memory");
#endif
}

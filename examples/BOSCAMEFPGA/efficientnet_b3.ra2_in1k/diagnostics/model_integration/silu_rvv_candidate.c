// Independent integration candidate, not a production kernel replacement.
// Reuses the exact binary32 exp coefficients/thresholds from
// BOSCAMEMathApprox{,Coefficients}.h, with separate multiply/add operations.
// Restricted NR lacks verified vector bit shifts/comparisons: range reduction
// and exponent construction stay scalar; Horner/scale/SiLU use RVV.
//
#include <stddef.h>
#include <stdint.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

static const float coefficients[8] = {
    0x1p+0f,        0x1p+0f,        0x1p-1f,         0x1.555556p-3f,
    0x1.555556p-5f, 0x1.111112p-7f, 0x1.6c16c2p-10f, 0x1.a01a02p-13f};
static inline float pow2_bits(int exponent) {
  union {
    uint32_t u;
    float f;
  } value = {(uint32_t)(exponent + 127) << 23};
  return value.f;
}
static inline float reduced(float input, float *scale1, float *scale2) {
  float x = input == input ? input : 0.0f;
  if (x < -104.0f)
    x = -104.0f;
  if (x > 89.0f)
    x = 89.0f;
  float rounded = x * 0x1.715476p+0f;
  rounded = rounded + (x < 0.0f ? -0.5f : 0.5f);
  int k = (int)rounded;
  float r = x - (float)k * 0x1.63p-1f;
  r = r - (float)k * (-0x1.bd0106p-13f);
  int k1 = k < -126 ? -126 : k;
  if (k1 > 127)
    k1 = 127;
  *scale1 = pow2_bits(k1);
  *scale2 = pow2_bits(k - k1);
  return r;
}
static inline float restore(float x, float result) {
  if (x != x)
    return x;
  if (x == -__builtin_inff())
    return x - x;
  if (-x >= 88.72283935546875f)
    return -0.0f;
  if (-x <= -103.97208404541015625f)
    return x;
  return result;
}
#ifdef __riscv_vector
static inline void read_phase(void) {
  uintptr_t policy = UINT64_C(0x0f0f000f000f000f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
}
static inline void write_phase(void) {
  uintptr_t policy = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
}
#endif

void efficientnet_silu_rvv_candidate(const float *input, float *output,
                                     size_t count) {
  for (size_t base = 0; base < count;) {
    size_t vl = count - base < 16 ? count - base : 16;
    float values[16], r[16], scale1[16], scale2[16], results[16];
#ifdef __riscv_vector
    vl = __riscv_vsetvl_e32m1(vl);
    read_phase();
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(input + base, vl);
    __riscv_vse32_v_f32m1(values, vx, vl);
#else
    for (size_t lane = 0; lane < vl; ++lane)
      values[lane] = input[base + lane];
#endif
    for (size_t lane = 0; lane < vl; ++lane)
      r[lane] = reduced(-values[lane], &scale1[lane], &scale2[lane]);
#ifdef __riscv_vector
    vfloat32m1_t vr = __riscv_vle32_v_f32m1(r, vl);
    vfloat32m1_t vp = __riscv_vfmv_v_f_f32m1(coefficients[7], vl);
    for (int coefficient = 6; coefficient >= 0; --coefficient) {
      vp = __riscv_vfmul_vv_f32m1(vp, vr, vl);
      vp = __riscv_vfadd_vf_f32m1(vp, coefficients[coefficient], vl);
    }
    vp = __riscv_vfmul_vv_f32m1(vp, __riscv_vle32_v_f32m1(scale2, vl), vl);
    vp = __riscv_vfmul_vv_f32m1(vp, __riscv_vle32_v_f32m1(scale1, vl), vl);
    vp = __riscv_vfadd_vf_f32m1(vp, 1.0f, vl);
    vx = __riscv_vle32_v_f32m1(values, vl);
    vp = __riscv_vfdiv_vv_f32m1(vx, vp, vl);
    __riscv_vse32_v_f32m1(results, vp, vl);
#else
    for (size_t lane = 0; lane < vl; ++lane) {
      float p = coefficients[7];
      for (int coefficient = 6; coefficient >= 0; --coefficient) {
        p = p * r[lane];
        p = p + coefficients[coefficient];
      }
      p = p * scale2[lane];
      p = p * scale1[lane];
      p = p + 1.0f;
      results[lane] = values[lane] / p;
    }
#endif
    for (size_t lane = 0; lane < vl; ++lane)
      results[lane] = restore(values[lane], results[lane]);
#ifdef __riscv_vector
    vx = __riscv_vle32_v_f32m1(results, vl);
    write_phase();
    __riscv_vse32_v_f32m1(output + base, vx, vl);
#else
    for (size_t lane = 0; lane < vl; ++lane)
      output[base + lane] = results[lane];
#endif
    base += vl;
  }
}

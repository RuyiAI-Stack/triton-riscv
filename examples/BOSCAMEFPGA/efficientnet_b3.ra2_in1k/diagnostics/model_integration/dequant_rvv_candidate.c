// Independent NCHW dequantization connector candidate.
// Keep int32->f32, multiply weight scale, multiply activation scale, add bias
// in the original order. Batch DDR phases and amortize the public raw ABI.
//
#include <stddef.h>
#include <stdint.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
static inline void read_phase(void) {
  uintptr_t policy = UINT64_C(0x0f0f000f000f000f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
}
static inline void write_phase(void) {
  uintptr_t policy = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
}
#endif

void efficientnet_dequant_rvv_candidate(const int32_t *accumulator,
                                        const float *activation_scale,
                                        const float *weight_scale,
                                        const float *bias, float *output,
                                        size_t channels, size_t spatial) {
  for (size_t channel = 0; channel < channels; ++channel) {
    for (size_t position = 0; position < spatial;) {
      size_t vl = spatial - position < 16 ? spatial - position : 16;
      int32_t gathered[16];
#ifdef __riscv_vector
      vl = __riscv_vsetvl_e32m1(vl);
      read_phase();
#endif
      float ws = weight_scale[channel], b = bias[channel];
      for (size_t lane = 0; lane < vl; ++lane)
        gathered[lane] = accumulator[(position + lane) * channels + channel];
#ifdef __riscv_vector
      vint32m1_t integers = __riscv_vle32_v_i32m1(gathered, vl);
      vfloat32m1_t values = __riscv_vfcvt_f_x_v_f32m1(integers, vl);
      values = __riscv_vfmul_vf_f32m1(values, ws, vl);
      vfloat32m1_t scales =
          __riscv_vle32_v_f32m1(activation_scale + position, vl);
      values = __riscv_vfmul_vv_f32m1(values, scales, vl);
      values = __riscv_vfadd_vf_f32m1(values, b, vl);
      write_phase();
      __riscv_vse32_v_f32m1(output + channel * spatial + position, values, vl);
#else
      for (size_t lane = 0; lane < vl; ++lane) {
        float value = (float)gathered[lane];
        value = value * ws;
        value = value * activation_scale[position + lane];
        value = value + b;
        output[channel * spatial + position + lane] = value;
      }
#endif
      position += vl;
    }
  }
}

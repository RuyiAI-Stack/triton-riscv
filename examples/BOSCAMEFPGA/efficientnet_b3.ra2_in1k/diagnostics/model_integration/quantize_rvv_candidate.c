// Independent NCHW -> [spatial, channel] W8A8 pack candidate.
#include <stddef.h>
#include <stdint.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

static inline void read_phase(void) {
#if defined(__riscv_vector)
  uintptr_t policy = UINT64_C(0x0f0f000f000f000f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
#endif
}

static inline void write_phase(void) {
#if defined(__riscv_vector)
  uintptr_t policy = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
#endif
}

static inline float abs_f32(float value) {
  return value < 0.0f ? -value : value;
}

static inline int8_t quantize_one(float value, float scale) {
  float ratio = value / scale;
  if (ratio < -127.0f)
    ratio = -127.0f;
  if (ratio > 127.0f)
    ratio = 127.0f;
  int32_t base = (int32_t)ratio;
  float fraction = ratio - (float)base;
  if (abs_f32(fraction) >= 0.5f)
    base += ratio >= 0.0f ? 1 : -1;
  return (int8_t)base;
}

void efficientnet_quantize_pack_candidate(const float *input, int8_t *packed,
                                          float *activation_scale,
                                          size_t channels, size_t spatial) {
  for (size_t position = 0; position < spatial; ++position) {
    float maximum = 0.0f;
#if defined(EFFICIENTNET_QUANTIZE_SCALAR)
    read_phase();
    for (size_t channel = 0; channel < channels; ++channel) {
      float value = input[channel * spatial + position];
      float magnitude = abs_f32(value);
      if (magnitude > maximum)
        maximum = magnitude;
    }
#else
    read_phase();
    for (size_t channel = 0; channel < channels; ++channel) {
      float magnitude = abs_f32(input[channel * spatial + position]);
      if (magnitude > maximum)
        maximum = magnitude;
    }
#endif
    float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;

#if defined(__riscv_vector) && !defined(EFFICIENTNET_QUANTIZE_SCALAR)
    for (size_t channel = 0; channel < channels;) {
      size_t vl = __riscv_vsetvl_e32m1(channels - channel);
      float gathered[16];
      float ratios[16];
      read_phase();
      for (size_t lane = 0; lane < vl; ++lane)
        gathered[lane] = input[(channel + lane) * spatial + position];
      vfloat32m1_t values = __riscv_vle32_v_f32m1(gathered, vl);
      vfloat32m1_t scaled = __riscv_vfdiv_vf_f32m1(values, scale, vl);
      scaled = __riscv_vfmax_vf_f32m1(scaled, -127.0f, vl);
      scaled = __riscv_vfmin_vf_f32m1(scaled, 127.0f, vl);
      write_phase();
      __riscv_vse32_v_f32m1(ratios, scaled, vl);
      write_phase();
      for (size_t lane = 0; lane < vl; ++lane)
        packed[position * channels + channel + lane] =
            quantize_one(ratios[lane], 1.0f);
      channel += vl;
    }
#else
    read_phase();
    for (size_t channel = 0; channel < channels; ++channel) {
      float value = input[channel * spatial + position];
      packed[position * channels + channel] = quantize_one(value, scale);
    }
#endif
    write_phase();
    activation_scale[position] = scale;
  }
}

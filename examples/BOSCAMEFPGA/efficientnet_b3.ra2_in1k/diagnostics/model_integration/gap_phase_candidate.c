// Isolated GAP candidate. Production GAP sources/objects are not replaced.
// Preserve the generated Triton fixed BLOCK accumulator and ordered FP32 sum.
// Only the DDR policy granularity changes: read once per channel, output
// once per channel. No associative/reciprocal FP32 transformation is used.
//
#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
static inline void read_phase(void) {
  uintptr_t policy = UINT64_C(0x0f0f000f000f000f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
}
static inline void write_phase(void) {
  uintptr_t policy = UINT64_C(0x0f0f0f0f0f0f0f0f);
  __asm__ volatile("fence iorw,iorw; csrw 0xbc0,%0" ::"r"(policy) : "memory");
}
#else
static inline void read_phase(void) {}
static inline void write_phase(void) {}
#endif
void efficientnet_gap_phase_candidate(const float *input, float *output,
                                      size_t spatial, size_t channels,
                                      size_t block_size) {
  for (size_t channel = 0; channel < channels; ++channel) {
    float acc[128] = {0.0f};
    read_phase();
    for (size_t block = 0; block < (spatial + block_size - 1) / block_size;
         ++block)
      for (size_t lane = 0; lane < block_size; ++lane) {
        size_t pixel = block * block_size + lane;
        float value = pixel < spatial ? input[channel * spatial + pixel] : 0.0f;
        acc[lane] = acc[lane] + value;
      }
    float sum = 0.0f;
    for (size_t lane = 0; lane < block_size; ++lane)
      sum = sum + acc[lane];
    float value = sum / (float)spatial;
    write_phase();
    output[channel] = value;
  }
}

// Isolated GAP arithmetic-order candidate, not a production replacement.
// Explicitly adopts pinned Torch 2.10 CPU SumKernel AVX2 eight-lane,
// four-ILP, four-level cascade order and true float32 division.
// This differs from original Triton fixed-BLOCK reduction order.
// Read/write policy switches once per channel; scalar stack arithmetic only.
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
void efficientnet_gap_torch_order_candidate(const float *input, float *output,
                                            size_t spatial, size_t channels,
                                            size_t ignored_block_size) {
  (void)ignored_block_size;
  for (size_t channel = 0; channel < channels; ++channel) {
    const float *x = input + channel * spatial;
    float acc[4][4][8] = {0.0f};
    read_phase();
    size_t vectors = spatial / 8, groups = vectors / 4;
    size_t power = 0, value = groups ? groups - 1 : 0;
    while (value) {
      ++power;
      value >>= 1;
    }
    power /= 4;
    if (power < 4)
      power = 4;
    size_t step = (size_t)1 << power, mask = step - 1, i = 0;
    for (; i + step <= groups;) {
      for (size_t j = 0; j < step; ++j, ++i)
        for (size_t k = 0; k < 4; ++k)
          for (size_t lane = 0; lane < 8; ++lane)
            acc[0][k][lane] = acc[0][k][lane] + x[(i * 4 + k) * 8 + lane];
      for (size_t level = 1; level < 4; ++level) {
        for (size_t k = 0; k < 4; ++k)
          for (size_t lane = 0; lane < 8; ++lane) {
            acc[level][k][lane] = acc[level][k][lane] + acc[level - 1][k][lane];
            acc[level - 1][k][lane] = 0.0f;
          }
        if (i & (mask << (level * power)))
          break;
      }
    }
    for (; i < groups; ++i)
      for (size_t k = 0; k < 4; ++k)
        for (size_t lane = 0; lane < 8; ++lane)
          acc[0][k][lane] = acc[0][k][lane] + x[(i * 4 + k) * 8 + lane];
    for (size_t level = 1; level < 4; ++level)
      for (size_t k = 0; k < 4; ++k)
        for (size_t lane = 0; lane < 8; ++lane)
          acc[0][k][lane] = acc[0][k][lane] + acc[level][k][lane];
    for (size_t v = groups * 4; v < vectors; ++v)
      for (size_t lane = 0; lane < 8; ++lane)
        acc[0][0][lane] = acc[0][0][lane] + x[v * 8 + lane];
    for (size_t k = 1; k < 4; ++k)
      for (size_t lane = 0; lane < 8; ++lane)
        acc[0][0][lane] = acc[0][0][lane] + acc[0][k][lane];
    float sum = 0.0f;
    for (size_t pixel = vectors * 8; pixel < spatial; ++pixel)
      sum = sum + x[pixel];
    for (size_t lane = 0; lane < 8; ++lane)
      sum = sum + acc[0][0][lane];
    float result = sum / (float)spatial;
    write_phase();
    output[channel] = result;
  }
}

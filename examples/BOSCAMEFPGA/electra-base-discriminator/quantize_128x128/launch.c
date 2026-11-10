//
// Host / bare-metal harness for FPGA case 'quantize_128x128' (model
// 'electra-base-discriminator').
//
// Inputs: workspace buffers filled with synthetic operands, then the generated
// _mlir_ciface_* kernel entry.
//
// Outputs: UART or stdout check line from print_check(), plus max error versus
// a CPU oracle.
//
// When used: linked with support.c after AOT compile; run on NH/E6 or as
// HOST_TEST.
//

#include "support.h"
#include "tools/launch_checks.h"
#define M 128
#define K 128
extern void _mlir_ciface_kernel_quantize_128x128(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
int launch(void) {
  size_t scale_offset = vit_align64(64 + (size_t)M * K * sizeof(float) + 128);
  size_t output_offset =
      vit_align64(scale_offset + (size_t)M * sizeof(float) + 128);
  float *x = workspace(64), *s = workspace(scale_offset);
  int8_t *q = workspace(output_offset);
  vit_guard_set(x, (size_t)M * K * 4);
  vit_guard_set(s, (size_t)M * 4);
  vit_guard_set(q, (size_t)M * K);
  unsigned errors = 0;
  for (int trial = 0; trial < 2; trial++) {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < K; j++) {
        float base = (j == 0 ? 127.0f
                             : (j == 1 ? -127.0f
                                       : (float)((j * 13) % 31 - 15) * 0.25f +
                                             (j % 3) * 0.0625f));
        x[(size_t)i * K + j] = (trial == 0 && (i % 5) == 0)
                                   ? 0.0f
                                   : base * (i % 2 ? 1.137f : 1.0f);
      }
    for (int i = 0; i < M; ++i)
      s[i] = -999.0f;
    for (size_t i = 0; i < (size_t)M * K; ++i)
      q[i] = -128;
    MemRef2 X = make_2(x, M, K), Q = make_2(q, M, K);
    MemRef1 S = make_1(s, M);
    uint64_t start = nr_cycles();
    _mlir_ciface_kernel_quantize_128x128(&X, &Q, &S);
    nr_puts("cycles quantize_128x128 trial=");
    nr_hex32((uint32_t)trial);
    nr_puts(" ");
    nr_hex64(nr_cycles() - start);
    nr_puts("\r\n");
    for (int i = 0; i < M; i++) {
      float maximum = 0.0f;
      for (int j = 0; j < K; j++) {
        float v = x[(size_t)i * K + j];
        if (v < 0)
          v = -v;
        if (v > maximum)
          maximum = v;
      }
      float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
      if (!vit_f32_equal_bits(s[i], scale))
        errors++;
      for (int j = 0; j < K; j++) {
        float v = x[(size_t)i * K + j] / scale;
        int base = (int)v;
        float fraction = v - (float)base;
        int want = base + (fabsf(fraction) >= 0.5f ? (v >= 0.0f ? 1 : -1) : 0);
        if (want > 127)
          want = 127;
        if (want < -127)
          want = -127;
        if (q[(size_t)i * K + j] != (int8_t)want)
          errors++;
      }
    }
    errors += vit_guard_errors(x, (size_t)M * K * 4);
    errors += vit_guard_errors(s, (size_t)M * 4);
    errors += vit_guard_errors(q, (size_t)M * K);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < K; ++j) {
        float base = (j == 0 ? 127.0f
                             : (j == 1 ? -127.0f
                                       : (float)((j * 13) % 31 - 15) * 0.25f +
                                             (j % 3) * 0.0625f));
        float original =
            (trial == 0 && i % 5 == 0) ? 0.0f : base * (i % 2 ? 1.137f : 1.0f);
        errors += !vit_f32_equal_bits(x[(size_t)i * K + j], original);
      }
  }
  return print_check("quantize_128x128", errors, 0.0f);
}

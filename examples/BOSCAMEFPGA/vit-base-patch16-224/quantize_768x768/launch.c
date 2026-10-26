#include "support.h"
#define M 768
#define K 768
extern void _mlir_ciface_kernel_quantize_768x768(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
int launch(void) {
  size_t scale_offset = vit_align64((size_t)M * K * sizeof(float));
  size_t output_offset = vit_align64(scale_offset + (size_t)M * sizeof(float));
  float *x = workspace(0), *s = workspace(scale_offset);
  int8_t *q = workspace(output_offset);
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
    MemRef2 X = make_2(x, M, K), Q = make_2(q, M, K);
    MemRef1 S = make_1(s, M);
    uint64_t start = nr_cycles();
    _mlir_ciface_kernel_quantize_768x768(&X, &Q, &S);
    nr_puts("cycles quantize_768x768 trial=");
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
      if (!check_close(s[i], scale, 1e-6f, 1e-6f))
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
  }
  return print_check("quantize_768x768", errors, 0.0f);
}

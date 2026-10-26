#include "support.h"
#define M 196
#define N 768
extern void _mlir_ciface_kernel_dequantize_196x768(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
int launch(void) {
  size_t row_offset = vit_align64((size_t)M * N * sizeof(int32_t));
  size_t column_offset = vit_align64(row_offset + (size_t)M * sizeof(float));
  size_t output_offset = vit_align64(column_offset + (size_t)N * sizeof(float));
  int32_t *x = workspace(0);
  float *r = workspace(row_offset), *s = workspace(column_offset),
        *y = workspace(output_offset);
  for (int i = 0; i < M; i++)
    r[i] = (i % 17 + 1) * 0.00390625f;
  for (int j = 0; j < N; j++)
    s[j] = (j % 31 + 1) * 0.001953125f;
  for (size_t i = 0; i < (size_t)M * N; i++)
    x[i] = ((int)i % 1023 - 511) * 7919;
  MemRef2 X = make_2(x, M, N), Y = make_2(y, M, N);
  MemRef1 R = make_1(r, M), S = make_1(s, N);
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_dequantize_196x768(&X, &R, &S, &Y);
  nr_puts("cycles dequantize_196x768: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  float max_error = 0.0f;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float want = (float)x[(size_t)i * N + j] * r[i] * s[j],
            dlt = y[(size_t)i * N + j] - want;
      if (dlt < 0)
        dlt = -dlt;
      if (dlt > max_error)
        max_error = dlt;
      if (!check_close(y[(size_t)i * N + j], want, 1e-5f, 1e-6f))
        errors++;
    }
  return print_check("dequantize_196x768", errors, max_error);
}

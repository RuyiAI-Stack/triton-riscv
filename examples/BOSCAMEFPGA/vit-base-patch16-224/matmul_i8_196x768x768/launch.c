#include "support.h"
#define M 196
#define N 768
#define K 768
extern void _mlir_ciface_kernel_matmul_i8_196x768x768(MemRef2 *, MemRef2 *,
                                                      MemRef2 *);
int launch(void) {
  int8_t *a = workspace(0), *b = workspace((size_t)M * K + 64);
  int32_t *c = workspace((size_t)M * K + (size_t)N * K + 128);
  for (int i = 0; i < M; i++)
    for (int q = 0; q < K; q++)
      a[(size_t)i * K + q] = (int8_t)(((i * 13 + q * 7) % 31) - 15);
  for (int j = 0; j < N; j++)
    for (int q = 0; q < K; q++)
      b[(size_t)j * K + q] = (int8_t)(((j * 11 + q * 5) % 29) - 14);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      c[(size_t)i * N + j] = (i * 3 + j) % 7 - 3;
  MemRef2 A = make_2(a, M, K), B = make_2(b, N, K), C = make_2(c, M, N);
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_matmul_i8_196x768x768(&A, &B, &C);
  nr_puts("cycles matmul_i8_196x768x768: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  int32_t max_error = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      int32_t want = (i * 3 + j) % 7 - 3;
      for (int q = 0; q < K; q++)
        want += (int32_t)a[(size_t)i * K + q] * (int32_t)b[(size_t)j * K + q];
      int32_t got = c[(size_t)i * N + j], dlt = got - want;
      if (dlt < 0)
        dlt = -dlt;
      if (dlt > max_error)
        max_error = dlt;
      if (got != want) {
        if (errors < 4) {
          nr_puts("mismatch ");
          nr_hex32(i);
          nr_puts(" ");
          nr_hex32(j);
          nr_puts("\r\n");
        }
        errors++;
      }
    }
  return print_check("matmul_i8_196x768x768", errors, (float)max_error);
}

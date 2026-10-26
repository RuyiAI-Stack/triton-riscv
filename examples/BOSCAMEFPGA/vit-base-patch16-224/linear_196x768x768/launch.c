#include "support.h"

static void compare(float actual, float expected, float atol, float rtol,
                    int *errors, float *maximum) {
  if (actual == expected)
    return;
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected)) {
    ++*errors;
    *maximum = __builtin_inff();
    return;
  }
  if (!check_close(actual, expected, atol, rtol))
    ++*errors;
  float delta = actual - expected;
  if (delta < 0)
    delta = -delta;
  if (delta > *maximum)
    *maximum = delta;
}

extern void _mlir_ciface_kernel_linear_196x768x768(MemRef2 *, MemRef2 *,
                                                   MemRef2 *);

int launch(void) {
  const int M = 196, N = 768, K = 768;
  size_t a_bytes = (size_t)M * K * sizeof(float);
  size_t b_bytes = (size_t)N * K * sizeof(float);
  size_t aoff = 0;
  size_t boff = (a_bytes + 63) & ~(size_t)63;
  size_t coff = (boff + b_bytes + 63) & ~(size_t)63;
  float *a = (float *)workspace(aoff);
  float *b = (float *)workspace(boff);
  float *c = (float *)workspace(coff);
  for (int i = 0; i < M * K; ++i)
    a[i] = (float)((i * 13) % 29 - 14) * 0.0078125f;
  for (int i = 0; i < N * K; ++i)
    b[i] = (float)((i * 7) % 23 - 11) * 0.01171875f;
  for (int i = 0; i < M * N; ++i)
    c[i] = 0.0f;

  MemRef2 A = make_2(a, M, K);
  MemRef2 B = make_2(b, N, K);
  MemRef2 C = make_2(c, M, N);
  _mlir_ciface_kernel_linear_196x768x768(&A, &B, &C);

  int errors = 0;
  float maximum = 0.0f;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float want = 0.0f;
      for (int k = 0; k < K; ++k)
        want += a[i * K + k] * b[j * K + k];
      compare(c[i * N + j], want, 3e-3f, 3e-3f, &errors, &maximum);
    }
  }
  return print_check("linear_196x768x768", errors, maximum);
}

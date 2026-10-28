#include "support.h"
#ifdef HOST_TEST
static float oracle_exp(float x) { return (float)exp((double)x); }
static float oracle_sqrt(float x) { return (float)sqrt((double)x); }
static float oracle_erf(float x) { return (float)erf((double)x); }
#else
static float oracle_exp(float x) { return expf(x); }
static float oracle_sqrt(float x) { return sqrtf(x); }
static float oracle_erf(float x) { return erff(x); }
#endif
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
  float d = actual - expected;
  if (d < 0)
    d = -d;
  if (d > *maximum)
    *maximum = d;
}
extern void _mlir_ciface_kernel_layernorm_128x768(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);
int launch(void) {
  const int ROWS = 128, WIDTH = 768, COUNT = ROWS * WIDTH;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)COUNT * 4u);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)WIDTH * 4u);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)WIDTH * 4u);
  float *o = (float *)workspace(off);
  for (int i = 0; i < COUNT; ++i)
    x[i] = (float)((i * 5) % 37 - 18) * 0.015625f;
  for (int i = 0; i < WIDTH; ++i) {
    w[i] = 1.0f + (i % 11) * 0.01f;
    b[i] = (i % 17 - 8) * 0.00390625f;
  }
  MemRef2 X = make_2(x, ROWS, WIDTH), O = make_2(o, ROWS, WIDTH);
  MemRef1 W = make_1(w, WIDTH), B = make_1(b, WIDTH);
  _mlir_ciface_kernel_layernorm_128x768(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int r = 0; r < ROWS; ++r) {
    float mean = 0, var = 0;
    for (int j = 0; j < WIDTH; ++j)
      mean += x[r * WIDTH + j];
    mean /= WIDTH;
    for (int j = 0; j < WIDTH; ++j) {
      float d = x[r * WIDTH + j] - mean;
      var += d * d;
    }
    var /= WIDTH;
    for (int j = 0; j < WIDTH; ++j) {
      float want = (x[r * WIDTH + j] - mean) / oracle_sqrt(var + 1e-12f) * w[j] + b[j];
      compare(o[r * WIDTH + j], want, 4e-4f, 4e-4f, &errors, &maximum);
    }
  }
  return print_check("layernorm_128x768", errors, maximum);
}

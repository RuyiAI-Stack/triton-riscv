//
// Host / bare-metal harness for FPGA case 'quantize_384x1536' (model
// 'gte-small').
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
#ifdef HOST_TEST
static double oracle_exp(double x) { return exp(x); }
static double oracle_erf(double x) { return erf(x); }
static double oracle_sqrt(double x) { return sqrt(x); }
#else
extern float expf(float), erff(float), sqrtf(float), fabsf(float);
static double oracle_exp(double x) { return expf((float)x); }
static double oracle_erf(double x) { return erff((float)x); }
static double oracle_sqrt(double x) { return sqrtf((float)x); }
#endif
static void compare(float got, float want, float atol, float rtol,
                    unsigned *errors, float *maximum) {
  if (got == want)
    return;
  if (!__builtin_isfinite(got) || !__builtin_isfinite(want)) {
    ++*errors;
    *maximum = __builtin_inff();
    return;
  }
  float d = got - want;
  if (d < 0)
    d = -d;
  if (d > *maximum)
    *maximum = d;
  if (!check_close(got, want, atol, rtol))
    ++*errors;
}
static void check_integer(int32_t got, int32_t want, unsigned *errors,
                          float *maximum) {
  if (got != want) {
    ++*errors;
    float d = (float)((int64_t)got - want);
    if (d < 0)
      d = -d;
    if (d > *maximum)
      *maximum = d;
  }
}

extern void _mlir_ciface_kernel_quantize_384x1536(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
int launch(void) {
  float *a0 = workspace(0u);
  for (int i = 0; i < 589824; ++i)
    a0[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef2 d0 = make_2(a0, 384, 1536);
  int8_t *a1 = workspace(2359296u);
  for (int i = 0; i < 589824; ++i)
    a1[i] = -77;
  MemRef2 d1 = make_2(a1, 384, 1536);
  float *a2 = workspace(2949120u);
  for (int i = 0; i < 384; ++i)
    a2[i] = -77;
  MemRef1 d2 = make_1(a2, 384);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_quantize_384x1536(&d0, &d1, &d2);
  nr_puts("cycles quantize_384x1536: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int r = 0; r < 384; ++r) {
    float max = 0;
    for (int j = 0; j < 1536; ++j) {
      float v = fabsf(a0[r * 1536 + j]);
      if (v > max)
        max = v;
    }
    float scale = max == 0 ? 1 : max / 127;
    compare(a2[r], scale, 1e-6f, 1e-6f, &errors, &maximum);
    for (int j = 0; j < 1536; ++j) {
      float v = a0[r * 1536 + j] / scale;
      int base = (int)v;
      float fraction = v - base;
      int want = base + (fabsf(fraction) >= 0.5f ? (v >= 0 ? 1 : -1) : 0);
      if (want > 127)
        want = 127;
      if (want < -127)
        want = -127;
      check_integer(a1[r * 1536 + j], want, &errors, &maximum);
    }
  }
  return print_check("quantize_384x1536", errors, maximum);
}

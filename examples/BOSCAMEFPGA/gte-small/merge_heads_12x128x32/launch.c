//
// Host / bare-metal harness for FPGA case 'merge_heads_12x128x32' (model
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

extern void _mlir_ciface_kernel_merge_heads_12x128x32(MemRef3 *, MemRef2 *);
int launch(void) {
  float *a0 = workspace(0u);
  for (int i = 0; i < 49152; ++i)
    a0[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef3 d0 = make_3(a0, 12, 128, 32);
  float *a1 = workspace(196608u);
  for (int i = 0; i < 49152; ++i)
    a1[i] = -77;
  MemRef2 d1 = make_2(a1, 128, 384);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_merge_heads_12x128x32(&d0, &d1);
  nr_puts("cycles merge_heads_12x128x32: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int h = 0; h < 12; ++h)
    for (int r = 0; r < 128; ++r)
      for (int j = 0; j < 32; ++j)
        compare(a1[r * 384 + h * 32 + j], a0[(h * 128 + r) * 32 + j], 0, 0,
                &errors, &maximum);
  return print_check("merge_heads_12x128x32", errors, maximum);
}

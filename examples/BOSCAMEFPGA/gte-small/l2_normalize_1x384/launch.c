//
// Host / bare-metal harness for FPGA case 'l2_normalize_1x384' (model
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

extern void _mlir_ciface_kernel_l2_normalize_1x384(MemRef1 *, MemRef1 *);
int launch(void) {
  float *a0 = workspace(0u);
  for (int i = 0; i < 384; ++i)
    a0[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef1 d0 = make_1(a0, 384);
  float *a1 = workspace(1536u);
  for (int i = 0; i < 384; ++i)
    a1[i] = -77;
  MemRef1 d1 = make_1(a1, 384);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_l2_normalize_1x384(&d0, &d1);
  nr_puts("cycles l2_normalize_1x384: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  double sq = 0;
  for (int j = 0; j < 384; ++j)
    sq += a0[j] * a0[j];
  double norm = oracle_sqrt(sq);
  if (norm < 1e-12)
    norm = 1e-12;
  for (int j = 0; j < 384; ++j)
    compare(a1[j], (float)(a0[j] / norm), 4e-5f, 4e-5f, &errors, &maximum);
  return print_check("l2_normalize_1x384", errors, maximum);
}

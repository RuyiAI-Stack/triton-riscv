//
// Host / bare-metal harness for FPGA case 'embedding_lookup_128x384' (model
// 'e5-small-v2').
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

extern void _mlir_ciface_kernel_embedding_lookup_128x384(MemRef1 *, MemRef2 *,
                                                         MemRef2 *);
int launch(void) {
  int32_t *a0 = workspace(0u);
  for (int i = 0; i < 128; ++i)
    a0[i] = (i == 127 ? 30521 : (i * 227) % 30522);
  MemRef1 d0 = make_1(a0, 128);
  float *a1 = workspace(512u);
  for (int i = 0; i < 11720448; ++i)
    a1[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef2 d1 = make_2(a1, 30522, 384);
  float *a2 = workspace(46882304u);
  for (int i = 0; i < 49152; ++i)
    a2[i] = -77;
  MemRef2 d2 = make_2(a2, 128, 384);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_embedding_lookup_128x384(&d0, &d1, &d2);
  nr_puts("cycles embedding_lookup_128x384: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int r = 0; r < 128; ++r)
    for (int j = 0; j < 384; ++j)
      compare(a2[r * 384 + j], a1[a0[r] * 384 + j], 0, 0, &errors, &maximum);
  return print_check("embedding_lookup_128x384", errors, maximum);
}

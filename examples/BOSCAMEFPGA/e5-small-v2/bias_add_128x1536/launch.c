//
// Host / bare-metal harness for FPGA case 'bias_add_128x1536' (model
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

extern void _mlir_ciface_kernel_bias_add_128x1536(MemRef2 *, MemRef1 *,
                                                  MemRef2 *);
int launch(void) {
  float *a0 = workspace(0u);
  for (int i = 0; i < 196608; ++i)
    a0[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef2 d0 = make_2(a0, 128, 1536);
  float *a1 = workspace(786432u);
  for (int i = 0; i < 1536; ++i)
    a1[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef1 d1 = make_1(a1, 1536);
  float *a2 = workspace(792576u);
  for (int i = 0; i < 196608; ++i)
    a2[i] = -77;
  MemRef2 d2 = make_2(a2, 128, 1536);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_bias_add_128x1536(&d0, &d1, &d2);
  nr_puts("cycles bias_add_128x1536: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int r = 0; r < 128; ++r)
    for (int j = 0; j < 1536; ++j)
      compare(a2[r * 1536 + j], a0[r * 1536 + j] + a1[j], 0, 0, &errors,
              &maximum);
  return print_check("bias_add_128x1536", errors, maximum);
}

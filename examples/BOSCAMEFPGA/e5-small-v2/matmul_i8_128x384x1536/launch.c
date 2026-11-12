//
// Host / bare-metal harness for FPGA case 'matmul_i8_128x384x1536' (model
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

extern void _mlir_ciface_kernel_matmul_i8_128x384x1536(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
int launch(void) {
  int8_t *a0 = workspace(0u);
  for (int i = 0; i < 196608; ++i)
    a0[i] = (int8_t)((i * 13) % 31 - 15);
  MemRef2 d0 = make_2(a0, 128, 1536);
  int8_t *a1 = workspace(196608u);
  for (int i = 0; i < 589824; ++i)
    a1[i] = (int8_t)((i * 13) % 31 - 15);
  MemRef2 d1 = make_2(a1, 384, 1536);
  int32_t *a2 = workspace(786432u);
  for (int i = 0; i < 49152; ++i)
    a2[i] = -77;
  MemRef2 d2 = make_2(a2, 128, 384);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_matmul_i8_128x384x1536(&d0, &d1, &d2);
  nr_puts("cycles matmul_i8_128x384x1536: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 384; ++j) {
      int32_t want = 0;
      for (int q = 0; q < 1536; ++q)
        want += (int32_t)a0[i * 1536 + q] * (int32_t)a1[j * 1536 + q];
      check_integer(a2[i * 384 + j], want, &errors, &maximum);
    }
  return print_check("matmul_i8_128x384x1536", errors, maximum);
}

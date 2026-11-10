//
// Host / bare-metal harness for FPGA case 'embedding_add_128x384' (model
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

extern void _mlir_ciface_kernel_embedding_add_128x384(MemRef2 *, MemRef1 *,
                                                      MemRef1 *, MemRef2 *,
                                                      MemRef2 *, MemRef2 *);
int launch(void) {
  float *a0 = workspace(0u);
  for (int i = 0; i < 49152; ++i)
    a0[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef2 d0 = make_2(a0, 128, 384);
  int32_t *a1 = workspace(196608u);
  for (int i = 0; i < 128; ++i)
    a1[i] = i;
  MemRef1 d1 = make_1(a1, 128);
  int32_t *a2 = workspace(197120u);
  for (int i = 0; i < 128; ++i)
    a2[i] = (i / 17) % 2;
  MemRef1 d2 = make_1(a2, 128);
  float *a3 = workspace(197632u);
  for (int i = 0; i < 196608; ++i)
    a3[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef2 d3 = make_2(a3, 512, 384);
  float *a4 = workspace(984064u);
  for (int i = 0; i < 768; ++i)
    a4[i] = (float)((i * 17) % 67 - 33) * 0.015625f;
  MemRef2 d4 = make_2(a4, 2, 384);
  float *a5 = workspace(987136u);
  for (int i = 0; i < 49152; ++i)
    a5[i] = -77;
  MemRef2 d5 = make_2(a5, 128, 384);
#ifndef HOST_TEST
  ame_fence();
#endif
  uint64_t begin = nr_cycles();
  _mlir_ciface_kernel_embedding_add_128x384(&d0, &d1, &d2, &d3, &d4, &d5);
  nr_puts("cycles embedding_add_128x384: ");
  nr_hex64(nr_cycles() - begin);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int r = 0; r < 128; ++r)
    for (int j = 0; j < 384; ++j)
      compare(a5[r * 384 + j],
              (a0[r * 384 + j] + a3[a1[r] * 384 + j]) + a4[a2[r] * 384 + j], 0,
              0, &errors, &maximum);
  return print_check("embedding_add_128x384", errors, maximum);
}

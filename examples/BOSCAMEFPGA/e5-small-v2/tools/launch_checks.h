// Bounded standalone validation helpers; never used in production adapters.
#ifndef VIT_LAUNCH_CHECKS_H
#define VIT_LAUNCH_CHECKS_H
#include "support.h"
static void vit_guard_set(void *data, size_t bytes) {
  unsigned char *p = data;
  for (int i = 0; i < 64; ++i) {
    p[i - 64] = 0xa5;
    p[bytes + i] = 0x5a;
  }
}
static unsigned vit_guard_errors(const void *data, size_t bytes) {
  const unsigned char *p = data;
  unsigned errors = 0;
  for (int i = 0; i < 64; ++i) {
    errors += p[i - 64] != 0xa5;
    errors += p[bytes + i] != 0x5a;
  }
  return errors;
}
static int vit_f32_equal_bits(float a, float b) {
  union {
    float f;
    uint32_t u;
  } x = {a}, y = {b};
  return x.u == y.u;
}
#endif

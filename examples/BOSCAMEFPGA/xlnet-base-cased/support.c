#include "support.h"

#ifndef HOST_TEST
#error                                                                         \
    "support.c implements the standalone host oracles; compile with HOST_TEST"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned char arena[VIT_WORKSPACE_BYTES] __attribute__((aligned(64)));

void nr_copy_bytes(void *destination, const void *source, size_t bytes) {
  memcpy(destination, source, bytes);
}
void nr_puts(const char *s) {
  fputs(s, stdout);
  fflush(stdout);
}
void nr_hex32(uint32_t x) { printf("%08x", x); }
void nr_hex64(uint64_t x) { printf("%016llx", (unsigned long long)x); }
uint64_t nr_cycles(void) { return (uint64_t)clock(); }

void *workspace(size_t byte_offset) {
  if (byte_offset >= sizeof(arena)) {
    fputs("EfficientNet operator workspace offset exceeds VIT_WORKSPACE_BYTES\n",
          stderr);
    abort();
  }
  return arena + byte_offset;
}

int check_close(float actual, float expected, float atol, float rtol) {
  union {
    float f;
    uint32_t u;
  } a = {actual}, b = {expected};
  if ((a.u & 0x7f800000u) == 0x7f800000u || (b.u & 0x7f800000u) == 0x7f800000u)
    return actual == expected;
  float difference = actual - expected;
  if (difference < 0)
    difference = -difference;
  float magnitude = expected < 0 ? -expected : expected;
  return difference <= atol + rtol * magnitude;
}

int print_check(const char *name, unsigned errors, float max_error) {
  nr_puts("verify ");
  nr_puts(name);
  nr_puts(errors ? ": FAIL errors=" : ": PASS errors=");
  nr_hex32(errors);
  nr_puts(" max_abs_error_f32_bits=");
  union {
    float f;
    uint32_t u;
  } value = {max_error};
  nr_hex32(value.u);
  nr_puts("\r\n");
  return errors != 0;
}

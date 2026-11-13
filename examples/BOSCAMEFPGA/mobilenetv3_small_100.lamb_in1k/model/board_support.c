// NH application support; the shared runtime owns UART initialization and
// the scheduler START/RESULT protocol. The model owns only its computation.
#include "support.h"
#include "uart.h"

static unsigned char arena[2u * 1024u * 1024u]
    __attribute__((section(".workspace"), aligned(64)));

void *workspace(size_t offset) {
  if (offset >= sizeof arena)
    __builtin_trap();
  return arena + offset;
}

void *memset(void *destination, int value, size_t count) {
  unsigned char *out = destination;
  for (size_t i = 0; i < count; ++i)
    out[i] = (unsigned char)value;
  return destination;
}

void *memcpy(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  for (size_t i = 0; i < count; ++i)
    out[i] = in[i];
  return destination;
}

void nr_copy_bytes(void *destination, const void *source, size_t count) {
  memcpy(destination, source, count);
}

uint64_t nr_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

void nr_puts(const char *s) { print_uart(s); }
void nr_hex32(uint32_t value) { print_uart_hex32(value); }
void nr_hex64(uint64_t value) { print_uart_hex64(value); }
float fabsf(float x) { return x < 0.0f ? -x : x; }

int check_close(float actual, float expected, float atol, float rtol) {
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected))
    return actual == expected;
  return fabsf(actual - expected) <= atol + rtol * fabsf(expected);
}

int print_check(const char *name, unsigned errors, float max_error) {
  union {
    float f;
    uint32_t u;
  } value = {max_error};
  nr_puts("verify ");
  nr_puts(name);
  nr_puts(errors ? ": FAIL errors=" : ": PASS errors=");
  nr_hex32(errors);
  nr_puts(" max_abs_error_f32_bits=");
  nr_hex32(value.u);
  nr_puts("\r\n");
  return errors != 0;
}

int main(void) {
  nr_puts("mobilenetv3_small_100.lamb_in1k complete float32 inference\r\n");
  return launch();
}

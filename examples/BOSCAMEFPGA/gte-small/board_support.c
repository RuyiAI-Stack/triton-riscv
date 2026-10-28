#include "support.h"
#include "nh_e6_platform.h"
#include "uart.h"

static unsigned char arena[VIT_WORKSPACE_BYTES]
    __attribute__((section(".workspace"), aligned(64)));
static char log_buffer[4096];
static unsigned log_used;

void *workspace(size_t byte_offset) {
  if (byte_offset >= sizeof arena)
    __builtin_trap();
  return arena + byte_offset;
}
void *memset(void *destination, int value, size_t count) {
  unsigned char *out = destination;
  for (size_t i = 0; i < count; ++i)
    out[i] = (unsigned char)value;
  return destination;
}
void nr_copy_bytes(void *d, const void *s, size_t n) {
  for (size_t i = 0; i < n; ++i)
    ((unsigned char *)d)[i] = ((const unsigned char *)s)[i];
}
uint64_t nr_cycles(void) {
  uint64_t c;
  __asm__ volatile("rdcycle %0" : "=r"(c));
  return c;
}
float fabsf(float x) { return x < 0.0f ? -x : x; }
static void put(const char *s) {
  while (*s && log_used + 1 < sizeof log_buffer)
    log_buffer[log_used++] = *s++;
}
void nr_puts(const char *s) { put(s); }
void nr_hex32(uint32_t x) {
  char t[9];
  for (int i = 7; i >= 0; --i) {
    unsigned d = x & 15u;
    t[i] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
    x >>= 4;
  }
  t[8] = 0;
  put(t);
}
void nr_hex64(uint64_t x) {
  nr_hex32((uint32_t)(x >> 32));
  nr_hex32((uint32_t)x);
}
int check_close(float actual, float expected, float atol, float rtol) {
  union {
    float f;
    uint32_t u;
  } a = {actual}, b = {expected};
  if ((a.u & 0x7f800000u) == 0x7f800000u ||
      (b.u & 0x7f800000u) == 0x7f800000u)
    return actual == expected;
  float d = actual - expected;
  if (d < 0)
    d = -d;
  float m = expected < 0 ? -expected : expected;
  return d <= atol + rtol * m;
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
int e6_main(void) { return launch(); }
const char *nh_e6_app_tag(void) { return "effnet"; }
void nh_e6_app_report(void) {
  nh_e6_print_tag_prefix();
  uart_puts(log_used ? log_buffer : "(no launch log)\r\n");
}

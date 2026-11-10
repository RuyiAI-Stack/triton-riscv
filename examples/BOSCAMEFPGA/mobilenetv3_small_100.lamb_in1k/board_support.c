//
// C support runtime for BOSCAME FPGA model 'mobilenetv3_small_100.lamb_in1k'.
//
// Inputs: workspace size, operand pointers, and UART/status hooks shared by
// every launch.c in this model tree.
//
// Outputs: aligned workspace(), MemRef helpers, print_check(), and numeric
// oracles used on host and on the board.
//
// When used: linked into each case image via common.mk / board.mk after the
// kernel object is produced.
//
#include "support.h"
#ifndef BOSC_NH_ONLY
#include "nh_e6_platform.h"
#endif
#include "uart.h"

static unsigned char arena[VIT_WORKSPACE_BYTES]
    __attribute__((section(".workspace"), aligned(64)));
#ifndef BOSC_NH_ONLY
static char log_buffer[4096] __attribute__((aligned(64)));
static unsigned log_used;
#endif

// Keep final validation separate from the invocation control line, as in the
// ViT harness. Complete volatile cache lines preserve the layout under LTO.
typedef struct __attribute__((aligned(64))) {
  const char *name;
  uint64_t begin, cycles;
  unsigned started, completed, returned;
  unsigned char reserved[28];
} InvocationState;
typedef struct __attribute__((aligned(64))) {
  unsigned executed, checked, errors, maximum_bits;
  unsigned char reserved[48];
} ValidationState;
static volatile InvocationState invocation;
static volatile ValidationState validation;
_Static_assert(sizeof(InvocationState) == 64, "invocation must fill its line");
_Static_assert(sizeof(ValidationState) == 64, "validation must fill its line");

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
  __asm__ volatile("rdcycle %0" : "=r"(c)::"memory");
  return c;
}
float fabsf(float x) { return x < 0.0f ? -x : x; }
static void put(const char *s) {
#ifdef BOSC_NH_ONLY
  print_uart(s);
#else
  while (*s && log_used + 1 < sizeof log_buffer)
    log_buffer[log_used++] = *s++;
#endif
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
static void dec32(unsigned value) {
  char text[11];
  unsigned end = sizeof text - 1;
  text[end] = 0;
  do {
    text[--end] = (char)('0' + value % 10);
    value /= 10;
  } while (value);
  nr_puts(text + end);
}
int check_close(float actual, float expected, float atol, float rtol) {
  union {
    float f;
    uint32_t u;
  } a = {actual}, b = {expected};
  if ((a.u & 0x7f800000u) == 0x7f800000u || (b.u & 0x7f800000u) == 0x7f800000u)
    return actual == expected;
  float d = actual - expected;
  if (d < 0)
    d = -d;
  float m = expected < 0 ? -expected : expected;
  return d <= atol + rtol * m;
}
int print_check_count(const char *name, unsigned checked, unsigned errors,
                      float max_error) {
#ifndef BOSC_NH_ONLY
  nh_e6_cache_writes();
#endif
  union {
    float f;
    uint32_t u;
  } value = {max_error};
  validation.checked = checked;
  validation.errors = errors;
  validation.maximum_bits = value.u;
  validation.executed = 1;
  nr_puts("verify ");
  nr_puts(name);
  nr_puts(errors ? ": FAIL errors=" : ": PASS errors=");
  nr_hex32(errors);
  nr_puts(" max_abs_error_f32_bits=");
  nr_hex32(value.u);
  nr_puts(" checked=");
  nr_hex32(checked);
  nr_puts(" validation_executed=1\r\n");
  return errors != 0;
}
int print_check(const char *name, unsigned errors, float max_error) {
  // Legacy callers have no checked count; the final verdict rejects a run
  // without explicit coverage. Current standalone launchers all pass it.
  return print_check_count(name, 0, errors, max_error);
}
static void op_label(const char *name) {
  nr_puts("[mobilenet-op] 1/1 symbol=_mlir_ciface_kernel_");
  nr_puts(name);
  nr_puts(" case=");
  nr_puts(name);
}
void nr_kernel_begin(const char *name) {
#ifndef BOSC_NH_ONLY
  nh_e6_cache_writes();
#endif
  invocation.name = name;
  ++invocation.started;
  op_label(name);
  nr_puts(" BEGIN\r\n");
#ifndef BOSC_NH_ONLY
  nh_e6_progress(0x1001u);
#endif
  // UART/progress time is outside the measured adapter interval.
  invocation.begin = nr_cycles();
}
void nr_kernel_end(const char *name) {
  uint64_t end = nr_cycles();
  invocation.cycles += end - invocation.begin;
  ++invocation.completed;
  invocation.returned = 1;
  op_label(name);
  nr_puts(" END cycles=0x");
  nr_hex64(end - invocation.begin);
  nr_puts("\r\n");
#ifndef BOSC_NH_ONLY
  nh_e6_progress(0x2001u);
#endif
}
static void report_state(const char *phase) {
  nr_puts("[mobilenet-operator] ");
  nr_puts(phase);
  nr_puts(" case=");
  nr_puts(invocation.name ? invocation.name : "none");
  nr_puts(" calls_started=");
  dec32(invocation.started);
  nr_puts(" calls_completed=");
  dec32(invocation.completed);
  nr_puts(" kernel_returned=");
  dec32(invocation.returned);
  nr_puts(" validation_executed=");
  dec32(validation.executed);
  nr_puts(" checked=");
  dec32(validation.checked);
  nr_puts(" errors=");
  dec32(validation.errors);
  nr_puts(" max_abs_error_f32_bits=0x");
  nr_hex32(validation.maximum_bits);
  nr_puts(" kernel_cycles=0x");
  nr_hex64(invocation.cycles);
  nr_puts("\r\n");
}
void nr_report_failure(void) { report_state("TRAP"); }
static int run_operator(void) {
#ifndef BOSC_NH_ONLY
  nh_e6_cache_writes();
#endif
  nr_puts("[mobilenet-operator] START\r\n");
  int status = launch();
  if (invocation.started != 1 || invocation.completed != 1 ||
      !invocation.returned || !validation.executed || !validation.checked ||
      validation.errors)
    status = 1;
  report_state(status ? "FAIL" : "PASS");
  return status;
}
#ifdef BOSC_NH_ONLY
int main(void) { return run_operator(); }
// Scalar kernels need only a memory fence, never an AME transaction.
void ame_fence(void) { __asm__ volatile("fence rw, rw" ::: "memory"); }
#else
int e6_main(void) { return run_operator(); }
const char *nh_e6_app_tag(void) { return "mobilenet-operator"; }
void nh_e6_app_report(void) {
  nh_e6_print_tag_prefix();
  uart_puts(log_used ? log_buffer : "(no launch log)\r\n");
}
#endif

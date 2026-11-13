//
// C support runtime for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.
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
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "support.h"
#include "uart.h"

static unsigned char arena[VIT_WORKSPACE_BYTES]
    __attribute__((section(".workspace"), aligned(64)));
static char log_buffer[4096];
static unsigned log_used;
static volatile unsigned launch_returned;
static volatile int launch_status;
static volatile uint64_t launch_cycles;

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
void *memcpy(void *d, const void *s, size_t n) {
  for (size_t i = 0; i < n; ++i)
    ((unsigned char *)d)[i] = ((const unsigned char *)s)[i];
  return d;
}
void nr_copy_bytes(void *d, const void *s, size_t n) { memcpy(d, s, n); }
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
  if ((a.u & 0x7f800000u) == 0x7f800000u || (b.u & 0x7f800000u) == 0x7f800000u)
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
int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  nh_e6_progress(0xeff1u);
  uint64_t begin = nr_cycles();
  int status = launch();
  launch_cycles = nr_cycles() - begin;
  launch_status = status;
  launch_returned = 1;
  nh_e6_progress(0xeff2u);
  return status;
}
const char *nh_e6_app_tag(void) { return "effnet"; }

// Only the platform object's init_uart call is redirected to this hook.
void effnet_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}

void nh_e6_app_report(void) {
  nh_e6_print_tag_prefix();
  uart_puts(log_used ? log_buffer : "(no launch log)\r\n");
  nh_e6_print_tag_prefix();
  print_uart("launch_returned=");
  print_uart_dec32(launch_returned);
  print_uart(" launch_cycles=0x");
  print_uart_hex64(launch_cycles);
  print_uart(" launch_exit_code=0x");
  print_uart_hex32((uint32_t)launch_status);
  print_uart("\r\n");
  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
  if (done == NH_E6_DONE_TRAP) {
    print_uart(
        "@@FPGA_RESULT {\"version\":1,\"status\":\"trap\",\"mcause\":\"0x");
    print_uart_hex64(e6_diag_mcause);
    print_uart("\",\"mepc\":\"0x");
    print_uart_hex64(e6_diag_mepc);
    print_uart("\",\"mtval\":\"0x");
    print_uart_hex64(e6_diag_mtval);
    print_uart("\"}\r\n");
  } else {
    int ok = done == NH_E6_DONE_PASS && launch_returned && launch_status == 0;
    print_uart(
        ok ? "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
           : "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
  }
}

// NH/E6 board harness, following the standalone INT8 matmul example.
// The existing launch.c initializes and checks every one of the 768 outputs.
// Only the complete typed kernel adapter (six programs) is timed.
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "support.h"
#include "uart.h"

static unsigned char arena[VIT_WORKSPACE_BYTES]
    __attribute__((section(".dma_source"), aligned(64), used));
static volatile uint64_t kernel_cycles;
static volatile unsigned checked, errors, max_error_bits, kernel_returned;

extern int launch(void);
extern void vit_add_implementation(MemRef2 *, MemRef2 *, MemRef2 *);

uint64_t nr_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

void *workspace(size_t offset) {
  if (offset >= sizeof(arena))
    __builtin_trap();
  return arena + offset;
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

int print_check(const char *name, unsigned count, float maximum) {
  (void)name;
  union {
    float f;
    uint32_t u;
  } value = {maximum};
  checked = 768;
  errors = count;
  max_error_bits = value.u;
  return count != 0;
}

void _mlir_ciface_kernel_add_1x768(MemRef2 *x, MemRef2 *y, MemRef2 *out) {
  nh_e6_progress(0x1001);
  nh_e6_cache_writes();
  uint64_t begin = nr_cycles();
  vit_add_implementation(x, y, out);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  uint64_t end = nr_cycles();
  kernel_cycles = end - begin;
  kernel_returned = 1;
  nh_e6_progress(0x1002);
  nh_e6_cache_uncached_loads();
}

int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  return launch();
}

const char *nh_e6_app_tag(void) { return "add_1x768"; }

// The scheduler requires UART v1 records, in addition to human-readable PASS.
// Compile the shared platform with init_uart renamed to this wrapper.
void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}

void nh_e6_app_report(void) {
  print_uart("[add_1x768] kernel_returned=");
  print_uart_dec32(kernel_returned);
  print_uart(" checked=");
  print_uart_dec32(checked);
  print_uart(" errors=");
  print_uart_dec32(errors);
  print_uart(" max_abs_error_f32_bits=0x");
  print_uart_hex32(max_error_bits);
  print_uart(" kernel_cycles=0x");
  print_uart_hex64(kernel_cycles);
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
    int ok =
        done == NH_E6_DONE_PASS && kernel_returned && checked == 768 && !errors;
    print_uart(
        ok ? "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
           : "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
  }
}

// Shared UART v1 transport hooks for the BOSCAMEFPGA single-hart bare runtime.
// Link with an application; the runtime calls these after UART setup and
// after main/application reports. This protocol is for trusted team programs.
//
#include "uart.h"
#include <stdint.h>

void bare_runtime_report_start(void) {
  print_uart("\r\n@@FPGA_START {\"version\":1}\r\n");
}

void bare_runtime_report_exit(int status) {
  char digits[10];
  unsigned count = 0;
  uint32_t value = (uint32_t)status;
  print_uart("\r\n@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":");
  if (status < 0) {
    print_uart("-");
    value = 0u - value;
  }
  do {
    digits[count++] = (char)('0' + value % 10);
    value /= 10;
  } while (value);
  while (count) {
    char text[2] = {digits[--count], 0};
    print_uart(text);
  }
  print_uart("}\r\n");
}

void bare_runtime_report_trap(uintptr_t cause, uintptr_t pc, uintptr_t value) {
  print_uart(
      "\r\n@@FPGA_RESULT {\"version\":1,\"status\":\"trap\",\"mcause\":\"0x");
  print_uart_addr(cause);
  print_uart("\",\"mepc\":\"0x");
  print_uart_addr(pc);
  print_uart("\",\"mtval\":\"0x");
  print_uart_addr(value);
  print_uart("\"}\r\n");
}
